"""
LinUCB 上下文 Bandit 路由器
符合宪章 v5.0.0 第14章、第15章要求
- Sherman-Morrison 增量更新 (O(d²) 复杂度)
- 自动矩阵重构（每1000次更新）
- Redis 持久化
- 配置化模型列表
"""

import numpy as np
import pickle
import redis
from typing import List, Dict, Any, Optional


class LinUCBRouter:
    def __init__(self, redis_client: redis.Redis, config: dict,
                 feature_dim: int = 12, alpha: float = 1.0):
        """
        redis_client: Redis 连接实例
        config: 从 config.yaml 加载的完整配置
        feature_dim: 特征向量维度
        alpha: UCB 探索系数
        """
        self.redis = redis_client
        self.config = config
        self.dim = feature_dim
        self.alpha = alpha
        
        # 从配置读取模型列表（带容错）
        try:
            self.enabled_models = [m["name"] for m in config.get("models", {}).get("enabled", [])]
            if not self.enabled_models:
                self.enabled_models = ["hy3-preview", "qwen3:14b", "gemma3:12b"]
        except (KeyError, TypeError):
            self.enabled_models = ["hy3-preview", "qwen3:14b", "gemma3:12b"]

        # 每个 arm（模型）的 A 和 b 矩阵
        self.A: Dict[str, np.ndarray] = {}
        self.b: Dict[str, np.ndarray] = {}
        # 增量维护的逆矩阵 A_inv（避免每次求逆）
        self.A_inv: Dict[str, np.ndarray] = {}

        # 更新计数器（用于定期重构逆矩阵）
        self.update_counter = 0
        self.recompute_interval = 1000   # 每1000次更新重构一次

        self._load_or_init()

    # ---------- 持久化 ----------
    def _load_or_init(self):
        """从 Redis 加载历史参数，若不存在则初始化"""
        for model in self.enabled_models:
            key_A = f"linucb:A:{model}"
            key_b = f"linucb:b:{model}"
            key_A_inv = f"linucb:A_inv:{model}"
            
            if self.redis:
                try:
                    A_bytes = self.redis.get(key_A)
                    b_bytes = self.redis.get(key_b)
                    A_inv_bytes = self.redis.get(key_A_inv)
                    
                    if A_bytes and b_bytes:
                        self.A[model] = pickle.loads(A_bytes)
                        self.b[model] = pickle.loads(b_bytes)
                        
                        # 加载 A_inv，如果不存在则计算
                        if A_inv_bytes:
                            self.A_inv[model] = pickle.loads(A_inv_bytes)
                        else:
                            # 计算 A 的逆
                            self.A_inv[model] = np.linalg.inv(self.A[model])
                            # 保存到 Redis
                            self.redis.set(key_A_inv, pickle.dumps(self.A_inv[model]))
                        continue
                except Exception as e:
                    print(f"⚠️ 从 Redis 加载 {model} 失败: {e}，使用初始化值")
            
            # 初始化：A = I (单位矩阵)，b = 0
            self.A[model] = np.identity(self.dim)
            self.b[model] = np.zeros(self.dim)
            self.A_inv[model] = np.identity(self.dim)  # A_inv = I 的逆 = I

    def _save_model(self, model: str):
        """保存单个模型的参数到 Redis"""
        if not self.redis:
            return
        try:
            self.redis.set(f"linucb:A:{model}", pickle.dumps(self.A[model]))
            self.redis.set(f"linucb:b:{model}", pickle.dumps(self.b[model]))
            self.redis.set(f"linucb:A_inv:{model}", pickle.dumps(self.A_inv[model]))
        except Exception as e:
            print(f"⚠️ 保存 {model} 到 Redis 失败: {e}")

    # ---------- 特征提取 ----------
    def _extract_features(self, context: dict) -> np.ndarray:
        """
        将上下文映射为特征向量（自适应维度）
        特征设计：
        - 前4维：任务类型 one-hot
        - 第5维：输入长度归一化
        - 第6-7维：小时的正弦/余弦
        - 第8维：用户满意度
        - 扩展维度：模型近期成功率、成本、任务复杂度
        """
        feat = np.zeros(self.dim)
        try:
            # 任务类型 one-hot（前4维，如果维度足够）
            if self.dim > 4:
                task_type = context.get("task_type", "default")
                task_map = {
                    "code_generation": 0,
                    "long_text_analysis": 1,
                    "creative_writing": 2,
                    "reasoning": 3,
                    "default": 4
                }
                idx = task_map.get(task_type, 4)
                if idx < min(4, self.dim):
                    feat[idx] = 1.0
            
            # 输入长度归一化（第5维，如果维度足够）
            if self.dim > 5:
                input_len = context.get("input_length", 0)
                feat[4] = min(input_len / 4096, 1.0)
            
            # 小时的正弦/余弦（第6、7维）
            if self.dim > 6:
                hour = context.get("hour_of_day", 12)
                feat[5] = np.sin(2 * np.pi * hour / 24)
            if self.dim > 7:
                feat[6] = np.cos(2 * np.pi * hour / 24)
            
            # 用户满意度平均（第8维）
            if self.dim > 7:
                feat[7] = context.get("user_satisfaction_avg", 0.5)
            
            # 扩展维度（如果维度足够）
            if self.dim > 8:
                feat[8] = context.get("model_recent_success", 0.5)
            if self.dim > 9:
                feat[9] = context.get("model_recent_cost", 0.5)
            if self.dim > 10:
                feat[10] = context.get("complexity", 0.5)
            
        except Exception as e:
            print(f"⚠️ 特征提取失败: {e}，使用零向量")
            feat = np.zeros(self.dim)
        
        return feat

    # ---------- 决策与更新 ----------
    def select_arm(self, context: dict) -> str:
        """根据 UCB 公式选择最优模型（使用增量维护的 A_inv，O(d²)）"""
        x = self._extract_features(context)
        best_model = None
        best_ucb = -float('inf')
        
        for model in self.enabled_models:
            if model not in self.A_inv:
                continue
            
            # ✅ 使用增量维护的 A_inv，复杂度 O(d²)
            theta = self.A_inv[model] @ self.b[model]
            mean = theta @ x
            std = np.sqrt(x @ self.A_inv[model] @ x)
            ucb = mean + self.alpha * std
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_model = model
        
        return best_model if best_model else self.enabled_models[0]

    def update(self, model: str, context: dict, reward: float):
        """
        更新选定 arm 的参数（使用 Sherman-Morrison 增量更新 A_inv）
        
        公式：A_inv_new = A_inv_old - (A_inv_old @ x @ x^T @ A_inv_old) / (1 + x^T @ A_inv_old @ x)
        复杂度：O(d²) 而非 O(d³)
        """
        if model not in self.A:
            print(f"⚠️ 模型 {model} 未初始化，跳过更新")
            return
        
        x = self._extract_features(context)
        x_col = x.reshape(-1, 1)  # 列向量
        
        # 1. 更新 A 和 b
        self.A[model] += x_col @ x_col.T        # A += x * x^T
        self.b[model] += reward * x
        
        # 2. ✅ 使用 Sherman-Morrison 公式增量更新 A_inv (O(d²) 而非 O(d³))
        A_inv_old = self.A_inv[model]
        x_vec = x.reshape(-1, 1)
        
        # 计算 x^T @ A_inv @ x（标量）
        x_Ainv_x = float((x_vec.T @ A_inv_old @ x_vec).item())
        denominator = 1.0 + x_Ainv_x
        
        # 避免除零（当 x 是零向量时）
        if abs(denominator) < 1e-10:
            print(f"⚠️ 特征向量为零或分母过小，跳过 A_inv 更新")
        else:
            # 计算分子：A_inv @ x @ x^T @ A_inv
            numerator = A_inv_old @ x_vec @ x_vec.T @ A_inv_old
            
            # 更新 A_inv
            self.A_inv[model] = A_inv_old - numerator / denominator
        
        # 3. 定期强制重构逆矩阵（数值稳定性）
        self.update_counter += 1
        if self.update_counter >= self.recompute_interval:
            self.force_recompute_inv(model)
            self.update_counter = 0
        
        # 4. 持久化到 Redis
        self._save_model(model)

    def force_recompute_inv(self, model: str):
        """
        ✅ 强制重新计算逆矩阵（用于数值稳定性检查）
        
        Args:
            model: 模型名称
        """
        if model in self.A:
            self.A_inv[model] = np.linalg.inv(self.A[model])
            print(f"✅ 已强制重新计算 {model} 的 A_inv")
        else:
            print(f"⚠️ 模型 {model} 不存在，无法重新计算 A_inv")

    # ---------- 辅助方法 ----------
    def get_model_list(self) -> List[str]:
        """获取当前启用的模型列表"""
        return self.enabled_models.copy()
    
    def get_stats(self, model: Optional[str] = None) -> Dict:
        """
        获取统计信息
        
        Args:
            model: 模型名称（为 None 时返回所有模型）
            
        Returns:
            统计信息字典
        """
        if model:
            if model in self.A:
                return {
                    "model": model,
                    "A_trace": np.trace(self.A[model]),
                    "b_norm": np.linalg.norm(self.b[model]),
                    "A_inv_stable": np.allclose(self.A[model] @ self.A_inv[model], np.eye(self.dim))
                }
            return {}
        
        stats = {}
        for m in self.A.keys():
            stats[m] = self.get_stats(m)
        return stats
    

def create_linucb_router(redis_url: Optional[str] = None, config: dict = None,
                         feature_dim: int = 12, alpha: float = 1.0):
    """
    工厂函数：创建 LinUCBRouter 实例
    
    Args:
        redis_url: Redis URL（可选）
        config: 配置字典（从 config.yaml 加载）
        feature_dim: 特征维度
        alpha: 探索参数
        
    Returns:
        LinUCBRouter 实例
    """
    redis_client = None
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url, decode_responses=False)
            redis_client.ping()
            print(f"✅ Redis 连接成功: {redis_url}")
        except Exception as e:
            print(f"⚠️ Redis 连接失败，使用内存模式: {e}")
            redis_client = None
    
    return LinUCBRouter(redis_client=redis_client, config=config, 
                        feature_dim=feature_dim, alpha=alpha)


if __name__ == "__main__":
    # 简单测试
    import numpy as np
    
    # 模拟配置
    test_config = {
        "models": {
            "enabled": [
                {"name": "hy3-preview"},
                {"name": "qwen3:14b"},
                {"name": "gemma3:12b"}
            ]
        }
    }
    
    router = create_linucb_router(redis_url=None, config=test_config, 
                                   feature_dim=12, alpha=1.0)
    
    print("🧪 开始测试 LinUCB 路由（Sherman-Morrison 增量更新）...")
    
    for i in range(100):
        # 构造上下文
        context = {
            "task_type": ["code_generation", "long_text_analysis", "default"][i % 3],
            "input_length": 500 + i * 10,
            "hour_of_day": i % 24,
            "user_satisfaction_avg": 0.7 + 0.1 * (i % 5),
            "model_recent_success": 0.8 if i % 2 == 0 else 0.6,
            "model_recent_cost": 0.3 + 0.05 * (i % 3),
            "complexity": 0.5,
        }
        
        # 选择模型
        chosen = router.select_arm(context)
        
        # 模拟奖励（假设 hy3-preview 效果更好）
        if chosen == "hy3-preview":
            reward = 0.9 + 0.05 * np.random.randn()
        elif chosen == "qwen3:14b":
            reward = 0.8 + 0.08 * np.random.randn()
        else:
            reward = 0.7 + 0.1 * np.random.randn()
        
        # 更新模型
        router.update(chosen, context, reward)
        
        if (i + 1) % 20 == 0:
            print(f"  迭代 {i+1}: 选择 {chosen}, 奖励 {reward:.3f}")
    
    print(f"\n✅ 测试完成！")
    print(f"统计信息: {router.get_stats()}")
    
    # 测试强制重构
    if router.enabled_models:
        router.force_recompute_inv(router.enabled_models[0])
