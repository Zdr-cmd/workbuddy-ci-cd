"""
数值稳定性测试 - LinUCB Router
运行 100000 次更新后检查矩阵条件数
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from core.linucb_router import LinUCBRouter


class MockRedis:
    """模拟 Redis 客户端"""
    def __init__(self):
        self.data = {}
    
    def get(self, key):
        return self.data.get(key)
    
    def set(self, key, value):
        self.data[key] = value
    
    def ping(self):
        return True


def numeric_stability_test():
    """数值稳定性测试：长时间运行后检查矩阵条件数"""
    d = 12
    n_updates = 100000
    recompute_interval = 1000
    
    config = {"models": {"enabled": [{"name": "hy3"}]}}
    router = LinUCBRouter(MockRedis(), config, feature_dim=d, alpha=1.0)
    router.recompute_interval = recompute_interval  # 每1000次重构一次
    
    print("=" * 60)
    print("数值稳定性测试")
    print("=" * 60)
    print(f"特征维度: {d}")
    print(f"总更新次数: {n_updates}")
    print(f"重构间隔: 每 {recompute_interval} 次")
    print()
    
    cond_history = []
    warnings = 0
    
    for i in range(n_updates):
        # 模拟多样化的特征
        task_type = np.random.choice(["code_generation", "long_text_analysis", "creative_writing", "default"])
        input_len = np.random.randint(10, 5000)
        hour = np.random.randint(0, 24)
        reward = np.random.normal(0.7, 0.1)
        
        context = {
            "task_type": task_type,
            "input_length": input_len,
            "hour_of_day": hour,
            "user_satisfaction_avg": np.random.rand(),
            "model_recent_success": np.random.rand(),
            "model_recent_cost": np.random.rand()
        }
        
        router.update("hy3", context, reward)
        
        # 每 10000 次检查条件数
        if (i + 1) % 10000 == 0:
            A = router.A["hy3"]
            cond = np.linalg.cond(A)
            cond_history.append(cond)
            
            if cond > 1e12:
                print(f"⚠️  Step {i+1}: condition number = {cond:.2e} 警告！")
                warnings += 1
            elif cond > 1e8:
                print(f"⚙️  Step {i+1}: condition number = {cond:.2e} (较高但可接受)")
            else:
                print(f"✅ Step {i+1}: condition number = {cond:.2e} (正常)")
    
    # 最终检查
    final_A = router.A["hy3"]
    final_A_inv = router.A_inv["hy3"]
    final_cond = np.linalg.cond(final_A)
    
    print()
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)
    print(f"最终条件数: {final_cond:.2e}")
    print(f"警告次数 (cond > 1e12): {warnings}")
    print()
    
    # 验证 A * A_inv ≈ I
    identity_approx = final_A @ final_A_inv
    is_stable = np.allclose(identity_approx, np.eye(d), atol=1e-6)
    
    print(f"A * A_inv ≈ I: {'✅ 是' if is_stable else '❌ 否'}")
    if not is_stable:
        print(f"最大偏差: {np.max(np.abs(identity_approx - np.eye(d))):.2e}")
    print()
    
    # 断言检查
    try:
        assert final_cond < 1e12, f"矩阵病态，条件数过高: {final_cond:.2e}"
        assert is_stable, "A * A_inv 不接近单位矩阵"
        print("✅ 所有检查通过！数值稳定性良好。")
        return True
    except AssertionError as e:
        print(f"❌ 检查失败: {e}")
        return False


if __name__ == "__main__":
    success = numeric_stability_test()
    sys.exit(0 if success else 1)
