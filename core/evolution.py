"""
自动进化框架（符合 WorkBuddy 宪章 v5.0.0）
- 基于错题本和 MEMORY.md 的进化机制
- 指标-动作映射
- 进化冻结与回滚
- ML 告警自动降级（新增）
"""

import json
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional

# 全局状态
_evolution_frozen = False
_frozen_until = None
_last_rollback_time = None

# ML 告警相关全局变量（与 core/ml_monitor 联动）
_bandit_mode = True          # 当前是否启用 Bandit 路由（默认开启）
_bandit_epsilon = 0.1        # 当前探索率


# ========== 错题本管理 ==========
MISTAKE_NOTEBOOK_PATH = "C:/Users/Lenovo/WorkBuddy/Claw/错题本.md"

def record_failure(task_id: str, error_msg: str, root_cause: str = ""):
    """记录失败到错题本（步骤1）"""
    entry = f"\n## {datetime.now().isoformat()} | 任务: {task_id}\n- 错误: {error_msg}\n- 根因: {root_cause}\n- 整改: 待定\n"
    os.makedirs(os.path.dirname(MISTAKE_NOTEBOOK_PATH), exist_ok=True)
    with open(MISTAKE_NOTEBOOK_PATH, "a", encoding="utf-8") as f:
        f.write(entry)


# ========== 进化动作执行函数 ==========
def param_adjust(param_name: str, delta: float) -> bool:
    """调整参数（例如工作记忆容量）"""
    try:
        print(f"[Evolution] 调整参数 {param_name} 增量 {delta}")
        # 实际应修改 config 文件
        return True
    except Exception as e:
        record_failure("param_adjust", str(e), "参数调整失败")
        return False

def strategy_switch(task_type: str, new_strategy: str) -> bool:
    """切换任务分解策略"""
    print(f"[Evolution] 切换策略 {task_type} -> {new_strategy}")
    return True

def communication_template(template_name: str) -> bool:
    """更改沟通摘要格式"""
    print(f"[Evolution] 变更沟通模板为 {template_name}")
    return True

def skill_invocation_order(skill_list: list) -> bool:
    """调整多技能调用顺序"""
    print(f"[Evolution] 新技能调用顺序: {skill_list}")
    return True

def model_selection_tweak(weights: Dict[str, float]) -> bool:
    """调整模型路由权重"""
    try:
        from core.model_router import WeightedModelRouter
        router = WeightedModelRouter()
        mode = "balanced"
        if weights.get("cost", 0) > weights.get("latency", 0):
            mode = "cost"
        elif weights.get("latency", 0) > weights.get("cost", 0):
            mode = "latency"
        router.set_preference_mode(mode)
        return True
    except ImportError:
        print("[Evolution] model_router 不可用，跳过模型选择调整")
        return False


# ========== ML 告警自动降级动作（新增） ==========
def set_bandit_mode(enabled: bool) -> bool:
    """全局开关 LinUCB Bandit 路由（/ml bandit on/off）"""
    global _bandit_mode
    _bandit_mode = enabled
    # 实际应修改配置并通知路由器
    print(f"[Evolution] Bandit 模式已 {'启用' if enabled else '关闭'}")
    return True

def set_bandit_epsilon(epsilon: float) -> bool:
    """调整 Bandit 探索率（ε 值）"""
    global _bandit_epsilon
    epsilon = max(0.0, min(epsilon, 0.5))   # 限制在 [0, 0.5]
    _bandit_epsilon = epsilon
    print(f"[Evolution] Bandit 探索率调整为 {epsilon:.2f}")
    return True

def send_user_notification(message: str):
    """通过沟通协议向用户发送通知"""
    # 实际应调用 core.communication 中的方法，此处简化
    print(f"[Notification] {message}")


def on_ml_alert(alert_type: str, metric_value: float, threshold: float):
    """
    ML 告警处理函数（自动降级/调参）
    由 AlertManager 或 ml_monitor 在检测到告警时调用
    """
    global _evolution_frozen, _frozen_until

    # 如果处于冻结期，不处理新告警
    if is_evolution_frozen():
        print(f"[Evolution] 进化已冻结，忽略 ML 告警 {alert_type}")
        return

    if alert_type == "bandit_cumulative_regret":
        # 累积后悔持续增长，自动降级为加权路由
        record_failure("ml_bandit", f"累积后悔 {metric_value} 超阈值 {threshold}", "自动降级")
        set_bandit_mode(False)               # 关闭 LinUCB
        freeze_evolution(duration_hours=24)  # 暂停进化24小时，避免反复降级
        send_user_notification(
            "⚠️ 模型效果下降，已自动切换为保守路由策略。"
            "此时路由将使用加权评分（成本+成功率）。"
        )

    elif alert_type == "model_prediction_error":
        # 预测误差过高，增加探索率 ε (每次增加0.05，但不超过0.5)
        current_eps = get_bandit_epsilon()
        new_eps = min(current_eps + 0.05, 0.5)
        set_bandit_epsilon(new_eps)
        send_user_notification(
            f"🔍 预测误差偏高 (MAE={metric_value:.3f})，"
            f"已提高探索率至 {new_eps:.0%} 以收集更多数据。"
        )

    elif alert_type == "feature_distribution_shift":
        # 特征漂移：触发特征编码器增量训练（此处仅占位，实际需调用 ml_monitor 内方法）
        print(f"[Evolution] 检测到特征漂移 KL={metric_value:.3f}，建议重训特征编码器")
        # 可选：启动后台任务训练新编码器
        send_user_notification(
            "📊 检测到输入特征分布发生变化，建议重新训练特征编码器。\n"
            "运行 /ml feature retrain 以开始训练。"
        )

    else:
        print(f"[Evolution] 未知 ML 告警类型: {alert_type}")


# ========== ML 状态查询 ==========
def get_bandit_mode() -> bool:
    """获取当前 Bandit 模式（True=启用，False=降级到加权路由）"""
    return _bandit_mode

def get_bandit_epsilon() -> float:
    """获取当前探索率 ε"""
    return _bandit_epsilon


# ========== 原有进化框架函数 ==========
def select_evolution_action(metric_name: str, current_value: float) -> str:
    """根据超标指标返回建议的进化动作（名称）"""
    mapping = {
        "communication_clarity_rate": "communication_template",
        "skill_adoption_rate": "skill_invocation_order",
        "cost_efficiency": "model_selection_tweak",
        "task_success_rate": "strategy_switch",
        "evolution_activity": "param_adjust"
    }
    return mapping.get(metric_name, "param_adjust")

def freeze_evolution(duration_hours: int = 24):
    """冻结自动进化"""
    global _evolution_frozen, _frozen_until
    _evolution_frozen = True
    _frozen_until = datetime.now() + timedelta(hours=duration_hours)

def unfreeze_evolution():
    global _evolution_frozen, _frozen_until
    _evolution_frozen = False
    _frozen_until = None

def is_evolution_frozen() -> bool:
    if _evolution_frozen:
        if _frozen_until and datetime.now() > _frozen_until:
            unfreeze_evolution()
            return False
        return True
    return False

def rollback_to_version(target_version: str) -> Tuple[bool, str]:
    global _last_rollback_time
    backup_dir = f"backups/{target_version}"
    if not os.path.exists(backup_dir):
        return False, f"备份 {target_version} 不存在"
    try:
        if os.path.exists(f"{backup_dir}/config.json"):
            shutil.copy(f"{backup_dir}/config.json", "config.json")
        if os.path.exists(f"{backup_dir}/model_pref.json"):
            os.makedirs("state", exist_ok=True)
            shutil.copy(f"{backup_dir}/model_pref.json", "state/model_pref.json")
        freeze_evolution(24)
        _last_rollback_time = datetime.now()
        return True, f"已回滚到 {target_version}"
    except Exception as e:
        return False, str(e)


# ========== 在线学习接口（增量训练） ==========
_linucb_router: Optional[Any] = None  # 全局 LinUCB 路由器引用
_config: Optional[Dict[str, Any]] = None


def register_linucb_router(router: Any, config: Optional[Dict[str, Any]] = None):
    """注册 LinUCB 路由器实例，供 incremental_train 使用"""
    global _linucb_router, _config
    _linucb_router = router
    _config = config or {}


def incremental_train(model_id: str, new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    增量更新模型参数（在线学习）
    符合宪章 v5.0.0 第14章在线学习进化要求
    """
    global _linucb_router, _config

    if model_id == "bandit":
        if _linucb_router is None:
            # 尝试懒加载
            try:
                from core.linucb_router import LinUCBRouter
                cfg = _config or {}
                _linucb_router = LinUCBRouter(
                    redis_url=cfg.get("redis_url"),
                    feature_dim=cfg.get("feature_dim", 12)
                )
            except Exception:
                return {"error": "LinUCB router not initialized"}
        context = new_data.get("context", {})
        reward = float(new_data.get("reward", 0.0))
        model_name = new_data.get("model", "hy3")
        try:
            _linucb_router.update(model_name, context, reward)
            return {"status": "updated", "model": "bandit", "reward": reward}
        except Exception as e:
            return {"error": str(e), "model": "bandit"}

    elif model_id == "weighted":
        try:
            from core.model_router import WeightedModelRouter
            cfg = _config or {}
            router = WeightedModelRouter(config=cfg)
            router.update_stats(
                model=new_data.get("model", "hy3"),
                task_type=new_data.get("task_type", "default"),
                success=new_data.get("success", True),
                cost=float(new_data.get("cost", 0.0)),
                latency=float(new_data.get("latency", 0.0))
            )
            return {"status": "updated", "model": "weighted"}
        except ImportError:
            return {"error": "WeightedModelRouter not available"}
        except Exception as e:
            return {"error": str(e), "model": "weighted"}

    else:
        return {"error": f"Unknown model_id: {model_id}"}

