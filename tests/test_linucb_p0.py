"""
P0 验证测试 - LinUCB 路由器
测试自动重构、延迟统计、配置化模型列表
"""

import pytest
import numpy as np
import sys
import os

# 添加 core 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.linucb_router import LinUCBRouter
from core.model_router import WeightedModelRouter
import yaml


# ---------- Fixtures ----------
@pytest.fixture
def config():
    """测试配置"""
    return {
        "models": {
            "enabled": [{"name": "hy3"}, {"name": "qwen"}],
            "fallback": "hy3"
        },
        "routing": {
            "task_type_mapping": {
                "code": ["hy3", "qwen"],
                "default": ["hy3"]
            }
        }
    }


class MockRedis:
    """模拟 Redis 客户端（用于单元测试）"""
    def __init__(self):
        self.data = {}
    
    def get(self, key):
        return self.data.get(key)
    
    def set(self, key, value):
        self.data[key] = value
    
    def ping(self):
        return True


@pytest.fixture
def mock_redis():
    """模拟 Redis 夹具"""
    return MockRedis()


# ---------- LinUCB 测试 ----------
def test_auto_recompute_inv(mock_redis, config):
    """验证自动重构计数器每 N 次触发"""
    router = LinUCBRouter(mock_redis, config, feature_dim=4, alpha=1.0)
    router.recompute_interval = 3  # 为测试调低阈值
    
    # 初始计数器应为 0
    assert router.update_counter == 0
    
    # 执行 10 次更新
    for i in range(10):
        router.update("hy3", {"task_type": "code", "input_length": 100}, reward=0.8)
    
    # 每 3 次会重置计数器，所以 10 次后计数器应为 1 (10 % 3 = 1)
    assert router.update_counter == 1


def test_force_recompute_inv(mock_redis, config):
    """验证强制重构功能"""
    router = LinUCBRouter(mock_redis, config, feature_dim=4, alpha=1.0)
    
    # 先更新几次，改变 A 矩阵
    for i in range(5):
        router.update("hy3", {"task_type": "code", "input_length": 100}, reward=0.8)
    
    # 记录重构前的 A_inv
    A_inv_old = router.A_inv["hy3"].copy()
    
    # 强制重构
    router.force_recompute_inv("hy3")
    
    # 验证 A_inv 被重新计算（应该更接近真实逆矩阵）
    A = router.A["hy3"]
    A_inv_new = router.A_inv["hy3"]
    
    # 检查 A * A_inv ≈ I
    identity_approx = A @ A_inv_new
    assert np.allclose(identity_approx, np.eye(4), atol=1e-6)


def test_latency_statistics(config):
    """验证延迟数据被记录"""
    router = WeightedModelRouter(stats_file="test_stats.json", window_size=10)
    router.update_stats("hy3", "code", success=True, cost=0.02, latency=0.5)
    
    key = "hy3_code"
    assert key in router.stats
    assert "latency" in router.stats[key]
    assert len(router.stats[key]["latency"]) == 1
    assert router.stats[key]["latency"][0] == 0.5
    
    # 清理测试文件
    import os
    if os.path.exists("test_stats.json"):
        os.remove("test_stats.json")


def test_configurable_models(config):
    """验证模型列表从配置读取"""
    router = LinUCBRouter(None, config, feature_dim=4)
    assert "hy3" in router.enabled_models
    assert "qwen" in router.enabled_models
    assert len(router.enabled_models) == 2


def test_select_model_uses_latency(config):
    """验证延迟分数真正使用 latency 而非 cost 占位"""
    # ✅ 传递配置给 WeightedModelRouter
    router = WeightedModelRouter(config=config, stats_file="test_stats.json", window_size=10)
    
    # 设置偏好为延迟优先
    router.preference.set_mode("latency")
    
    # hy3 延迟低，qwen 延迟高
    router.update_stats("hy3", "code", success=True, cost=0.02, latency=0.5)
    router.update_stats("qwen", "code", success=True, cost=0.00, latency=2.0)
    
    # 多次选择，统计选择结果
    hy3_count = 0
    qwen_count = 0
    for _ in range(100):
        chosen = router.select_model("code")
        if chosen == "hy3":
            hy3_count += 1
        elif chosen == "qwen":
            qwen_count += 1
    
    # hy3 延迟更低，应该被选择更多次
    assert hy3_count > qwen_count, f"hy3 被选 {hy3_count} 次，qwen 被选 {qwen_count} 次"
    
    # 清理测试文件
    import os
    if os.path.exists("test_stats.json"):
        os.remove("test_stats.json")


def test_sherman_morrison_correctness(mock_redis, config):
    """验证 Sherman-Morrison 增量更新正确性"""
    router = LinUCBRouter(mock_redis, config, feature_dim=4, alpha=1.0)
    
    # 更新多次
    for i in range(10):
        context = {"task_type": "code", "input_length": 100 + i}
        router.update("hy3", context, reward=0.8)
    
    # 验证 A * A_inv ≈ I（数值稳定性）
    A = router.A["hy3"]
    A_inv = router.A_inv["hy3"]
    identity_approx = A @ A_inv
    
    assert np.allclose(identity_approx, np.eye(4), atol=1e-6), \
        f"A * A_inv 不接近单位矩阵:\n{identity_approx}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
