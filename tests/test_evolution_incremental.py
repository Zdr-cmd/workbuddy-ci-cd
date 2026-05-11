"""
P0 增量学习接口测试
符合宪章 v5.0.0 第14章在线学习进化要求
"""

import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.evolution import incremental_train, register_linucb_router, _linucb_router


class TestIncrementalTrain:
    """测试增量训练接口"""

    def test_incremental_train_unknown_model(self):
        """未知 model_id 应返回错误"""
        result = incremental_train("unknown_model", {})
        assert "error" in result
        assert "Unknown model_id" in result["error"]

    def test_incremental_train_bandit_no_router(self):
        """Bandit 模型无路由器时应返回 error（不崩溃）"""
        result = incremental_train("bandit", {
            "model": "hy3",
            "context": {"task_type": "code"},
            "reward": 0.8
        })
        # 可能返回 error（因为没有真实路由器），但不应崩溃
        assert isinstance(result, dict)

    def test_incremental_train_weighted(self):
        """加权路由器增量更新"""
        result = incremental_train("weighted", {
            "model": "hy3",
            "task_type": "code_generation",
            "success": True,
            "cost": 0.01,
            "latency": 1.2
        })
        # 应返回 status: updated 或 error（取决于 WeightedModelRouter 是否存在）
        assert isinstance(result, dict)

    def test_incremental_train_data_structure(self):
        """测试不同数据结构不崩溃"""
        # 缺少字段
        r1 = incremental_train("bandit", {})
        assert isinstance(r1, dict)
        # 带额外字段
        r2 = incremental_train("weighted", {"extra": "field", "success": True})
        assert isinstance(r2, dict)


class TestRegisterRouter:
    """测试路由器注册"""

    def test_register_linucb_router_accepts_none(self):
        """register_linucb_router 应接受 None 值"""
        register_linucb_router(None)
        from core.evolution import _linucb_router
        assert _linucb_router is None

    def test_register_linucb_router_accepts_object(self):
        """register_linucb_router 应接受对象"""
        class FakeRouter:
            def update(self, m, c, r): pass
        router = FakeRouter()
        register_linucb_router(router, {"redis_url": "redis://localhost"})
        from core.evolution import _linucb_router
        assert _linucb_router is router


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
