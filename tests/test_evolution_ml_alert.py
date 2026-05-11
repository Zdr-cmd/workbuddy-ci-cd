"""
P0 验证测试 - ML 告警自动降级
符合宪章 v5.0.0 第4.3节要求
"""

import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 导入前确保 evolution.py 无语法错误
from core.evolution import (
    on_ml_alert,
    get_bandit_mode,
    get_bandit_epsilon,
    set_bandit_mode,
    set_bandit_epsilon,
    freeze_evolution,
    is_evolution_frozen,
    unfreeze_evolution,
)


class TestMLAlertAutoDegradation:
    """第4.3节：三种 ML 告警类型测试"""

    def setup_method(self):
        """每个测试前重置全局状态"""
        set_bandit_mode(True)
        set_bandit_epsilon(0.1)
        unfreeze_evolution()

    def test_bandit_cumulative_regret_alert(self):
        """
        触发 bandit_cumulative_regret 告警
        → 关闭 Bandit，降级为加权路由
        → 冻结进化 24h
        """
        initial_mode = get_bandit_mode()
        assert initial_mode is True

        on_ml_alert(
            alert_type="bandit_cumulative_regret",
            metric_value=0.95,
            threshold=0.80
        )

        # ✅ Bandit 模式应关闭
        assert get_bandit_mode() is False, \
            "bandit_cumulative_regret 触发后 Bandit 应降级为 False"

        # ✅ 进化应冻结
        assert is_evolution_frozen() is True, \
            "bandit_cumulative_regret 触发后进化应冻结"

    def test_model_prediction_error_alert(self):
        """
        触发 model_prediction_error 告警
        → ε 探索率增加 0.05（上限 0.5）
        """
        set_bandit_epsilon(0.10)

        on_ml_alert(
            alert_type="model_prediction_error",
            metric_value=0.15,
            threshold=0.10
        )

        # ✅ ε 应增加 0.05
        new_eps = get_bandit_epsilon()
        assert new_eps == pytest.approx(0.15, abs=0.01), \
            f"预测误差告警后 ε 应为 0.15，实际 {new_eps}"

    def test_model_prediction_error_caps_at_0_5(self):
        """
        ε 探索率有上限 0.5，不应无限增长
        """
        set_bandit_epsilon(0.48)

        on_ml_alert(
            alert_type="model_prediction_error",
            metric_value=0.20,
            threshold=0.10
        )

        new_eps = get_bandit_epsilon()
        assert new_eps == pytest.approx(0.5, abs=0.01), \
            f"ε 上限为 0.5，实际 {new_eps}"

    def test_feature_distribution_shift_alert(self):
        """
        触发 feature_distribution_shift 告警
        → 仅通知用户，不改变 Bandit 模式或 ε
        """
        set_bandit_mode(True)
        set_bandit_epsilon(0.1)

        on_ml_alert(
            alert_type="feature_distribution_shift",
            metric_value=0.5,
            threshold=0.3
        )

        # ✅ 模式不变
        assert get_bandit_mode() is True
        # ✅ ε 不变
        assert get_bandit_epsilon() == pytest.approx(0.1, abs=0.01)
        # ✅ 进化不冻结
        assert is_evolution_frozen() is False

    def test_unknown_alert_type_no_crash(self):
        """
        未知告警类型应安全忽略，不抛异常
        """
        try:
            on_ml_alert(
                alert_type="unknown_alert_type",
                metric_value=1.0,
                threshold=0.0
            )
            passed = True
        except Exception as e:
            passed = False
            pytest.fail(f"未知告警类型不应抛异常: {e}")

        assert passed is True

    def test_evolution_frozen_blocks_regret_alert(self):
        """
        进化冻结期间，bandit_cumulative_regret 告警被忽略
        """
        freeze_evolution(duration_hours=24)
        assert is_evolution_frozen() is True

        # 保存当前状态
        mode_before = get_bandit_mode()

        on_ml_alert(
            alert_type="bandit_cumulative_regret",
            metric_value=0.95,
            threshold=0.80
        )

        # ✅ 告警被忽略，模式不变
        assert get_bandit_mode() == mode_before, \
            "冻结期间不应处理新告警"

    def test_set_bandit_mode_boundaries(self):
        """
        set_bandit_mode 应接受 True/False
        """
        assert set_bandit_mode(True) is True
        assert get_bandit_mode() is True

        assert set_bandit_mode(False) is True
        assert get_bandit_mode() is False

    def test_set_bandit_epsilon_boundaries(self):
        """
        ε 探索率限制在 [0.0, 0.5]
        """
        set_bandit_epsilon(-0.5)
        assert get_bandit_epsilon() == 0.0  # 下限

        set_bandit_epsilon(0.8)
        assert get_bandit_epsilon() == 0.5  # 上限


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
