"""
P0 验证测试 - 合规保障框架
符合宪章 v5.0.0 第15.5节要求
"""

import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.compliance import (
    CharterCompliance,
    RuleViolation,
    check_rule,
    get_compliance
)


class TestComplianceFramework:
    """第15.5节：合规检查测试"""

    def setup_method(self):
        self.c = CharterCompliance()
        self.c.reset()

    # ---- 安全规则测试 ----

    def test_safety_1_1_blocks_unconfirmed_native_process(self):
        """safety.1.1: 未经确认的外部操作应被阻止"""
        with pytest.raises(RuleViolation) as exc:
            self.c.check("safety.1.1", {
                "action": "execute_native_process",
                "confirmed": False
            })
        assert "safety.1.1" in str(exc.value)

    def test_safety_1_1_allows_confirmed_native_process(self):
        """safety.1.1: 已确认的外部操作应放行"""
        result = self.c.check("safety.1.1", {
            "action": "execute_native_process",
            "confirmed": True
        })
        assert result is True

    def test_safety_1_2_blocks_workbuddy_delete(self):
        """safety.1.2: 禁止删除 .workbuddy"""
        with pytest.raises(RuleViolation) as exc:
            self.c.check("safety.1.2", {"path": "C:\\Users\\.workbuddy\\something"})
        assert "safety.1.2" in str(exc.value)

    def test_safety_1_3_blocks_recursive_personal_delete(self):
        """safety.1.3: 禁止递归删除个人目录"""
        with pytest.raises(RuleViolation):
            self.c.check("safety.1.3", {
                "action": "delete_recursive",
                "path": "C:\\Users\\Lenovo\\Desktop\\temp"
            })

    # ---- 进化规则测试 ----

    def test_evolution_2_1_blocks_auto_adjust_when_frozen(self):
        """evolution.2.1: 冻结期不得自动调整"""
        with pytest.raises(RuleViolation):
            self.c.check("evolution.2.1", {
                "evolution_frozen": True,
                "force": False
            })

    def test_evolution_2_1_allows_forced_action(self):
        """evolution.2.1: force=True 可绕过冻结"""
        result = self.c.check("evolution.2.1", {
            "evolution_frozen": True,
            "force": True
        })
        assert result is True

    def test_evolution_2_2_requires_alert_recorded(self):
        """evolution.2.2: ML 降级必须记录"""
        with pytest.raises(RuleViolation):
            self.c.check("evolution.2.2", {
                "alert_type": "bandit_cumulative_regret",
                "recorded": False
            })

        # recorded=True 放行
        result = self.c.check("evolution.2.2", {
            "alert_type": "bandit_cumulative_regret",
            "recorded": True
        })
        assert result is True

    # ---- 记忆规则测试 ----

    def test_memory_3_1_blocks_sensitive_info(self):
        """memory.3.1: 敏感信息不得写入 MEMORY.md"""
        with pytest.raises(RuleViolation):
            self.c.check("memory.3.1", {
                "sensitive": True,
                "target": "MEMORY.md"
            })

    # ---- 路由规则测试 ----

    def test_routing_4_1_requires_notification_on_downgrade(self):
        """routing.4.1: Bandit 降级必须通知用户"""
        with pytest.raises(RuleViolation):
            self.c.check("routing.4.1", {
                "bandit_downgrade": True,
                "notified": False
            })

        result = self.c.check("routing.4.1", {
            "bandit_downgrade": True,
            "notified": True
        })
        assert result is True

    # ---- check_all 测试 ----

    def test_check_all_returns_failed_rule_ids(self):
        """check_all 返回所有失败规则 ID"""
        failed = self.c.check_all({
            "action": "execute_native_process",
            "confirmed": False,
            "evolution_frozen": False,
            "force": False,
            "alert_type": None,
            "recorded": True
        })
        assert "safety.1.1" in failed

    # ---- 合规报告测试 ----

    def test_compliance_report_structure(self):
        """get_compliance_report 返回正确的结构"""
        report = self.c.get_compliance_report()
        assert "charter_version" in report
        assert "total_rules" in report
        assert "total_violations" in report
        assert report["charter_version"] == "5.0.0"

    # ---- 全局单例测试 ----

    def test_get_compliance_returns_singleton(self):
        """get_compliance() 应返回同一实例"""
        c1 = get_compliance()
        c2 = get_compliance()
        assert c1 is c2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
