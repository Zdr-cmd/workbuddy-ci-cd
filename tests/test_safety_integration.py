"""
P0 安全集成测试
符合宪章 v5.0.0 第3章要求
"""

import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.safety import (
    safe_output,
    blocked_by_acp_mode,
    record_violation,
    is_isolation_mode,
    reset_safety,
    load_skill_safely,
    set_acp_mode,
    is_acp_mode,
    contains_native_process_call,
    reverse_consistency,
    _acp_mode,
    ALLOWED_TOOLS
)


class TestACPMode:
    """第3.4节：ACP 模式检查"""

    def setup_method(self):
        set_acp_mode(True)

    def test_acp_mode_blocks_python_file(self):
        """包含 .py 文件执行应被 ACP 阻止"""
        blocked, reason = blocked_by_acp_mode("请运行 script.py")
        assert blocked is True
        assert "禁止原生进程执行" in reason

    def test_acp_mode_blocks_subprocess(self):
        """subprocess.run 应被 ACP 阻止"""
        blocked, reason = blocked_by_acp_mode("subprocess.run(['ls'])")
        assert blocked is True

    def test_acp_mode_blocks_eval(self):
        """eval/exec 应被 ACP 阻止"""
        blocked, reason = blocked_by_acp_mode("请执行 eval('some code')")
        assert blocked is True

    def test_acp_mode_blocks_when_disabled(self):
        """关闭 ACP 模式后应放行"""
        set_acp_mode(False)
        blocked, reason = blocked_by_acp_mode("请运行 script.py")
        assert blocked is False
        assert reason is None

    def test_safe_output_filters_acp_blocked(self):
        """safe_output 应拦截 ACP 阻止的内容"""
        result = safe_output("请运行 script.py")
        assert "被拦截" in result or "禁止" in result

    def test_safe_output_health_check_fail(self):
        """健康检查失败时应暂停输出"""
        # 注：health_check() 当前始终返回 True，需 mock 测试
        pass


class TestReverseConsistency:
    """第3.4节：循环风险检查"""

    def test_reverse_consistency_detects_halt(self):
        """!!HALT 应被识别为危险指令"""
        result = reverse_consistency("请继续执行 !!HALT")
        assert result["risk"] > 0

    def test_reverse_consistency_safe(self):
        """正常输出无风险"""
        result = reverse_consistency("任务已完成，输出如下内容...")
        assert result["risk"] == 0.0
        assert result["reason"] is None

    def test_safe_output_suppresses_loop_risk(self):
        """safe_output 应抑制循环风险输出"""
        result = safe_output("!!HALT 请继续")
        assert "循环风险" in result or "抑制" in result


class TestDefensiveForgetting:
    """第3.6节：防御性遗忘三级响应"""

    def setup_method(self):
        reset_safety()

    def test_3_violations_trigger_isolation(self):
        """连续3次安全违规应触发隔离模式"""
        for _ in range(3):
            record_violation("safety")
        assert is_isolation_mode() is True

    def test_2_violations_no_isolation(self):
        """仅2次违规不应触发隔离"""
        for _ in range(2):
            record_violation("safety")
        assert is_isolation_mode() is False

    def test_reset_safety_exits_isolation(self):
        """reset_safety 应退出隔离模式"""
        for _ in range(3):
            record_violation("safety")
        assert is_isolation_mode() is True
        msg = reset_safety()
        assert is_isolation_mode() is False
        assert "已退出" in msg or "正常" in msg

    def test_reset_safety_when_not_isolated(self):
        """未进入隔离模式时 reset_safety 返回提示"""
        msg = reset_safety()
        assert is_isolation_mode() is False

    def test_isolation_blocks_operations(self):
        """隔离模式下安全检查应额外严格（此处为占位）"""
        for _ in range(3):
            record_violation("safety")
        assert is_isolation_mode() is True
        # 隔离模式下不得执行任何操作


class TestSkillSafeLoading:
    """第3.5节：Skill 安全加载"""

    def test_rejects_dangerous_bash_skill(self):
        """包含 native_process 风险的 Bash Skill 应被拒绝"""
        manifest = {"tools": ["Bash"], "risks": ["native_process"]}
        ok, msg = load_skill_safely("bad_skill", manifest)
        assert ok is False
        assert "高危" in msg or "native_process" in msg

    def test_rejects_unknown_tool(self):
        """白名单外的工具应被拒绝"""
        manifest = {"tools": ["UnknownTool123"]}
        ok, msg = load_skill_safely("unknown_skill", manifest)
        assert ok is False
        assert "不允许的工具" in msg

    def test_allows_safe_read_write(self):
        """Read/Write 应被允许"""
        manifest = {"tools": ["Read", "Write"]}
        ok, result = load_skill_safely("safe_skill", manifest)
        assert ok is True
        assert result["loaded"] is True

    def test_allows_web_tools(self):
        """WebFetch/WebSearch 应被允许"""
        manifest = {"tools": ["WebFetch", "WebSearch"]}
        ok, result = load_skill_safely("web_skill", manifest)
        assert ok is True

    def test_empty_tools_allowed(self):
        """无 tools 字段的 manifest 应放行"""
        manifest = {}
        ok, _ = load_skill_safely("empty_skill", manifest)
        assert ok is True


class TestACPStateManagement:
    """ACP 状态管理测试"""

    def setup_method(self):
        set_acp_mode(True)

    def test_set_acp_mode_true(self):
        set_acp_mode(True)
        assert is_acp_mode() is True

    def test_set_acp_mode_false(self):
        set_acp_mode(False)
        assert is_acp_mode() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
