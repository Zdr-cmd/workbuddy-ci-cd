"""
安全防护体系（符合 WorkBuddy 宪章 v5.0.0 第3章）
- 三明治检查（健康检查 + ACP模式 + 循环风险）
- Skill 安全加载（工具白名单）
- 防御性遗忘（三级响应）
"""

import re
import threading
from typing import Tuple, Optional

# 全局状态
_isolation_mode = False
_consecutive_violations = {}  # {rule_type: count}
_violation_lock = threading.Lock()
_acp_mode = True  # ACP 模式默认开启

# 允许的工具白名单（宪章第3章）
ALLOWED_TOOLS = {
    "Read", "Write", "Edit", "Bash", "PowerShell",
    "WebFetch", "WebSearch", "TaskCreate", "TaskUpdate",
    "TaskList", "TaskGet", "Bash", "PowerShell",
    "Agent", "SendMessage", "Glob", "Grep"
}


# ========== 健康检查 ==========
def health_check() -> bool:
    """检查系统内部状态（内存、队列、死循环等）"""
    # 实际可扩展检查内存占用、线程数等
    return True


# ========== ACP 模式检查 ==========
def contains_native_process_call(text: str) -> bool:
    """检测文本中是否包含原生进程执行模式"""
    patterns = [
        r"\.py\b", r"\.exe\b", r"subprocess\.run", r"os\.system",
        r"exec\(", r"eval\(", r"__import__", r"compile\("
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def blocked_by_acp_mode(output: str) -> Tuple[bool, Optional[str]]:
    """ACP模式检查：禁止原生进程执行"""
    if not _acp_mode:
        return False, None
    if contains_native_process_call(output):
        return True, "ACP模式禁止原生进程执行"
    return False, None


# ========== 循环风险检查 ==========
def reverse_consistency(output: str) -> dict:
    """输出若再次作为输入是否引发危险循环"""
    risk = 0.0
    dangerous_terms = ["!!HALT", "请继续", "重复执行"]
    for term in dangerous_terms:
        if term in output:
            risk += 0.05
    return {"risk": min(risk, 1.0), "reason": "包含危险指令" if risk > 0 else None}


# ========== 三明治检查 ==========
def safe_output(raw: str) -> str:
    """输出前的三明治安全检查（前-后-输出）"""
    # 前置检查
    if not health_check():
        return "系统内部状态异常，暂停输出。"

    # ACP 检查
    blocked, reason = blocked_by_acp_mode(raw)
    if blocked:
        return f"输出被拦截：{reason}"

    # 循环风险检查
    risk_info = reverse_consistency(raw)
    if risk_info["risk"] >= 0.05:
        return "输出可能引发循环风险，已抑制。"

    return raw


# ========== Skill 安全加载 ==========
def load_skill_safely(skill_name: str, skill_manifest: dict) -> Tuple[bool, any]:
    """检查 tools 字段在白名单内，拒绝危险工具"""
    tools = skill_manifest.get("tools", [])
    for tool in tools:
        if tool not in ALLOWED_TOOLS:
            return False, f"Skill '{skill_name}' 包含不允许的工具：{tool}"
        if tool == "Bash" and "native_process" in skill_manifest.get("risks", []):
            return False, f"Skill '{skill_name}' 包含高危 Bash 操作"
    return True, {"name": skill_name, "tools": tools, "loaded": True}


# ========== 防御性遗忘 ==========
def record_violation(rule_type: str):
    """记录规则违反，连续3次进入隔离模式"""
    global _consecutive_violations, _isolation_mode
    with _violation_lock:
        _consecutive_violations[rule_type] = _consecutive_violations.get(rule_type, 0) + 1
        if rule_type == "safety" and _consecutive_violations[rule_type] >= 3:
            enter_isolation_mode()


def enter_isolation_mode():
    """进入隔离模式：停止所有工具调用"""
    global _isolation_mode
    _isolation_mode = True


def reset_safety() -> str:
    """退出隔离模式"""
    global _isolation_mode, _consecutive_violations
    _isolation_mode = False
    _consecutive_violations.clear()
    return "已退出隔离模式，系统恢复正常。"


def is_isolation_mode() -> bool:
    return _isolation_mode


def set_acp_mode(enabled: bool):
    """供测试和配置使用：设置全局 ACP 模式状态"""
    global _acp_mode
    _acp_mode = enabled


def is_acp_mode() -> bool:
    """返回当前 ACP 模式状态"""
    return _acp_mode
