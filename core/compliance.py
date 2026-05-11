"""
合规保障框架（符合 WorkBuddy 宪章 v5.0.0 合规章节）
- 确保智能体严格按宪章执行所有任务
- 提供校验器、审计挂钩、运行时检查
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import json
import os


class RuleViolation(Exception):
    """规则违反异常"""
    pass


class CharterCompliance:
    """
    宪章合规检查器
    在每次重要操作前后调用 check() 进行合规验证
    """

    # 核心规则表（可扩展）
    RULES = {
        # 安全规则
        "safety.1.1": {
            "description": "禁止在未经确认前执行外部操作",
            "check": lambda ctx: ctx.get("action") != "execute_native_process" or ctx.get("confirmed") is True
        },
        "safety.1.2": {
            "description": "禁止删除 .workbuddy 目录",
            "check": lambda ctx: ".workbuddy" not in str(ctx.get("path", ""))
        },
        "safety.1.3": {
            "description": "禁止递归删除桌面/下载/文档目录",
            "check": lambda ctx: not (
                ctx.get("action") in ("delete_recursive", "rm_rf")
                and any(p in str(ctx.get("path", "")) for p in ["Desktop", "Downloads", "Documents", "Home"])
            )
        },
        # 进化规则
        "evolution.2.1": {
            "description": "进化冻结期不得触发自动调整",
            "check": lambda ctx: ctx.get("evolution_frozen", False) is False or ctx.get("force", False) is True
        },
        "evolution.2.2": {
            "description": "ML 告警自动降级必须记录到错题本",
            "check": lambda ctx: ctx.get("alert_type") is None or ctx.get("recorded") is True
        },
        # 记忆规则
        "memory.3.1": {
            "description": "敏感信息不得写入 MEMORY.md",
            "check": lambda ctx: ctx.get("sensitive") is not True
        },
        # 路由规则
        "routing.4.1": {
            "description": "Bandit 降级后必须通知用户",
            "check": lambda ctx: ctx.get("bandit_downgrade") is not True or ctx.get("notified") is True
        }
    }

    def __init__(self, charter_version: str = "5.0.0", audit_log: str = "logs/compliance.log"):
        self.version = charter_version
        self.audit_log = audit_log
        self.violations: List[Dict] = []
        self._load_rules()

    def _load_rules(self):
        """从配置文件扩展规则表（如有）"""
        cfg_path = "config/compliance_rules.json"
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    extra = json.load(f)
                    self.RULES.update(extra)
            except Exception:
                pass

    def check(self, rule_id: str, context: Dict[str, Any]) -> bool:
        """
        检查特定规则是否被违反
        返回 True（通过）或触发 RuleViolation
        """
        rule = self.RULES.get(rule_id)
        if rule is None:
            return True  # 未知规则默认放行

        try:
            passed = rule["check"](context)
        except Exception as e:
            self._record_violation(rule_id, context, str(e))
            raise RuleViolation(f"规则 {rule_id} 检查异常: {e}")

        if not passed:
            self._record_violation(rule_id, context, "规则判定为 False")
            raise RuleViolation(f"规则违反: {rule_id} — {rule['description']}")

        return True

    def _record_violation(self, rule_id: str, context: Dict, reason: str):
        record = {
            "rule_id": rule_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "reason": reason,
            "context_summary": {k: str(v)[:100] for k, v in list(context.items())[:5]},
            "charter_version": self.version
        }
        self.violations.append(record)
        self._persist_violation(record)

    def _persist_violation(self, record: Dict):
        os.makedirs(os.path.dirname(self.audit_log), exist_ok=True)
        with open(self.audit_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def check_all(self, context: Dict[str, Any]) -> List[str]:
        """
        执行所有规则检查，返回所有失败的规则 ID 列表
        """
        failed = []
        for rule_id in self.RULES:
            try:
                self.check(rule_id, context)
            except RuleViolation:
                failed.append(rule_id)
        return failed

    def get_compliance_report(self) -> Dict:
        return {
            "charter_version": self.version,
            "total_rules": len(self.RULES),
            "total_violations": len(self.violations),
            "recent_violations": self.violations[-10:],
            "last_check": datetime.utcnow().isoformat() + "Z"
        }

    def reset(self):
        self.violations = []


# 全局实例
_compliance_instance: Optional[CharterCompliance] = None


def get_compliance() -> CharterCompliance:
    global _compliance_instance
    if _compliance_instance is None:
        _compliance_instance = CharterCompliance()
    return _compliance_instance


def check_rule(rule_id: str, context: Dict[str, Any]) -> bool:
    return get_compliance().check(rule_id, context)
