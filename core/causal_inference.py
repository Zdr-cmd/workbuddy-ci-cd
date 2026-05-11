"""
因果推断框架（符合 WorkBuddy 宪章 v5.0.0 第15.3节）
- 双重差分（DiD）评估进化动作的长期因果效应
- 倾向性得分匹配（PSM）辅助
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class CausalImpactEvaluator:
    def __init__(self, log_path: str = "logs/audit.log"):
        self.log_path = log_path
        self.df = self._load_logs()

    def _load_logs(self) -> pd.DataFrame:
        """加载审计日志，转换为 DataFrame"""
        records = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("event_type") == "task_completed":
                            records.append({
                                "timestamp": pd.to_datetime(entry["timestamp"]),
                                "user_id": entry.get("user_id"),
                                "user_group": entry.get("user_group", "control"),
                                "success": entry.get("success", False),
                                "cost_usd": entry.get("cost_usd", 0.0),
                                "latency": entry.get("latency_sec", 0.0)
                            })
                    except (json.JSONDecodeError, KeyError):
                        continue
        except FileNotFoundError:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def evaluate_evolution(self, action_name: str, action_time,
                           metric: str = "success") -> Dict:
        """
        使用双重差分评估进化动作的因果效应。
        action_time: 动作部署时间点（pd.Timestamp）
        metric: 待评估指标（success/cost_usd/latency）
        """
        if self.df.empty:
            return {"error": "Insufficient data", "causal_effect": None, "p_value": 1.0}

        df = self.df.copy()
        df["post"] = (df["timestamp"] >= action_time).astype(int)

        # 若没有 treated 标记，将前一半用户视为实验组（仅演示）
        if "treated" not in df.columns:
            users = df["user_id"].unique()
            treated_users = set(users[:len(users) // 2])
            df["treated"] = df["user_id"].isin(treated_users).astype(int)

        try:
            # 简单回归：metric ~ treated * post
            from statsmodels.formula.api import ols
            formula = f"{metric} ~ C(treated) * C(post)"
            model = ols(formula, data=df).fit()
            key = "C(treated)[T.1]:C(post)[T.1]"
            coeff = model.params.get(key, 0.0)
            p_value = model.pvalues.get(key, 1.0)
            return {
                "causal_effect": float(coeff),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "action": action_name
            }
        except Exception as e:
            return {"error": str(e), "causal_effect": None, "p_value": 1.0}

    def propensity_score_matching(self, covariates: List[str],
                                  caliper: float = 0.05) -> tuple:
        """
        返回匹配后的（处理组索引，对照组索引）
        简化实现，需配合完整 PSM 流程使用
        """
        if self.df.empty:
            return [], []
        # 占位：完整 PSM 需 sklearn LogisticRegression + NearestNeighbors
        return [], []

    def summary(self) -> Dict:
        """返回评估摘要"""
        if self.df.empty:
            return {"status": "no_data", "total_records": 0}
        return {
            "status": "ready",
            "total_records": len(self.df),
            "unique_users": int(self.df["user_id"].nunique()),
            "columns": list(self.df.columns)
        }
