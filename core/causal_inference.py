"""
因果推断框架（符合 WorkBuddy 宪章 v5.0.0 第15.3节）
- 双重差分（DiD）评估进化动作的长期因果效应
- 倾向性得分匹配（PSM）辅助
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


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

    def propensity_score_matching(self, treatment_col: str = "treated",
                                  covariates: List[str] = None,
                                  caliper: float = 0.05) -> Tuple[List[int], List[int]]:
        """
        使用 sklearn 实现完整的倾向性得分匹配（1:1 最近邻，带卡钳）
        返回 (matched_treatment_idx, matched_control_idx)
        符合宪章 v5.0.0 第15.3节要求
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors

        if self.df.empty:
            return [], []

        if covariates is None:
            covariates = ["cost_usd", "latency"]

        df = self.df.copy()

        # 构建处理组标记
        if treatment_col not in df.columns:
            df["treated"] = (
                df.get("user_group", "control") == "experiment"
            ).astype(int)
        else:
            df["treated"] = df[treatment_col]

        # 过滤有效协变量
        available_covs = [c for c in covariates if c in df.columns]
        if not available_covs:
            return [], []

        df = df.dropna(subset=available_covs + ["treated"])
        if len(df) == 0:
            return [], []

        X = df[available_covs].values.astype(float)
        y = df["treated"].values

        n_treated = int(y.sum())
        n_control = len(y) - n_treated
        if n_treated == 0 or n_control == 0:
            return [], []

        # 1. 使用 LogisticRegression 估计倾向性得分
        ps_model = LogisticRegression(C=1e6, max_iter=1000)
        ps_model.fit(X, y)
        df["pscore"] = ps_model.predict_proba(X)[:, 1]

        # 2. 1:1 最近邻匹配（带卡钳限制）
        treated_idx = df[df["treated"] == 1].index.tolist()
        control_idx = df[df["treated"] == 0].index.tolist()

        control_pscores = df.loc[control_idx, "pscore"].values.reshape(-1, 1)
        nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(control_pscores)

        matched_treated = []
        matched_control = []
        used_control = set()

        for t in treated_idx:
            pscore_t = df.loc[t, "pscore"]
            distances, indices = nbrs.kneighbors([[pscore_t]])
            closest_c = control_idx[indices[0][0]]
            if closest_c in used_control:
                continue
            pscore_c = df.loc[closest_c, "pscore"]
            if abs(pscore_t - pscore_c) <= caliper:
                matched_treated.append(t)
                matched_control.append(closest_c)
                used_control.add(closest_c)

        return matched_treated, matched_control

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
