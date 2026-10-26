"""模型库：随机森林 / GBDT / LSTM / 融合（参考 qlib 的模型组织方式）。

对比基线（qlib examples/benchmarks，CSI300，Alpha158）
----------------------------------------------------
- LightGBM：lr 0.2 / max_depth 8 / num_leaves 210，是 GBDT 类里的强基线。
  本机没装 lightgbm，但有 xgboost，两者同族，用 xgboost 承担 GBDT 角色；
  xgboost 也不可用时退回 sklearn HistGradientBoosting（原生支持 NaN，很快）。
- LSTM：qlib 用 TSDatasetH，step_len=20、hidden 64、2 层、lr 1e-3、batch 800、
  early_stop 10。这里照抄这套超参（见 config.LSTM），但把 4 个周期做成
  **一个共享编码器 + 多任务输出头**，因为 CPU 上训练 8 个独立网络会慢到不可用。

统一的对外接口（与 ForecastModel 保持一致，便于替换）
----------------------------------------------------
    fit(samples, ctx=None)          samples = {h: (X, y_dir, y_reg)}
    predict_horizon(h, X, ctx=None) -> (pred_ret, prob_up)
    .horizons / .fit_ok / .name
ctx = {"panel": 面板(已 z-score), "symbol": 目标股代码}，只有 LSTM 需要。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from .features import FEATURES
from .model import ForecastModel


# ------------------------- 通用基类 -------------------------
class BaseForecaster:
    name = "base"
    needs_seq = False

    def __init__(self):
        self.horizons = []
        self.fit_ok = False
        self.ret_scale: dict = {}
        self.note = ""

    # 子类实现
    def fit(self, samples: dict, ctx: dict | None = None):
        raise NotImplementedError

    def predict_horizon(self, h: int, X=None, ctx: dict | None = None):
        raise NotImplementedError

    def predict_batch(self, h: int, X, ctx: dict | None = None):
        """整批预测（默认逐行循环）。

        逐行调用 predict_proba 在 n_jobs=-1 时每次都要起线程池，
        实测 1 万次单行预测要 7 分钟；整批一次调用只要零点几秒。
        """
        rs, ps = [], []
        for i in range(len(X)):
            r, p = self.predict_horizon(h, X.iloc[[i]], ctx)
            rs.append(r)
            ps.append(p)
        return np.asarray(rs), np.asarray(ps)

    def predict_members(self, h: int, X=None, ctx: dict | None = None):
        """返回各成员的 (name, ret, prob_up)，用于估计模型分歧。

        单模型时只有一个成员，分歧为 0（调用方会退化为用波动率给带宽）。
        """
        r, p = self.predict_horizon(h, X, ctx)
        return [(getattr(self, "name", "model"), float(r), float(p))]

    def _post_scale(self, h: int, ret: float) -> float:
        """csrank 标签下把排名分换算回收益率口径。"""
        if (getattr(config, "LABEL_MODE", "ret") or "ret").lower() == "csrank":
            return float(ret) * float(self.ret_scale.get(h, 0.02))
        return float(ret)


# ------------------------- 随机森林（原实现） -------------------------
class RFForecaster(BaseForecaster):
    name = "rf"

    def __init__(self):
        super().__init__()
        self._impl = ForecastModel()

    def fit(self, samples: dict, ctx: dict | None = None):
        self._impl.fit(samples)
        self.horizons = self._impl.horizons
        self.fit_ok = self._impl.fit_ok
        return self

    def predict_horizon(self, h: int, X=None, ctx: dict | None = None):
        r, p = self._impl.predict_horizon(h, X)
        return self._post_scale(h, r), p

    def predict_batch(self, h: int, X, ctx: dict | None = None):
        reg, clf = self._impl.regs[h], self._impl.clfs[h]
        rs = reg.predict(X)
        ps = clf.predict_proba(X)[:, 1]
        if (getattr(config, "LABEL_MODE", "ret") or "ret").lower() == "csrank":
            rs = rs * float(self.ret_scale.get(h, 0.02))
        return np.asarray(rs), np.asarray(ps)


# ------------------------- GBDT -------------------------
class GBDTForecaster(BaseForecaster):
    """XGBoost（优先）/ sklearn HistGradientBoosting 兜底。"""

    name = "gbdt"

    def __init__(self):
        super().__init__()
        self.regs, self.clfs = {}, {}
        self.backend = ""

    def _make(self, kind: str):
        if self.backend == "xgboost":
            import xgboost as xgb
            if kind == "reg":
                return xgb.XGBRegressor(
                    n_estimators=400, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    objective="reg:squarederror", n_jobs=4,
                    random_state=config.RANDOM_STATE, tree_method="hist")
            return xgb.XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                objective="binary:logistic", eval_metric="logloss", n_jobs=4,
                random_state=config.RANDOM_STATE, tree_method="hist")
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        if kind == "reg":
            return HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=6,
                min_samples_leaf=20, l2_regularization=1.0,
                random_state=config.RANDOM_STATE)
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=6,
            min_samples_leaf=20, l2_regularization=1.0,
            random_state=config.RANDOM_STATE)

    def fit(self, samples: dict, ctx: dict | None = None):
        try:
            import xgboost  # noqa: F401
            self.backend = "xgboost"
        except Exception:
            self.backend = "sklearn"
        for h, (X, y_dir, y_reg) in samples.items():
            if len(X) < 200:
                continue
            try:
                reg = self._make("reg").fit(X, y_reg)
                clf = self._make("clf").fit(X, y_dir.astype(int))
            except Exception as e:      # 某些版本对 NaN/参数挑剔，失败就跳过该周期
                self.note = f"{h}日训练失败：{e}"
                continue
            self.regs[h], self.clfs[h] = reg, clf
        self.horizons = sorted(self.regs)
        self.fit_ok = bool(self.horizons)
        return self

    def predict_horizon(self, h: int, X=None, ctx: dict | None = None):
        r = float(self.regs[h].predict(X)[0])
        p = float(self.clfs[h].predict_proba(X)[0, 1])
        return self._post_scale(h, r), p

    def predict_batch(self, h: int, X, ctx: dict | None = None):
        rs = self.regs[h].predict(X)
        ps = self.clfs[h].predict_proba(X)[:, 1]
        if (getattr(config, "LABEL_MODE", "ret") or "ret").lower() == "csrank":
            rs = rs * float(self.ret_scale.get(h, 0.02))
        return np.asarray(rs), np.asarray(ps)


# ------------------------- LSTM（PyTorch） -------------------------
class _SeqDataset:
    """按 (股票, 结束位置) 惰性切滑动窗口，避免把 N×L×F 的序列整体拷进内存。

    27 只 × 700 天 × 20 步 × 33 特征 ≈ 50MB（f32），加上 alpha 特征会到 120MB+，
    在本机内存偏紧的情况下很容易 OOM。惰性切片峰值只是 batch 大小。
    """

    def __init__(self, panel: pd.DataFrame, feat: list[str], horizons: list[int],
                 seq_len: int):
        self.seq_len = seq_len
        self.feat = feat
        self.horizons = horizons
        self.arrays: list[np.ndarray] = []       # 每只股票的特征矩阵 (T, F)
        self.index: list[tuple[int, int]] = []    # (股票下标, 结束位置)
        self.y_reg = {h: [] for h in horizons}
        self.y_dir = {h: [] for h in horizons}
        self.symbols: list[str] = []
        self.end_dates: list = []

        for sym, g in panel.groupby("symbol"):
            g = g.sort_values("date").reset_index(drop=True)
            arr = g[feat].to_numpy(dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            close = g["close"].to_numpy(dtype=np.float64)
            dates = g["date"].tolist()
            sid = len(self.arrays)
            self.arrays.append(arr)
            self.symbols.append(sym)
            t_max = len(arr)
            fwd = {h: np.concatenate([close[h:] / close[:-h] - 1, np.full(h, np.nan)])
                   for h in horizons if 0 < h < t_max}
            for t in range(seq_len - 1, t_max):
                ok = all(np.isfinite(fwd[h][t]) for h in horizons if h in fwd)
                if not ok:
                    continue
                self.index.append((sid, t))
                self.end_dates.append(dates[t])
                for h in horizons:
                    r = float(fwd[h][t]) if h in fwd else 0.0
                    self.y_reg[h].append(r)
                    self.y_dir[h].append(1.0 if r > 0 else 0.0)

    def __len__(self):
        return len(self.index)

    def get_seq(self, sid: int, t: int) -> np.ndarray:
        return self.arrays[sid][t - self.seq_len + 1: t + 1]


class LSTMForecaster(BaseForecaster):
    name = "lstm"
    needs_seq = True

    def __init__(self):
        super().__init__()
        self.net = None
        self.dataset: _SeqDataset | None = None
        self.backend = "torch"

    # ---- 网络：共享 LSTM 编码器 + 每周期 (回归, 方向) 双头 ----
    @staticmethod
    def _build_net(n_feat: int, horizons: list[int]):
        import torch
        import torch.nn as nn
        cfg = config.LSTM

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(input_size=n_feat, hidden_size=cfg["hidden_size"],
                                   num_layers=cfg["num_layers"], batch_first=True,
                                   dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0)
                self.reg = nn.ModuleDict({str(h): nn.Linear(cfg["hidden_size"], 1) for h in horizons})
                self.clf = nn.ModuleDict({str(h): nn.Linear(cfg["hidden_size"], 1) for h in horizons})

            def forward(self, x):
                out, _ = self.rnn(x)
                last = out[:, -1, :]
                # 返回的 dict 用 int 作 key：ModuleDict 只能用字符串注册，
                # 但上层用 int 取 head，这里必须换算回去（否则 KeyError）。
                reg = {int(h): self.reg[h](last).squeeze(-1) for h in self.reg}
                clf = {int(h): self.clf[h](last).squeeze(-1) for h in self.clf}
                return reg, clf

        return Net()

    def fit(self, samples: dict, ctx: dict | None = None):
        if not ctx or "panel" not in ctx:
            raise RuntimeError("LSTM 需要 ctx={'panel':..., 'symbol':...}（序列样本）")
        try:
            import torch
        except Exception as e:
            raise RuntimeError(f"未安装 PyTorch：{e}")
        panel = ctx["panel"]
        feat = [c for c in FEATURES if c in panel.columns]
        horizons = sorted(samples.keys())
        cfg = config.LSTM
        ds = _SeqDataset(panel, feat, horizons, cfg["seq_len"])
        if len(ds) < 500:
            raise RuntimeError(f"序列样本不足（{len(ds)} < 500）")
        self.dataset = ds
        self._feat = feat
        self._horizons_req = horizons

        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        net = self._build_net(len(feat), horizons)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        bce = torch.nn.BCEWithLogitsLoss()
        mse = torch.nn.MSELoss()

        # 验证集必须按**时间**切（随机切会让相邻窗口同时出现在训练/验证，指标虚高）
        n = len(ds)
        uniq = sorted(set(ds.end_dates))
        cut_date = uniq[int(len(uniq) * 0.85)] if len(uniq) > 20 else uniq[-1]
        va_idx = np.array([i for i, d in enumerate(ds.end_dates) if d >= cut_date], dtype=int)
        tr_idx = np.array([i for i, d in enumerate(ds.end_dates) if d < cut_date], dtype=int)
        if len(va_idx) < 50 or len(tr_idx) < 200:
            split = int(n * 0.85)
            tr_idx, va_idx = np.arange(split), np.arange(split, n)
        if len(va_idx) > 3000:
            va_idx = va_idx[:: max(1, len(va_idx) // 3000)]

        y_reg = {h: np.asarray(ds.y_reg[h], dtype=np.float32) for h in horizons}
        y_dir = {h: np.asarray(ds.y_dir[h], dtype=np.float32) for h in horizons}
        # 每个周期用**各自的收益标准差**做缩放，而不是固定 0.02：
        # 30 日收益的波动是 1 日的 5 倍以上，用同一个系数会让 30 日样本的 MSE
        # 完全主导总损失（实测 val_loss 卡在 8 左右下不去）。
        self._h_scale = {h: float(np.std(y_reg[h])) or cfg["ret_scale"] for h in horizons}
        bs = cfg["batch_size"]

        def _batch(idx):
            xb = np.stack([ds.get_seq(*ds.index[i]) for i in idx])
            return (torch.from_numpy(xb),
                    {h: torch.from_numpy(y_reg[h][idx] / self._h_scale[h]) for h in horizons},
                    {h: torch.from_numpy(y_dir[h][idx]) for h in horizons})

        best, best_state, bad = float("inf"), None, 0
        for epoch in range(cfg["max_epochs"]):
            net.train()
            perm = np.random.permutation(tr_idx)
            for s in range(0, len(perm), bs):
                idx = perm[s:s + bs]
                if len(idx) < 8:
                    continue
                xb, yb, db = _batch(idx)
                reg_out, clf_out = net(xb)
                loss = sum(mse(reg_out[h], yb[h]) for h in horizons) / len(horizons)
                loss = loss + 0.5 * sum(bce(clf_out[h], db[h]) for h in horizons) / len(horizons)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                opt.step()

            # 验证
            net.eval()
            with torch.no_grad():
                xb, yb, db = _batch(va_idx[:2048])
                reg_out, clf_out = net(xb)
                vloss = sum(mse(reg_out[h], yb[h]) for h in horizons) / len(horizons)
                vloss = float(vloss) + 0.5 * float(
                    sum(bce(clf_out[h], db[h]) for h in horizons) / len(horizons))
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["early_stop"]:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net = net
        self.horizons = horizons
        self.fit_ok = True
        self.note = f"seq_len={cfg['seq_len']} hidden={cfg['hidden_size']} 样本={len(ds)} val_loss={best:.4f}"
        return self

    def predict_horizon(self, h: int, X=None, ctx: dict | None = None):
        import torch
        if not ctx or "panel" not in ctx or "symbol" not in ctx:
            raise RuntimeError("LSTM 预测需要 ctx={'panel':..., 'symbol':...}")
        panel, sym = ctx["panel"], ctx["symbol"]
        g = panel[panel["symbol"] == sym].sort_values("date")
        if len(g) < config.LSTM["seq_len"]:
            raise RuntimeError(f"{sym} 历史不足 {config.LSTM['seq_len']} 根，无法构造序列")
        arr = np.nan_to_num(g[self._feat].to_numpy(dtype=np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        xb = torch.from_numpy(arr[-config.LSTM["seq_len"]:][None])
        with torch.no_grad():
            reg_out, clf_out = self.net(xb)
        r = float(reg_out[int(h)].item()) * self._h_scale.get(int(h), config.LSTM["ret_scale"])
        p = float(torch.sigmoid(clf_out[int(h)]).item())
        return self._post_scale(h, r), p


# ------------------------- 融合 -------------------------
class EnsembleForecaster(BaseForecaster):
    """RF + GBDT(+LSTM) 的概率均值 + 收益均值。

    只用简单平均，不做权重搜索：在样本外只有几十个切分点的情况下，
    搜出来的权重基本是在拟合噪声。

    needs_seq = True：成员里含 LSTM，而 LSTM 预测需要目标股的历史面板
    （ctx={'panel':..., 'symbol':...}）。不声明的话 analysis.py 不会把面板传进来，
    融合里的 LSTM 成员在预测阶段会直接抛错。
    """

    name = "ensemble"
    needs_seq = True
    # 递归预测（逐日合成未来K线）时只用非序列成员：
    # LSTM 需要目标股的历史面板，合成出来的未来K线喂不进面板，硬跑只会报错。
    recurse_needs_seq = False

    def __init__(self, members: list[str] | None = None):
        super().__init__()
        self.members = []
        self.member_names = members or ["rf", "gbdt", "lstm"]
        self.failures: dict = {}
        # 预测阶段掉队的成员（如单股模式没有面板 → LSTM 无法预测）
        self.predict_failures: dict = {}

    def fit(self, samples: dict, ctx: dict | None = None):
        for name in self.member_names:
            try:
                m = build_forecaster(name).fit(samples, ctx)
                if m.fit_ok:
                    self.members.append(m)
            except Exception as e:
                self.failures[name] = str(e)[:120]
        if not self.members:
            raise RuntimeError(f"所有成员模型都训练失败：{self.failures}")
        # 取交集周期，保证每个成员都能给出预测
        hs = set(self.members[0].horizons)
        for m in self.members[1:]:
            hs &= set(m.horizons)
        self.horizons = sorted(hs)
        self.fit_ok = bool(self.horizons)
        self.note = "成员：" + "+".join(m.name for m in self.members)
        if self.failures:
            self.note += "；失败：" + ",".join(self.failures)
        return self

    def predict_horizon(self, h: int, X=None, ctx: dict | None = None):
        rs, ps = [], []
        for m in self.members:
            if h not in m.horizons:
                continue
            try:
                r, p = m.predict_horizon(h, X, ctx)
            except Exception as e:
                # 单个成员预测失败（典型：LSTM 缺面板）不应拖垮整个融合
                self.predict_failures[getattr(m, "name", "member")] = f"{type(e).__name__}: {str(e)[:80]}"
                continue
            rs.append(r)
            ps.append(p)
        if not rs:
            raise RuntimeError(f"所有成员模型预测失败：{self.predict_failures}")
        return float(np.mean(rs)), float(np.mean(ps))

    def predict_members(self, h: int, X=None, ctx: dict | None = None):
        """各成员的 (name, ret, prob_up)。

        成员之间的**分歧**是"认知不确定性"的直接度量：模型越不一致，
        说明这一期的预测越不可靠，目标价位区间就应当给得越宽。
        """
        out = []
        for m in self.members:
            if h not in m.horizons:
                continue
            try:
                r, p = m.predict_horizon(h, X, ctx)
            except Exception as e:
                self.predict_failures[getattr(m, "name", "member")] = f"{type(e).__name__}: {str(e)[:80]}"
                continue
            out.append((getattr(m, "name", "member"), float(r), float(p)))
        return out

    def predict_horizon_no_seq(self, h: int, X=None, ctx: dict | None = None):
        """只用非序列成员（rf/gbdt）预测，供逐日递归合成路径使用。"""
        rs, ps = [], []
        for m in self.members:
            if h not in m.horizons or getattr(m, "needs_seq", False):
                continue
            try:
                r, p = m.predict_horizon(h, X, ctx)
            except Exception as e:
                self.predict_failures[getattr(m, "name", "member")] = f"{type(e).__name__}: {str(e)[:80]}"
                continue
            rs.append(r)
            ps.append(p)
        if not rs:
            raise RuntimeError("融合模型没有可用于递归的非序列成员")
        return float(np.mean(rs)), float(np.mean(ps))

    def predict_batch(self, h: int, X, ctx: dict | None = None):
        rs, ps = [], []
        for m in self.members:
            if h not in m.horizons:
                continue
            r, p = m.predict_batch(h, X, ctx)
            rs.append(r)
            ps.append(p)
        if not rs:
            return np.zeros(len(X)), np.full(len(X), 0.5)
        return np.mean(rs, axis=0), np.mean(ps, axis=0)


def build_forecaster(name: str | None = None) -> BaseForecaster:
    """按名字构造模型（config.MODEL 默认值）。"""
    key = (name or getattr(config, "MODEL", "rf")).lower()
    if key == "gbdt":
        return GBDTForecaster()
    if key == "lstm":
        return LSTMForecaster()
    if key == "ensemble":
        return EnsembleForecaster()
    return RFForecaster()
