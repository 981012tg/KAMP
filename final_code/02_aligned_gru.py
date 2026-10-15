"""02_main_pipeline의 LightGBM 실험 조건을 GRU로 재현하는 03-2 지원 코드.

노트북에서 exec(..., globals())로 실행한다. ROOT, PREP, display가 정의되어 있어야 한다.
"""
import json
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression

AW_ROOT = ROOT
AW_PREP = AW_ROOT / "results/preprocessing_24h"
AW_OUT = AW_ROOT / "results/gru_24h_team_preprocessing/aligned_with_team_lgbm"
AW_OUT.mkdir(parents=True, exist_ok=True)
AW_NOTEBOOK = Path("02_main_pipeline.ipynb")
if not AW_NOTEBOOK.exists():
    AW_NOTEBOOK = AW_ROOT / "real_final/02_main_pipeline.ipynb"
aw_team_notebook = json.loads(AW_NOTEBOOK.read_text(encoding="utf-8"))

# 팀 LGBM과 피처 계산이 달라지지 않도록 02의 함수 정의를 그대로 사용한다.
for aw_cell_index in [17, 19, 21]:
    exec("".join(aw_team_notebook["cells"][aw_cell_index]["source"]), globals())
ROOT = AW_ROOT
PREP = AW_PREP
CFG = json.loads((PREP / "config.json").read_text(encoding="utf-8"))
SEQ = CFG["sequence_features"]
log = print
exec("".join(aw_team_notebook["cells"][25]["source"]), globals())

AW_PEAK = 176.0
AW_WEEKS = pd.date_range("2021-05-31", "2021-07-26", freq="7D")
AW_EPOCHS = 9
AW_BATCH = 64
AW_LR = 1e-3
AW_WD = 1e-4
AW_SEED = 42
assert len(W.BASE) == 30 and len(W.L13) == 47
assert len(W.ev_val) == 627 and W.y[W.ev_val].size == 60192
assert W.meta.loc[W.ev_val, "origin"].dt.month.eq(7).all()

def aw_gru_input(columns):
    values = W.F[columns].to_numpy(np.float32)
    values = values.reshape(W.N, 24, 4, len(columns))
    return values.reshape(W.N, 24, 4 * len(columns))

AW_X30 = aw_gru_input(W.BASE)
AW_X47 = aw_gru_input(W.L13)
assert AW_X30.shape == (W.N, 24, 120)
assert AW_X47.shape == (W.N, 24, 188)
assert np.isfinite(AW_X30).all() and np.isfinite(AW_X47).all()

class AWGRU(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.gru = nn.GRU(
            input_features, 32, num_layers=2, batch_first=True,
            dropout=0.15, bidirectional=True,
        )
        self.output = nn.Linear(64, 4)

    def forward(self, values):
        sequence_output, _ = self.gru(values)
        return self.output(sequence_output)

def aw_fit_predict(
    x_train, target_train, x_predict, objective,
    alpha=0.5, cell_weight=None,
):
    if cell_weight is None:
        cell_weight = np.ones_like(target_train, dtype=np.float32)
    cell_weight = cell_weight.astype(np.float32)
    active = cell_weight > 0
    if not active.any():
        raise ValueError("학습 대상 칸이 없습니다.")

    x_mean = x_train.mean(axis=(0, 1), keepdims=True)
    x_std = x_train.std(axis=(0, 1), keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    target_mean = float(target_train[active].mean())
    target_std = float(target_train[active].std())
    if target_std < 1e-6:
        target_std = 1.0

    x_train = ((x_train - x_mean) / x_std).astype(np.float32)
    x_predict = ((x_predict - x_mean) / x_std).astype(np.float32)
    target_train = (
        (target_train - target_mean) / target_std
    ).astype(np.float32)

    np.random.seed(AW_SEED)
    torch.manual_seed(AW_SEED)
    generator = torch.Generator().manual_seed(AW_SEED)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(target_train),
            torch.from_numpy(cell_weight),
        ),
        batch_size=AW_BATCH,
        shuffle=True,
        generator=generator,
    )
    model = AWGRU(x_train.shape[-1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=AW_LR, weight_decay=AW_WD
    )
    history = []
    for _ in range(AW_EPOCHS):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for batch_x, batch_target, batch_weight in loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            error = batch_target - prediction
            if objective == "quantile":
                element = torch.maximum(
                    alpha * error, (alpha - 1) * error
                )
            elif objective == "l1":
                element = error.abs()
            else:
                raise ValueError(objective)
            denominator = batch_weight.sum().clamp_min(1.0)
            loss = (element * batch_weight).sum() / denominator
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float((element * batch_weight).sum())
            weight_sum += float(batch_weight.sum())
        history.append(loss_sum / max(weight_sum, 1.0))

    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x_predict), 512):
            parts.append(
                model(torch.from_numpy(x_predict[start:start + 512])).numpy()
            )
    prediction = np.concatenate(parts) * target_std + target_mean
    return prediction.astype(np.float32), history

# 02와 같은 이례적 휴무 구간과 사후 안전장치.
aw_daily_production = pd.Series(W.hi.P).groupby(W.hi.date_d).sum()
aw_weekday_off = (
    aw_daily_production.le(0)
    & (pd.DatetimeIndex(aw_daily_production.index).dayofweek < 5)
)
AW_STREAKS = []
aw_run = []
for aw_day, aw_is_off in aw_weekday_off.items():
    if aw_is_off:
        aw_run.append(aw_day)
    else:
        if len(aw_run) >= 3:
            AW_STREAKS.append((aw_run[0], aw_run[-1]))
        aw_run = []
if len(aw_run) >= 3:
    AW_STREAKS.append((aw_run[0], aw_run[-1]))

def aw_known_off_windows(week_start):
    known = np.zeros(W.N, bool)
    for first_day, last_day in AW_STREAKS:
        if last_day + pd.Timedelta(days=1) <= week_start:
            known |= (
                W.meta["origin"].lt(last_day + pd.Timedelta(days=1))
                & W.meta["target_end"].gt(first_day)
            ).to_numpy()
    return known

AW_SCHEDULE = []

def aw_weekly_predict(
    candidate, inputs, target, objective, alpha=0.5,
    training_cells=None, use_off_weight=True,
):
    prediction = np.full_like(target, np.nan, dtype=np.float32)
    for week_start in AW_WEEKS:
        train_rows = W.meta["target_end"].le(week_start).to_numpy()
        predict_rows = (
            W.meta["origin"].ge(week_start)
            & W.meta["origin"].lt(week_start + pd.Timedelta(days=7))
        ).to_numpy()
        if not predict_rows.any():
            continue
        assert W.meta.loc[train_rows, "target_end"].max() <= week_start
        weights = np.ones_like(target[train_rows], dtype=np.float32)
        known = aw_known_off_windows(week_start)
        if use_off_weight:
            weights *= np.where(
                known[train_rows, None, None], 0.1, 1.0
            ).astype(np.float32)
        if training_cells is not None:
            weights *= training_cells[train_rows].astype(np.float32)

        started = time.time()
        week_prediction, history = aw_fit_predict(
            inputs[train_rows], target[train_rows], inputs[predict_rows],
            objective, alpha=alpha, cell_weight=weights,
        )
        prediction[predict_rows] = week_prediction
        AW_SCHEDULE.append({
            "candidate": candidate,
            "week_start": str(week_start.date()),
            "training_windows": int(train_rows.sum()),
            "prediction_windows": int(predict_rows.sum()),
            "downweighted_windows": int(
                (known & train_rows).sum() if use_off_weight else 0
            ),
            "epochs": AW_EPOCHS,
            "last_train_loss": history[-1],
            "seconds": time.time() - started,
        })
    return prediction

# 피크경보: 현재 Q1과 개선 Q1에 대응하는 GRU 후보.
aw_static_q090, aw_static_history = aw_fit_predict(
    AW_X30[W.tr_val], W.y3[W.tr_val],
    AW_X30[W.ev_val], "quantile", alpha=0.90,
)
aw_static_q090 = np.clip(aw_static_q090.reshape(-1, 96), 0, None)
AW_SCHEDULE.append({
    "candidate": "gru_current_static_30_q090",
    "week_start": "2021-07-01",
    "training_windows": int(W.tr_val.sum()),
    "prediction_windows": int(len(W.ev_val)),
    "downweighted_windows": 0,
    "epochs": AW_EPOCHS,
    "last_train_loss": aw_static_history[-1],
    "seconds": np.nan,
})

AW_PEAK_SPECS = [
    ("gru_official30_weekly_q075", AW_X30, 0.75, True),
    ("gru_enhanced47_weekly_q075_no_off_weight", AW_X47, 0.75, False),
    ("gru_enhanced47_weekly_q075", AW_X47, 0.75, True),
    ("gru_enhanced47_weekly_q080", AW_X47, 0.80, True),
    ("gru_enhanced47_weekly_q090", AW_X47, 0.90, True),
]
AW_PEAK_PREDICTIONS = {}
for candidate, inputs, alpha, use_off_weight in AW_PEAK_SPECS:
    AW_PEAK_PREDICTIONS[candidate] = np.clip(
        aw_weekly_predict(
            candidate, inputs, W.y3, "quantile",
            alpha=alpha, use_off_weight=use_off_weight,
        ).reshape(W.N, 96),
        0, None,
    )

# 수치예측: 3단 규칙 잔차, 학습 범위, 휴무일 처리를 분리해 비교.
AW_RULE = W.F["rule3_oof"].to_numpy().reshape(W.N, 96).astype(np.float32)
AW_GATE = W.F["st_day_on"].to_numpy().reshape(W.N, 24, 4) == 0
AW_RESIDUAL_TARGET = (
    W.y - AW_RULE
).reshape(W.N, 24, 4).astype(np.float32)
aw_curve_all = aw_weekly_predict(
    "gru_curve_residual_all_cells", AW_X47,
    AW_RESIDUAL_TARGET, "l1",
    training_cells=None, use_off_weight=False,
).reshape(W.N, 96) + AW_RULE
aw_curve_on = aw_weekly_predict(
    "gru_curve_residual_operating_cells", AW_X47,
    AW_RESIDUAL_TARGET, "l1",
    training_cells=~AW_GATE, use_off_weight=False,
).reshape(W.N, 96) + AW_RULE
aw_curve_all = np.clip(aw_curve_all, 0, None)
aw_curve_on = np.clip(aw_curve_on, 0, None)

# 02와 같은 과거 휴무일 시간별 중앙값.
aw_hourly = pd.read_csv(
    AW_PREP / "hourly_review.csv", parse_dates=["origin"]
)
aw_hourly["date"] = aw_hourly["origin"].dt.floor("D")
aw_hourly["hour"] = aw_hourly["origin"].dt.hour
aw_power_columns = ["15분", "30분", "45분", "60분"]
aw_day_total = aw_hourly.groupby("date")["생산량"].sum()
aw_off_days = aw_day_total[aw_day_total.eq(0)].index
aw_by_day = {
    day: group.sort_values("hour")[aw_power_columns].to_numpy()
    for day, group in aw_hourly.groupby("date") if len(group) == 24
}
AW_IDLE = np.full((W.N, 96), np.nan, np.float32)
aw_idle_cache = {}
aw_dates = W.meta["origin"].dt.floor("D").to_numpy()
aw_origin_hours = W.meta["origin"].dt.hour.to_numpy()
for index in range(W.N):
    day = pd.Timestamp(aw_dates[index])
    if day not in aw_idle_cache:
        previous_off = [
            value for value in aw_off_days
            if value < day and value in aw_by_day
        ]
        aw_idle_cache[day] = (
            np.median(
                np.stack([aw_by_day[value] for value in previous_off]),
                axis=0,
            ) if len(previous_off) >= 3 else None
        )
    profile = aw_idle_cache[day]
    if profile is not None:
        AW_IDLE[index] = profile[
            (
                aw_origin_hours[index] + np.repeat(np.arange(24), 4)
            ) % 24,
            np.tile(np.arange(4), 24),
        ]
aw_gate_flat = AW_GATE.reshape(W.N, 96)
aw_curve_on_rule_off = np.where(
    aw_gate_flat, AW_RULE, aw_curve_on
).astype(np.float32)
aw_curve_final = np.where(
    aw_gate_flat & np.isfinite(AW_IDLE),
    AW_IDLE,
    aw_curve_on_rule_off,
).astype(np.float32)

def aw_metrics(target, prediction, threshold=AW_PEAK):
    threshold_values = (
        np.broadcast_to(
            np.asarray(threshold, float).reshape(-1, 1), target.shape
        ) if np.ndim(threshold) else threshold
    )
    actual = target >= AW_PEAK
    alarm = prediction >= threshold_values
    tp = int((actual & alarm).sum())
    fp = int((~actual & alarm).sum())
    fn = int((actual & ~alarm).sum())
    tn = int((~actual & ~alarm).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if precision + recall else 0.0
    )
    error = prediction - target
    return {
        "observations": int(target.size),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "bias": float(error.mean()),
        "precision": precision, "recall": recall,
        "f1": f1, "f2": f2,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

# 개선 Q1과 같은 최근 4주 R85/R95 경보 기준.
AW_WEEK_OF = (
    (W.meta["origin"] - AW_WEEKS[0]) // pd.Timedelta(days=7)
).to_numpy()
def aw_weekly_thresholds(prediction, target_policy):
    thresholds = np.full(W.N, AW_PEAK, dtype=float)
    for week in np.unique(AW_WEEK_OF[W.ev_val]):
        week_start = AW_WEEKS[0] + pd.Timedelta(days=7) * int(week)
        history = (
            np.isfinite(prediction[:, 0])
            & W.meta["target_end"].le(week_start).to_numpy()
            & W.meta["origin"].ge(
                week_start - pd.Timedelta(days=28)
            ).to_numpy()
            & W.meta["origin"].ge(AW_WEEKS[0]).to_numpy()
        )
        if history.any():
            thresholds[AW_WEEK_OF == week] = choose(
                W.y[history], prediction[history], target_policy
            )
    return thresholds

AW_PRIMARY_PEAK_NAME = "gru_enhanced47_weekly_q075"
AW_PRIMARY_PEAK = AW_PEAK_PREDICTIONS[AW_PRIMARY_PEAK_NAME]
AW_THRESHOLD_R85 = aw_weekly_thresholds(AW_PRIMARY_PEAK, "R85")
AW_THRESHOLD_R95 = aw_weekly_thresholds(AW_PRIMARY_PEAK, "R95")

# 같은 최근 4주 결과만 사용한 피크 확률 보정.
AW_PROBABILITY = np.full((W.N, 96), np.nan, np.float32)
aw_labels = W.y >= AW_PEAK
for week in np.unique(AW_WEEK_OF[W.ev_val]):
    week_start = AW_WEEKS[0] + pd.Timedelta(days=7) * int(week)
    week_rows = AW_WEEK_OF == week
    history = (
        np.isfinite(AW_PRIMARY_PEAK[:, 0])
        & W.meta["target_end"].le(week_start).to_numpy()
        & W.meta["origin"].ge(
            week_start - pd.Timedelta(days=28)
        ).to_numpy()
        & W.meta["origin"].ge(AW_WEEKS[0]).to_numpy()
    )
    history_labels = aw_labels[history].reshape(-1)
    if history.any() and np.unique(history_labels).size == 2:
        calibration = LogisticRegression(C=1e4, max_iter=200)
        calibration.fit(
            (
                AW_PRIMARY_PEAK[history].reshape(-1, 1) - AW_PEAK
            ) / 20.0,
            history_labels,
        )
        AW_PROBABILITY[week_rows] = calibration.predict_proba(
            (
                AW_PRIMARY_PEAK[week_rows].reshape(-1, 1) - AW_PEAK
            ) / 20.0
        )[:, 1].reshape(-1, 96)

aw_peak_rows = [{
    "candidate": "gru_current_static_30_q090",
    "input_features": 30,
    "retraining": "once_before_july",
    "quantile": 0.90,
    "off_streak_weight": 1.0,
    "threshold_policy": "fixed_176",
    **aw_metrics(W.y[W.ev_val], aw_static_q090),
}]
for candidate, _, alpha, use_off_weight in AW_PEAK_SPECS:
    prediction = AW_PEAK_PREDICTIONS[candidate]
    aw_peak_rows.append({
        "candidate": candidate,
        "input_features": 30 if "official30" in candidate else 47,
        "retraining": "weekly",
        "quantile": alpha,
        "off_streak_weight": 0.1 if use_off_weight else 1.0,
        "threshold_policy": "fixed_176",
        **aw_metrics(W.y[W.ev_val], prediction[W.ev_val]),
    })
aw_peak_rows.append({
    "candidate": AW_PRIMARY_PEAK_NAME,
    "input_features": 47,
    "retraining": "weekly",
    "quantile": 0.75,
    "off_streak_weight": 0.1,
    "threshold_policy": "R85_recent_4_weeks",
    **aw_metrics(
        W.y[W.ev_val], AW_PRIMARY_PEAK[W.ev_val],
        AW_THRESHOLD_R85[W.ev_val],
    ),
})
aw_peak_results = pd.DataFrame(aw_peak_rows)
aw_peak_results.to_csv(
    AW_OUT / "aligned_peak_metrics_validation.csv",
    index=False, encoding="utf-8-sig",
)

AW_CURVE_PREDICTIONS = {
    "previous_week_baseline": W.base,
    "rule3_oof": AW_RULE,
    "gru_residual_all_cells": aw_curve_all,
    "gru_residual_operating_cells_rule_off": aw_curve_on_rule_off,
    "gru_residual_operating_cells_idle_profile": aw_curve_final,
}
aw_curve_rows = []
for candidate, prediction in AW_CURVE_PREDICTIONS.items():
    aw_curve_rows.append({
        "candidate": candidate,
        "training_schedule": (
            "weekly" if candidate.startswith("gru_") else "none"
        ),
        "training_cells": (
            "all" if candidate == "gru_residual_all_cells"
            else "operating_only" if candidate.startswith("gru_")
            else "not_applicable"
        ),
        "off_day_output": (
            "idle_profile" if candidate.endswith("idle_profile")
            else "rule3" if candidate.endswith("rule_off")
            else "model_or_baseline"
        ),
        **aw_metrics(W.y[W.ev_val], prediction[W.ev_val]),
    })
aw_curve_results = pd.DataFrame(aw_curve_rows)
aw_curve_results.to_csv(
    AW_OUT / "aligned_curve_metrics_validation.csv",
    index=False, encoding="utf-8-sig",
)
pd.DataFrame(AW_SCHEDULE).to_csv(
    AW_OUT / "training_schedule.csv",
    index=False, encoding="utf-8-sig",
)

aw_origins = W.meta.loc[W.ev_val, "origin"].to_numpy()
aw_steps = np.tile(np.arange(96), len(W.ev_val))
aw_origin_column = np.repeat(aw_origins, 96)
aw_target_time = (
    pd.to_datetime(aw_origin_column)
    + pd.to_timedelta(aw_steps // 4, unit="h")
    + pd.to_timedelta(15 * (aw_steps % 4 + 1), unit="m")
)
aw_prediction_table = pd.DataFrame({
    "기준시각": aw_origin_column,
    "예측대상시각": aw_target_time,
    "예측칸": aw_steps,
    "실제전력": W.y[W.ev_val].reshape(-1),
    "baseline_w1": W.base[W.ev_val].reshape(-1),
    "rule3_oof": AW_RULE[W.ev_val].reshape(-1),
    "gru_current_static_30_q090": aw_static_q090.reshape(-1),
    **{
        name: prediction[W.ev_val].reshape(-1)
        for name, prediction in AW_PEAK_PREDICTIONS.items()
    },
    **{
        name: prediction[W.ev_val].reshape(-1)
        for name, prediction in AW_CURVE_PREDICTIONS.items()
        if name.startswith("gru_")
    },
    "fixed_threshold": AW_PEAK,
    "r85_recent_4_weeks_threshold": np.repeat(
        AW_THRESHOLD_R85[W.ev_val], 96
    ),
    "r95_recent_4_weeks_threshold": np.repeat(
        AW_THRESHOLD_R95[W.ev_val], 96
    ),
    "peak_probability": AW_PROBABILITY[W.ev_val].reshape(-1),
})
aw_prediction_table.to_csv(
    AW_OUT / "aligned_predictions_validation.csv",
    index=False, encoding="utf-8-sig",
)

aw_manifest = {
    "shared_preprocessing": str(AW_PREP),
    "team_feature_source": str(AW_NOTEBOOK),
    "source_sha256": CFG["source_sha256"],
    "evaluation_period": "2021-07 primary_eligible",
    "evaluation_windows": int(len(W.ev_val)),
    "evaluation_cells": int(W.y[W.ev_val].size),
    "actual_peak_threshold": AW_PEAK,
    "weekly_retraining": [str(value.date()) for value in AW_WEEKS],
    "official_features": W.BASE,
    "enhanced_features": W.L13,
    "seed": AW_SEED,
    "epochs": AW_EPOCHS,
    "batch_size": AW_BATCH,
    "learning_rate": AW_LR,
    "weight_decay": AW_WD,
    "test_used_for_training_or_selection": False,
}
(AW_OUT / "experiment_manifest.json").write_text(
    json.dumps(aw_manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("피크경보 동일 조건 GRU 실험")
display(aw_peak_results.round(4))
print("수치예측 동일 조건 GRU 실험")
display(
    aw_curve_results[["candidate", "mae", "rmse", "bias"]].round(4)
)
print("저장 완료:", AW_OUT.resolve())
