"""
Predictive Maintenance AI
=========================
Predicts machine failures from temperature, vibration and power-consumption data.

Pipeline
--------
1. Generate realistic synthetic sensor data (swap in your own CSV, see load_csv()).
2. Engineer time-series features (rolling mean / std / trend per machine).
3. Label each row: "will this machine fail within the next HORIZON hours?"
4. Train a Gradient Boosting classifier (split BY MACHINE to avoid leakage).
5. Evaluate, show feature importance, save model + plots.
6. Provide predict_risk() to score live readings.

Install:  pip install numpy pandas scikit-learn joblib matplotlib
Run:      python predictive_maintenance.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve)

RNG = np.random.default_rng(42)
HORIZON = 24            # predict failure within next 24 hours
WINDOW = 12             # rolling window (hours) for features
SENSORS = ["temperature", "vibration", "power"]
MODEL_PATH = "pm_model.joblib"


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def generate_data(n_machines=60, hours=1500):
    """Simulate machines that degrade before failing, then get repaired."""
    frames = []
    for m in range(n_machines):
        t = np.arange(hours)
        # baseline sensor behaviour (+ daily cycle + noise)
        temp = 60 + 3 * np.sin(2 * np.pi * t / 24) + RNG.normal(0, 1.0, hours)
        vib = 2.0 + 0.2 * np.sin(2 * np.pi * t / 24) + RNG.normal(0, 0.15, hours)
        pwr = 45 + 4 * np.sin(2 * np.pi * t / 24) + RNG.normal(0, 1.5, hours)

        # random failure events; sensors drift upward in the ~48h before each
        n_fail = RNG.integers(2, 5)
        fail_times = np.sort(RNG.choice(np.arange(150, hours - 10), n_fail, replace=False))
        failure = np.zeros(hours, dtype=int)
        for ft in fail_times:
            failure[ft] = 1
            span = 48
            lo = max(0, ft - span)
            ramp = np.linspace(0, 1, ft - lo + 1) ** 1.5
            temp[lo:ft + 1] += 14 * ramp
            vib[lo:ft + 1] += 2.2 * ramp
            pwr[lo:ft + 1] += 12 * ramp

        frames.append(pd.DataFrame({
            "machine_id": m, "hour": t,
            "temperature": temp, "vibration": vib, "power": pwr,
            "failure": failure,
        }))
    return pd.concat(frames, ignore_index=True)


def load_csv(path):
    """Use your own data. Required columns:
    machine_id, hour (or timestamp, sortable), temperature, vibration, power, failure (0/1)."""
    return pd.read_csv(path)


# ----------------------------------------------------------------------------
# 2. FEATURES + LABELS
# ----------------------------------------------------------------------------
def add_features(df):
    df = df.sort_values(["machine_id", "hour"]).copy()
    g = df.groupby("machine_id")
    for s in SENSORS:
        roll = g[s].rolling(WINDOW, min_periods=3)
        df[f"{s}_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{s}_std"] = roll.std().reset_index(level=0, drop=True)
        df[f"{s}_max"] = roll.max().reset_index(level=0, drop=True)
        df[f"{s}_trend"] = g[s].diff(WINDOW)               # change over the window
        df[f"{s}_ewm"] = g[s].transform(lambda x: x.ewm(span=WINDOW).mean())
    # cross-sensor interaction
    df["temp_x_vib"] = df["temperature_mean"] * df["vibration_mean"]
    return df.dropna().reset_index(drop=True)


def add_labels(df):
    """label = 1 if a failure happens within the next HORIZON hours."""
    df = df.sort_values(["machine_id", "hour"]).copy()
    # reverse rolling max of the failure flag over the next HORIZON+1 rows, per machine
    rev = df.iloc[::-1]
    future = (rev.groupby("machine_id")["failure"]
                 .rolling(HORIZON + 1, min_periods=1).max()
                 .reset_index(level=0, drop=True))
    df["will_fail"] = future.reindex(df.index).astype(int)
    return df


def feature_columns(df):
    drop = {"machine_id", "hour", "failure", "will_fail"}
    return [c for c in df.columns if c not in drop]


# ----------------------------------------------------------------------------
# 3. TRAIN / EVALUATE
# ----------------------------------------------------------------------------
def train(df):
    feats = feature_columns(df)
    X, y, groups = df[feats], df["will_fail"], df["machine_id"]

    # 1) hold out 25% of MACHINES as a final test set (never seen in training)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    trainval, te = next(gss.split(X, y, groups))
    # 2) from the remaining machines, hold out 20% as a validation set
    #    used ONLY to choose the alert threshold (avoids test-set tuning)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=7)
    tr_rel, va_rel = next(gss2.split(X.iloc[trainval], y.iloc[trainval],
                                     groups.iloc[trainval]))
    tr, va = trainval[tr_rel], trainval[va_rel]

    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    Xva, yva = X.iloc[va], y.iloc[va]
    Xte, yte = X.iloc[te], y.iloc[te]

    # failures are rare -> weight positives more
    w = np.where(ytr == 1, (ytr == 0).sum() / max((ytr == 1).sum(), 1), 1.0)
    model = GradientBoostingClassifier(n_estimators=250, learning_rate=0.08,
                                       max_depth=3, subsample=0.8, random_state=42)
    model.fit(Xtr, ytr, sample_weight=w)

    # choose threshold on VALIDATION data (maximise F1)
    p, r, th = precision_recall_curve(yva, model.predict_proba(Xva)[:, 1])
    f1 = 2 * p * r / (p + r + 1e-9)
    best = th[np.argmax(f1[:-1])]

    # report on untouched TEST machines
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= best).astype(int)

    print("\n=== RESULTS (unseen test machines) ===")
    print(f"ROC-AUC: {roc_auc_score(yte, proba):.4f}")
    print(f"Alert threshold (chosen on validation set): {best:.3f}")
    print(classification_report(yte, pred, target_names=["healthy", "fails<24h"]))
    print("Confusion matrix:\n", confusion_matrix(yte, pred))

    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nTop features:\n", imp.head(8).round(4))

    joblib.dump({"model": model, "features": feats, "threshold": best}, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")
    return model, feats, best, df.iloc[te], proba, imp


# ----------------------------------------------------------------------------
# 4. PLOTS
# ----------------------------------------------------------------------------
def make_plots(test_df, proba, imp, threshold):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    imp.head(10)[::-1].plot.barh(ax=ax[0], color="steelblue")
    ax[0].set_title("Top 10 feature importances")

    m = test_df["machine_id"].iloc[0]
    mask = (test_df["machine_id"] == m).values
    ax[1].plot(test_df.loc[mask, "hour"], proba[mask], label="failure risk")
    ax[1].axhline(threshold, color="red", ls="--", label="alert threshold")
    for h in test_df.loc[mask & (test_df["failure"] == 1).values, "hour"]:
        ax[1].axvline(h, color="black", alpha=0.6)
    ax[1].set_title(f"Machine {m}: predicted risk (black lines = actual failures)")
    ax[1].set_xlabel("hour")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("pm_results.png", dpi=120)
    print("Plot saved -> pm_results.png")


# ----------------------------------------------------------------------------
# 5. INFERENCE ON NEW READINGS
# ----------------------------------------------------------------------------
def predict_risk(recent_readings: pd.DataFrame, model_path=MODEL_PATH):
    """
    recent_readings: last ~WINDOW+ hourly rows for ONE machine with columns
                     machine_id, hour, temperature, vibration, power
    Returns (probability_of_failure_within_24h, alert_bool)
    """
    bundle = joblib.load(model_path)
    d = recent_readings.copy()
    d["failure"] = 0
    d = add_features(d)
    if d.empty:
        raise ValueError(f"Need at least {WINDOW + 3} hourly rows.")
    prob = bundle["model"].predict_proba(d[bundle["features"]].iloc[[-1]])[0, 1]
    return float(prob), bool(prob >= bundle["threshold"])


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    data = generate_data()                 # or: data = load_csv("your_data.csv")
    print(f"Raw data: {data.shape}, failures: {int(data['failure'].sum())}")
    data = add_labels(add_features(data))
    print(f"After features: {data.shape}, positive rate: {data['will_fail'].mean():.2%}")

    model, feats, thr, test_df, proba, imp = train(data)
    make_plots(test_df, proba, imp, thr)

    # ---- demo: score two simulated live machines ----
    hours = np.arange(40)
    healthy = pd.DataFrame({
        "machine_id": 999, "hour": hours,
        "temperature": 60 + RNG.normal(0, 1, 40),
        "vibration": 2.0 + RNG.normal(0, 0.15, 40),
        "power": 45 + RNG.normal(0, 1.5, 40)})
    failing = pd.DataFrame({
        "machine_id": 998, "hour": hours,
        "temperature": 60 + np.linspace(0, 13, 40) + RNG.normal(0, 1, 40),
        "vibration": 2.0 + np.linspace(0, 2.0, 40) + RNG.normal(0, 0.15, 40),
        "power": 45 + np.linspace(0, 11, 40) + RNG.normal(0, 1.5, 40)})

    for name, df_live in [("Healthy machine", healthy), ("Degrading machine", failing)]:
        p, alert = predict_risk(df_live)
        print(f"{name}: risk={p:.1%} -> {'ALERT: schedule maintenance' if alert else 'OK'}")
