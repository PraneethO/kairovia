Skip to content
Chat history

You said:
# weather_pipeline_improved.py
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from datetime import datetime, timedelta

# ---------------------------
# Config / Hyperparameters
# ---------------------------
DATA_PATH = "data.csv"                     # change if needed
DATE_COL = "DATE"
FEATURE_COLS = ["TMP","DEWP","SLP","WDIR","WSPD","PRCP"]
INPUT_HOURS = 12                           # first 12 hours (0..11)
SPLIT_DATE = None                          # if None, use 80/20 time-based split automatically
LGB_PARAMS = {"n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 31, "verbose": -1}
TF_LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 80
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ---------------------------
# Utility / defensive funcs
# ---------------------------
def assert_no_nans(arr, name="array"):
    if np.isnan(arr).any():
        raise ValueError(f"NaN detected in {name} (shape={arr.shape})")
    if np.isinf(arr).any():
        raise ValueError(f"Inf detected in {name} (shape={arr.shape})")

def safe_inverse_transform(scaler, arr):
    """scaler.inverse_transform but robust to shape issues"""
    return scaler.inverse_transform(np.asarray(arr).reshape(-1,1)).reshape(-1)

# ---------------------------
# 1) Load & basic cleaning
# ---------------------------
df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

# normalize column names (in case)
df.columns = [c.strip() for c in df.columns]

if DATE_COL not in df.columns:
    raise ValueError(f"{DATE_COL} not present in CSV")

# Extract date & hour
df["date"] = df[DATE_COL].dt.date
df["hour"] = df[DATE_COL].dt.hour

# ---------------------------
# 2) Feature engineering (per-row)
# ---------------------------
# Cyclical time features
df["doy"] = df[DATE_COL].dt.dayofyear
df["doy_sin"] = np.sin(2*np.pi * df["doy"] / 365.25)
df["doy_cos"] = np.cos(2*np.pi * df["doy"] / 365.25)
df["hour_sin"] = np.sin(2*np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2*np.pi * df["hour"] / 24.0)

# Optionally compute relative humidity if you have dewpoint & temp (Magnus approximation)
# Only add if TMP & DEWP exist and are not NaN
if {"TMP", "DEWP"}.issubset(df.columns):
    # ensure degrees are in Celsius; if in Kelvin/Fahrenheit convert appropriately beforehand
    a = 17.625
    b = 243.04
    alpha_dewp = (a * df["DEWP"]) / (b + df["DEWP"])
    alpha_tmp  = (a * df["TMP"]) / (b + df["TMP"])
    # avoid NaN by clipping
    with np.errstate(all='ignore'):
        rh = 100.0 * (np.exp(alpha_dewp) / np.exp(alpha_tmp))
    df["RH"] = rh.clip(0, 100).fillna(np.nan)

# ---------------------------
# 3) Per-day aggregations and rolling features (use previous days)
# ---------------------------
# We'll compute for each date:
# - first 12h sequence of core features (TMP, DEWP, SLP, WDIR, WSPD, PRCP)
# - previous-day stats: prev_max_temp, prev_min_temp, prev_mean_temp
# - 24h rolling mean/std for TMP (using the full hourly data)
df = df.sort_values(DATE_COL)
# hourly rolling features: use center=False (past data)
df["TMP_roll24_mean"] = df.groupby("date")["TMP"].transform(lambda x: x.rolling(window=min(24, max(1, len(x))), min_periods=1).mean())
# For a robust rolling across the full timeseries, use groupby on whole df:
df["TMP_rolling_24h"] = df["TMP"].rolling(window=24, min_periods=1).mean()

# Build per-day dataset
X_list, y_list, meta_dates = [], [], []

# To avoid throwing away days with a single NaN, we'll impute hour-level missingness BEFORE we build sequences.
# Simple robust imputation: linear interpolation per day then forward/backfill; if still NaN, fill with global median.
for col in FEATURE_COLS + ["RH", "TMP_rolling_24h", "TMP_roll24_mean", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]:
    if col in df.columns:
        # per-day interpolation helps small gaps in the 12-hour window
        df[col] = df.groupby("date")[col].transform(lambda s: s.interpolate(limit_direction='both'))# final global fill
for col in FEATURE_COLS + ["RH", "TMP_rolling_24h", "TMP_roll24_mean", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]:
    if col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median(skipna=True)
            df[col] = df[col].fillna(median_val)

# Build sequences: for each date, collect first INPUT_HOURS rows sorted by hour
for date, g in df.groupby("date"):
    g = g.sort_values("hour")
    if g.shape[0] < INPUT_HOURS:
        continue
    seq = g[g["hour"] < INPUT_HOURS]  # hours 0..INPUT_HOURS-1
    if seq.shape[0] != INPUT_HOURS:
        continue
    # core sequence features
    seq_feats = seq[FEATURE_COLS].values  # shape (INPUT_HOURS, n_features)
    # additional static features for that day (avg of first 12 hours or aggregated)
    prev_day = date - timedelta(days=1)
    prev_day_rows = df[df["date"] == prev_day]
    prev_max = prev_day_rows["TMP"].max() if not prev_day_rows.empty else np.nan
    prev_min = prev_day_rows["TMP"].min() if not prev_day_rows.empty else np.nan
    prev_mean = prev_day_rows["TMP"].mean() if not prev_day_rows.empty else np.nan

    # fill prev_* NaN with recent history median
    if np.isnan(prev_max):
        prev_max = df.loc[df["date"] < date, "TMP"].tail(7).max() if not df.loc[df["date"] < date, "TMP"].empty else np.nan
    if np.isnan(prev_min):
        prev_min = df.loc[df["date"] < date, "TMP"].tail(7).min() if not df.loc[df["date"] < date, "TMP"].empty else np.nan
    if np.isnan(prev_mean):
        prev_mean = df.loc[df["date"] < date, "TMP"].tail(7).mean() if not df.loc[df["date"] < date, "TMP"].empty else np.nan
    # safety fill
    prev_max = prev_max if not np.isnan(prev_max) else df["TMP"].median()
    prev_min = prev_min if not np.isnan(prev_min) else df["TMP"].median()
    prev_mean = prev_mean if not np.isnan(prev_mean) else df["TMP"].median()

    # slope of TMP in first 12 hours (simple linear fit)
    x_hours = np.arange(INPUT_HOURS)
    y_tmp = seq["TMP"].values
    # linear slope (least squares)
    A = np.vstack([x_hours, np.ones_like(x_hours)]).T
    try:
        slope, intercept = np.linalg.lstsq(A, y_tmp, rcond=None)[0]
    except Exception:
        slope = 0.0

    # collect features
    # We'll create a flattened version of the sequence (INPUT_HOURS * n_features)
    flat_seq = seq_feats.flatten()
    # static features
    static_feats = np.array([
        prev_max, prev_min, prev_mean,
        seq["TMP_rolling_24h"].mean(),
        seq["TMP_roll24_mean"].mean(),
        slope,
        seq["hour_sin"].mean(), seq["hour_cos"].mean(),
        seq["doy_sin"].mean(), seq["doy_cos"].mean()
    ], dtype=float)

    X_list.append(np.concatenate([flat_seq, static_feats], axis=0))
    # Target: daily maximum temperature (computed from full day's TMP)
    y_list.append(g["TMP"].max())
    meta_dates.append(date)

X = np.array(X_list)
y = np.array(y_list)
meta_dates = np.array(meta_dates)

print("Built dataset: X.shape =", X.shape, "y.shape =", y.shape)
assert_no_nans(X, "X after feature build")
assert_no_nans(y, "y after feature build")

# ---------------------------
# 4) Time-based split (train=older, test=newer)
# ---------------------------
if SPLIT_DATE is None:
    # use 80% of chronological samples as training
    n = len(meta_dates)
    train_end_idx = int(n * 0.8)
    train_idx = np.arange(0, train_end_idx)
    test_idx = np.arange(train_end_idx, n)
else:
    split_date_obj = pd.to_datetime(SPLIT_DATE).date()
    train_idx = np.where(meta_dates < split_date_obj)[0]
    test_idx = np.where(meta_dates >= split_date_obj)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Bad SPLIT_DATE results in empty train or test")

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

# ---------------------------
# 5) Scaling (fit on train only)
# ---------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).reshape(-1)
y_test_scaled = scaler_y.transform(y_test.reshape(-1,1)).reshape(-1)

# final checks to avoid NaNs in training loop
assert_no_nans(X_train_scaled, "X_train_scaled")
assert_no_nans(X_test_scaled, "X_test_scaled")
assert_no_nans(y_train_scaled, "y_train_scaled")
assert_no_nans(y_test_scaled, "y_test_scaled")

# ---------------------------
# 6) LightGBM baseline (point forecast median)
# ---------------------------
print("\nTraining LightGBM baseline (point estimator)...")
lgb_model = lgb.LGBMRegressor(**LGB_PARAMS, random_state=RANDOM_STATE)
lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], callbacks=[lgb.early_stopping(stopping_rounds=50),  # replaces early_stopping_rounds
        lgb.log_evaluation(10)                   # log every 10 rounds
])

lgb_pred = lgb_model.predict(X_test_scaled)
lgb_mse = mean_squared_error(y_test, lgb_pred)
lgb_mae = mean_absolute_error(y_test, lgb_pred)
print(f"LightGBM - MSE: {lgb_mse:.3f}, MAE: {lgb_mae:.3f}")

# ---------------------------
# 7) LightGBM quantile models (0.1, 0.5, 0.9)
# ---------------------------
print("\nTraining LightGBM quantile models (0.1, 0.5, 0.9)...")
quantiles = [0.1, 0.5, 0.9]
lgb_quant_preds = {}
for q in quantiles:
    params = LGB_PARAMS.copy()
    # Using scikit-learn wrapper that supports objective='quantile' + alpha
    params["objective"] = "quantile"
    params["alpha"] = q
    model_q = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE)
    model_q.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(10)]
)    
    lgb_quant_preds[q] = model_q.predict(X_test_scaled)

# ---------------------------
# 8) Heteroscedastic Keras model (mu and log_sigma)
# ---------------------------
print("\nBuilding heteroscedastic Keras model...")

# We'll treat the flattened feature vector as input to a dense NN (not sequence LSTM)
# If you prefer preserving time dimension, you can reshape and use Conv1D/LSTM/Transformer variants.
n_input = X_train_scaled.shape[1]
tf.keras.backend.set_floatx('float32')

def nll_heteroscedastic(y_true, y_pred):
    # y_pred: concatenated [mu, log_sigma]
    mu = y_pred[:, 0]
    log_sigma = y_pred[:, 1]
    sigma = tf.exp(log_sigma) + 1e-6
    # Negative log-likelihood of Normal
    return tf.reduce_mean(0.5 * tf.math.log(2.0 * np.pi) + log_sigma + 0.5 * ((y_true - mu)**2) / (sigma**2))

# Build model
inp = Input(shape=(n_input,), dtype="float32")
x = layers.Dense(256, activation="relu")(inp)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
mu = layers.Dense(1, activation="linear")(x)
log_sigma = layers.Dense(1, activation="linear")(x)  # can be any real number
out = layers.Concatenate()([mu, log_sigma])
model = models.Model(inputs=inp, outputs=out)

opt = tf.keras.optimizers.Adam(learning_rate=TF_LEARNING_RATE, clipnorm=1.0)
model.compile(optimizer=opt, loss=nll_heteroscedastic, metrics=[])

model.summary()

# Defensive callbacks: stop on NaN if any, early stopping and reduce LR
callbacks = [
    TerminateOnNaN(),  # ensures training stops if NaN appears (you asked to avoid NaNs)
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-6, verbose=1)
]

# Prepare targets for heteroscedastic model: use scaled y
# Note: the loss uses raw y_true and outputs mu/log_sigma; we can feed scaled or original values consistently.
# We'll feed scaled targets for numerical stability, but ensure consistency with inverse transforms when predicting.
history = model.fit(
    X_train_scaled.astype(np.float32),
    y_train_scaled.astype(np.float32),
    validation_data=(X_test_scaled.astype(np.float32), y_test_scaled.astype(np.float32)),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)

# After training, predict mu & sigma on test set (in scaled space)
y_pred_het = model.predict(X_test_scaled.astype(np.float32))
mu_scaled = y_pred_het[:, 0]
log_sigma = y_pred_het[:, 1]
sigma_scaled = np.exp(log_sigma)

# Inverse-transform mu back to original scale
mu_pred = safe_inverse_transform(scaler_y, mu_scaled)
# approximate sigma in original units: multiply scaled sigma by scaler_y.scale_
sigma_pred = sigma_scaled * scaler_y.scale_[0]

# Basic checks
assert_no_nans(mu_pred, "mu_pred (keras)")
assert_no_nans(sigma_pred, "sigma_pred (keras)")

# Evaluate heteroscedastic mu as point forecast
mse_het = mean_squared_error(y_test, mu_pred)
mae_het = mean_absolute_error(y_test, mu_pred)
print(f"Heteroscedastic NN - MSE: {mse_het:.3f}, MAE: {mae_het:.3f}")

# ---------------------------
# 9) Ensemble: average LightGBM & Keras mu
# ---------------------------
ensemble_pred = 0.5 * lgb_pred + 0.5 * mu_pred
mse_ens = mean_squared_error(y_test, ensemble_pred)
mae_ens = mean_absolute_error(y_test, ensemble_pred)
print(f"\nEnsemble (50/50 LightGBM + NN) - MSE: {mse_ens:.3f}, MAE: {mae_ens:.3f}")

# Ensemble quantiles: average LGB quantiles + NN mu as central tendency
# For 0.1/0.5/0.9, we have lgb_quant_preds; combine by simple average with NN mu for median
q_preds = {
    0.1: lgb_quant_preds[0.1],
    0.5: 0.5 * lgb_quant_preds[0.5] + 0.5 * mu_pred,  # blend median
    0.9: lgb_quant_preds[0.9]
}

# ---------------------------
# 10) Diagnostics & summary
# ---------------------------
errors = ensemble_pred - y_test
print("\nSUMMARY:")
print(f"  LightGBM MAE: {lgb_mae:.3f}")
print(f"  Heteroscedastic NN MAE: {mae_het:.3f}")
print(f"  Ensemble MAE: {mae_ens:.3f}")
print(f"  Ensemble Bias (mean error): {np.mean(errors):.3f}, Std error: {np.std(errors):.3f}")

# Error percentiles
abs_err = np.abs(errors)
p50, p75, p90, p95 = np.percentile(abs_err, [50, 75, 90, 95])
print(f"  Ensemble absolute error percentiles: 50% {p50:.3f}, 75% {p75:.3f}, 90% {p90:.3f}, 95% {p95:.3f}")

# Print best/worst predictions (first 5 each)
idx_sorted = np.argsort(np.abs(errors))
print("\nBest predictions (top 5):")
for i in idx_sorted[:5]:
    print(f"  Pred={ensemble_pred[i]:.2f}, True={y_test[i]:.2f}, Err={errors[i]:+.2f}")

print("\nWorst predictions (top 5):")
for i in idx_sorted[-5:]:
    print(f"  Pred={ensemble_pred[i]:.2f}, True={y_test[i]:.2f}, Err={errors[i]:+.2f}")

# Optionally save models / scalers
import joblib
joblib.dump(scaler_X, "scaler_X.joblib")
joblib.dump(scaler_y, "scaler_y.joblib")
lgb_model.booster_.save_model("lgb_model.txt")
model.save("hetero_nn.keras")

print("\nAll done. Models + scalers saved to disk.")



import matplotlib.pyplot as plt

errors = ensemble_pred - y_test
abs_errors = np.abs(errors)

# Precompute stats
mean_err = np.mean(errors)
median_err = np.median(errors)
mae = np.mean(abs_errors)
std_err = np.std(errors)
min_err = np.min(errors)
max_err = np.max(errors)
q25 = np.percentile(abs_errors, 25)
q50 = np.percentile(abs_errors, 50)
q75 = np.percentile(abs_errors, 75)
q90 = np.percentile(abs_errors, 90)
q95 = np.percentile(abs_errors, 95)

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# 1) Histogram of errors
axes[0, 0].hist(errors, bins=20, color='skyblue', edgecolor='black')
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=1)
axes[0, 0].axvline(mean_err, color='green', linestyle='-', linewidth=2, label=f"Mean: {mean_err:.2f}")
axes[0, 0].axvline(median_err, color='orange', linestyle='-', linewidth=2, label=f"Median: {median_err:.2f}")
axes[0, 0].set_title("Error Distribution (Predicted - Actual)")
axes[0, 0].set_xlabel("Error (°C)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].legend()

# Annotate text inside histogram
axes[0, 0].text(0.95, 0.95,
                f"Mean: {mean_err:.2f}\nMedian: {median_err:.2f}\nStd: {std_err:.2f}",
                transform=axes[0, 0].transAxes,
                ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

# 2) Boxplot of errors
box = axes[0, 1].boxplot(errors, vert=False, patch_artist=True,
                         boxprops=dict(facecolor='lightgreen'))
axes[0, 1].set_title("Boxplot of Prediction Errors")
axes[0, 1].set_xlabel("Error (°C)")
axes[0, 1].text(0.95, 0.95,
                f"Min: {min_err:.2f}\nQ1: {np.percentile(errors, 25):.2f}\nMedian: {median_err:.2f}\nQ3: {np.percentile(errors, 75):.2f}\nMax: {max_err:.2f}",
                transform=axes[0, 1].transAxes,
                ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

# 3) Scatter plot: predicted vs actual
axes[1, 0].scatter(y_test, ensemble_pred, alpha=0.7)
axes[1, 0].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--', lw=2)
axes[1, 0].set_title("Predicted vs Actual High Temperatures")
axes[1, 0].set_xlabel("Actual (°C)")
axes[1, 0].set_ylabel("Predicted (°C)")
axes[1, 0].text(0.05, 0.95,
                f"MAE: {mae:.2f}\nMSE: {np.mean(errors**2):.2f}\nR² ~ {1 - np.sum(errors**2)/np.sum((y_test - np.mean(y_test))**2):.2f}",
                transform=axes[1, 0].transAxes,
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

# 4) Error over time
axes[1, 1].plot(meta_dates[test_idx], errors, marker='o', linestyle='-', alpha=0.7)
axes[1, 1].axhline(0, color='black', linewidth=1, linestyle='--')
axes[1, 1].set_title("Error Over Time")
axes[1, 1].set_xlabel("Date")
axes[1, 1].set_ylabel("Error (°C)")
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].text(0.05, 0.95,
                f"Mean Err: {mean_err:.2f}\nStd Err: {std_err:.2f}",
                transform=axes[1, 1].transAxes,
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

# 5) Cumulative absolute error
axes[2, 0].plot(np.cumsum(abs_errors), color='purple')
axes[2, 0].set_title("Cumulative Absolute Error")
axes[2, 0].set_xlabel("Test Sample Index")
axes[2, 0].set_ylabel("Cumulative |Error| (°C)")
axes[2, 0].text(0.95, 0.95,
                f"Final Total: {np.cumsum(abs_errors)[-1]:.2f}",
                transform=axes[2, 0].transAxes,
                ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

# 6) Error quantiles
quantiles = np.linspace(0, 100, 21)
quantile_vals = np.percentile(abs_errors, quantiles)
axes[2, 1].plot(quantiles, quantile_vals, marker='o')
axes[2, 1].set_title("Error Percentiles")
axes[2, 1].set_xlabel("Percentile")
axes[2, 1].set_ylabel("|Error| (°C)")
axes[2, 1].grid(True, linestyle='--', alpha=0.5)
axes[2, 1].text(0.95, 0.95,
                f"25%: {q25:.2f}\n50%: {q50:.2f}\n75%: {q75:.2f}\n90%: {q90:.2f}\n95%: {q95:.2f}",
                transform=axes[2, 1].transAxes,
                ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

plt.tight_layout()
plt.show()

get rid of the comments
ChatGPT said:
Here’s your weather_pipeline_improved.py cleaned up with all comments removed:

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import joblib

DATA_PATH = "data.csv"
DATE_COL = "DATE"
FEATURE_COLS = ["TMP","DEWP","SLP","WDIR","WSPD","PRCP"]
INPUT_HOURS = 12
SPLIT_DATE = None
LGB_PARAMS = {"n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 31, "verbose": -1}
TF_LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 80
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def assert_no_nans(arr, name="array"):
    if np.isnan(arr).any():
        raise ValueError(f"NaN detected in {name} (shape={arr.shape})")
    if np.isinf(arr).any():
        raise ValueError(f"Inf detected in {name} (shape={arr.shape})")

def safe_inverse_transform(scaler, arr):
    return scaler.inverse_transform(np.asarray(arr).reshape(-1,1)).reshape(-1)

df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)
df.columns = [c.strip() for c in df.columns]

if DATE_COL not in df.columns:
    raise ValueError(f"{DATE_COL} not present in CSV")

df["date"] = df[DATE_COL].dt.date
df["hour"] = df[DATE_COL].dt.hour
df["doy"] = df[DATE_COL].dt.dayofyear
df["doy_sin"] = np.sin(2*np.pi * df["doy"] / 365.25)
df["doy_cos"] = np.cos(2*np.pi * df["doy"] / 365.25)
df["hour_sin"] = np.sin(2*np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2*np.pi * df["hour"] / 24.0)

if {"TMP", "DEWP"}.issubset(df.columns):
    a = 17.625
    b = 243.04
    alpha_dewp = (a * df["DEWP"]) / (b + df["DEWP"])
    alpha_tmp  = (a * df["TMP"]) / (b + df["TMP"])
    with np.errstate(all='ignore'):
        rh = 100.0 * (np.exp(alpha_dewp) / np.exp(alpha_tmp))
    df["RH"] = rh.clip(0, 100).fillna(np.nan)

df = df.sort_values(DATE_COL)
df["TMP_roll24_mean"] = df.groupby("date")["TMP"].transform(lambda x: x.rolling(window=min(24, max(1, len(x))), min_periods=1).mean())
df["TMP_rolling_24h"] = df["TMP"].rolling(window=24, min_periods=1).mean()

X_list, y_list, meta_dates = [], [], []

for col in FEATURE_COLS + ["RH", "TMP_rolling_24h", "TMP_roll24_mean", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]:
    if col in df.columns:
        df[col] = df.groupby("date")[col].transform(lambda s: s.interpolate(limit_direction='both'))
for col in FEATURE_COLS + ["RH", "TMP_rolling_24h", "TMP_roll24_mean", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]:
    if col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median(skipna=True)
            df[col] = df[col].fillna(median_val)

for date, g in df.groupby("date"):
    g = g.sort_values("hour")
    if g.shape[0] < INPUT_HOURS:
        continue
    seq = g[g["hour"] < INPUT_HOURS]
    if seq.shape[0] != INPUT_HOURS:
        continue
    seq_feats = seq[FEATURE_COLS].values
    prev_day = date - timedelta(days=1)
    prev_day_rows = df[df["date"] == prev_day]
    prev_max = prev_day_rows["TMP"].max() if not prev_day_rows.empty else np.nan
    prev_min = prev_day_rows["TMP"].min() if not prev_day_rows.empty else np.nan
    prev_mean = prev_day_rows["TMP"].mean() if not prev_day_rows.empty else np.nan
    if np.isnan(prev_max):
        prev_max = df.loc[df["date"] < date, "TMP"].tail(7).max() if not df.loc[df["date"] < date, "TMP"].empty else np.nan
    if np.isnan(prev_min):
        prev_min = df.loc[df["date"] < date, "TMP"].tail(7).min() if not df.loc[df["date"] < date, "TMP"].empty else np.nan
    if np.isnan(prev_mean):
        prev_mean = df.loc[df["date"] < date, "TMP"].tail(7).mean() if not df.loc[df["date"] < date, "TMP"].empty else np.nan
    prev_max = prev_max if not np.isnan(prev_max) else df["TMP"].median()
    prev_min = prev_min if not np.isnan(prev_min) else df["TMP"].median()
    prev_mean = prev_mean if not np.isnan(prev_mean) else df["TMP"].median()

    x_hours = np.arange(INPUT_HOURS)
    y_tmp = seq["TMP"].values
    A = np.vstack([x_hours, np.ones_like(x_hours)]).T
    try:
        slope, intercept = np.linalg.lstsq(A, y_tmp, rcond=None)[0]
    except Exception:
        slope = 0.0

    flat_seq = seq_feats.flatten()
    static_feats = np.array([
        prev_max, prev_min, prev_mean,
        seq["TMP_rolling_24h"].mean(),
        seq["TMP_roll24_mean"].mean(),
        slope,
        seq["hour_sin"].mean(), seq["hour_cos"].mean(),
        seq["doy_sin"].mean(), seq["doy_cos"].mean()
    ], dtype=float)

    X_list.append(np.concatenate([flat_seq, static_feats], axis=0))
    y_list.append(g["TMP"].max())
    meta_dates.append(date)

X = np.array(X_list)
y = np.array(y_list)
meta_dates = np.array(meta_dates)

print("Built dataset: X.shape =", X.shape, "y.shape =", y.shape)
assert_no_nans(X, "X after feature build")
assert_no_nans(y, "y after feature build")

if SPLIT_DATE is None:
    n = len(meta_dates)
    train_end_idx = int(n * 0.8)
    train_idx = np.arange(0, train_end_idx)
    test_idx = np.arange(train_end_idx, n)
else:
    split_date_obj = pd.to_datetime(SPLIT_DATE).date()
    train_idx = np.where(meta_dates < split_date_obj)[0]
    test_idx = np.where(meta_dates >= split_date_obj)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Bad SPLIT_DATE results in empty train or test")

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).reshape(-1)
y_test_scaled = scaler_y.transform(y_test.reshape(-1,1)).reshape(-1)

assert_no_nans(X_train_scaled, "X_train_scaled")
assert_no_nans(X_test_scaled, "X_test_scaled")
assert_no_nans(y_train_scaled, "y_train_scaled")
assert_no_nans(y_test_scaled, "y_test_scaled")

print("\nTraining LightGBM baseline (point estimator)...")
lgb_model = lgb.LGBMRegressor(**LGB_PARAMS, random_state=RANDOM_STATE)
lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(10)])

lgb_pred = lgb_model.predict(X_test_scaled)
lgb_mse = mean_squared_error(y_test, lgb_pred)
lgb_mae = mean_absolute_error(y_test, lgb_pred)
print(f"LightGBM - MSE: {lgb_mse:.3f}, MAE: {lgb_mae:.3f}")

print("\nTraining LightGBM quantile models (0.1, 0.5, 0.9)...")
quantiles = [0.1, 0.5, 0.9]
lgb_quant_preds = {}
for q in quantiles:
    params = LGB_PARAMS.copy()
    params["objective"] = "quantile"
    params["alpha"] = q
    model_q = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE)
    model_q.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(10)])
    lgb_quant_preds[q] = model_q.predict(X_test_scaled)

print("\nBuilding heteroscedastic Keras model...")
n_input = X_train_scaled.shape[1]
tf.keras.backend.set_floatx('float32')

def nll_heteroscedastic(y_true, y_pred):
    mu = y_pred[:, 0]
    log_sigma = y_pred[:, 1]
    sigma = tf.exp(log_sigma) + 1e-6
    return tf.reduce_mean(0.5 * tf.math.log(2.0 * np.pi) + log_sigma + 0.5 * ((y_true - mu)**2) / (sigma**2))

inp = Input(shape=(n_input,), dtype="float32")
x = layers.Dense(256, activation="relu")(inp)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
mu = layers.Dense(1, activation="linear")(x)
log_sigma = layers.Dense(1, activation="linear")(x)
out = layers.Concatenate()([mu, log_sigma])
model = models.Model(inputs=inp, outputs=out)

opt = tf.keras.optimizers.Adam(learning_rate=TF_LEARNING_RATE, clipnorm=1.0)
model.compile(optimizer=opt, loss=nll_heteroscedastic, metrics=[])
model.summary()

callbacks = [
    TerminateOnNaN(),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train_scaled.astype(np.float32),
    y_train_scaled.astype(np.float32),
    validation_data=(X_test_scaled.astype(np.float32), y_test_scaled.astype(np.float32)),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)

y_pred_het = model.predict(X_test_scaled.astype(np.float32))
mu_scaled = y_pred_het[:, 0]
log_sigma = y_pred_het[:, 1]
sigma_scaled = np.exp(log_sigma)
mu_pred = safe_inverse_transform(scaler_y, mu_scaled)
sigma_pred = sigma_scaled * scaler_y.scale_[0]

assert_no_nans(mu_pred, "mu_pred (keras)")
assert_no_nans(sigma_pred, "sigma_pred (keras)")

mse_het = mean_squared_error(y_test, mu_pred)
mae_het = mean_absolute_error(y_test, mu_pred)
print(f"Heteroscedastic NN - MSE: {mse_het:.3f}, MAE: {mae_het:.3f}")

ensemble_pred = 0.5 * lgb_pred + 0.5 * mu_pred
mse_ens = mean_squared_error(y_test, ensemble_pred)
mae_ens = mean_absolute_error(y_test, ensemble_pred)
print(f"\nEnsemble (50/50 LightGBM + NN) - MSE: {mse_ens:.3f}, MAE: {mae_ens:.3f}")

q_preds = {
    0.1: lgb_quant_preds[0.1],
    0.5: 0.5 * lgb_quant_preds[0.5] + 0.5 * mu_pred,
    0.9: lgb_quant_preds[0.9]
}

errors = ensemble_pred - y_test
print("\nSUMMARY:")
print(f"  LightGBM MAE: {lgb_mae:.3f}")
print(f"  Heteroscedastic NN MAE: {mae_het:.3f}")
print(f"  Ensemble MAE: {mae_ens:.3f}")
print(f"  Ensemble Bias (mean error): {np.mean(errors):.3f}, Std error: {np.std(errors):.3f}")

abs_err = np.abs(errors)
p50, p75, p90, p95 = np.percentile(abs_err, [50, 75, 90, 95])
print(f"  Ensemble absolute error percentiles: 50% {p50:.3f}, 75% {p75:.3f}, 90% {p90:.3f}, 95% {p95:.3f}")

idx_sorted = np.argsort(np.abs(errors))
print("\nBest predictions (top 5):")
for i in idx_sorted[:5]:
    print(f"  Pred={ensemble_pred[i]:.2f}, True={y_test[i]:.2f}, Err={errors[i]:+.2f}")

print("\nWorst predictions (top 5):")
for i in idx_sorted[-5:]:
    print(f"  Pred={ensemble_pred[i]:.2f}, True={y_test[i]:.2f}, Err={errors[i]:+.2f}")

joblib.dump(scaler_X, "scaler_X.joblib")
joblib.dump(scaler_y, "scaler_y.joblib")
lgb_model.booster_.save_model("lgb_model.txt")
model.save("hetero_nn.keras")

errors = ensemble_pred - y_test
abs_errors = np.abs(errors)
mean_err = np.mean(errors)
median_err = np.median(errors)
mae = np.mean(abs_errors)
std_err = np.std(errors)
min_err = np.min(errors)
max_err = np.max(errors)
q25 = np.percentile(abs_errors, 25)
q50 = np.percentile(abs_errors, 50)
q75 = np.percentile(abs_errors, 75)
q90 = np.percentile(abs_errors, 90)
q95 = np.percentile(abs_errors, 95)

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes[0, 0].hist(errors, bins=20, color='skyblue', edgecolor='black')
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=1)
axes[0, 0].axvline(mean_err, color='green', linestyle='-', linewidth=2, label=f"Mean: {mean_err:.2f}")
axes[0, 0].axvline(median_err, color='orange', linestyle='-', linewidth=2, label=f"Median: {median_err:.2f}")
axes[0, 0].set_title("Error Distribution (Predicted - Actual)")
axes[0, 0].set_xlabel("Error (°C)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].legend()
axes[0, 0].text(0.95, 0.95, f"Mean: {mean_err:.2f}\nMedian: {median_err:.2f}\nStd: {std_err:.2f}", transform=axes[0, 0].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

axes[0, 1].boxplot(errors, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
axes[0, 1].set_title("Boxplot of Prediction Errors")
axes[0, 1].set_xlabel("Error (°C)")
axes[0, 1].text(0.95, 0.95, f"Min: {min_err:.2f}\nQ1: {np.percentile(errors, 25):.2f}\nMedian: {median_err:.2f}\nQ3: {np.percentile(errors, 75):.2f}\nMax: {max_err:.2f}", transform=axes[0, 1].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

axes[1, 0].scatter(y_test, ensemble_pred, alpha=0.7)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_title("Predicted vs Actual High Temperatures")
axes[1, 0].set_xlabel("Actual (°C)")
axes[1, 0].set_ylabel("Predicted (°C)")
axes[1, 0].text(0.05, 0.95, f"MAE: {mae:.2f}\nMSE: {np.mean(errors**2):.2f}\nR² ~ {1 - np.sum(errors**2)/np.sum((y_test - np.mean(y_test))**2):.2f}", transform=axes[1, 0].transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='


Continue generating


No file chosenNo file chosen
ChatGPT can make mistakes. Check important info.
ChatGPT says: Here’s your `weather_pipeline_improved.py` cleaned up with all comments removed: ```python import os import warnings warnings.filterwarnings("ignore") import numpy as np import pandas as pd from sklearn.preprocessing import StandardScaler from sklearn.metrics import mean_squared_error, mean_absolute_error from sklearn.model_selection import train_test_split import lightgbm as lgb import tensorflow as tf from tensorflow.keras import layers, models from tensorflow.keras.layers import Input from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN from datetime import datetime, timedelta import matplotlib.pyplot as plt import joblib DATA_PATH = "data.csv" DATE_COL = "DATE" FEATURE_COLS = ["TMP","DEWP","SLP","WDIR","WSPD","PRCP"] INPUT_HOURS = 12 SPLIT_DATE = None LGB_PARAMS = {"n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 31, "verbose": -1} TF_LEARNING_RATE = 1e-3 BATCH_SIZE = 16 EPOCHS = 80 RANDOM_STATE = 42 np.random.seed(RANDOM_STATE) tf.random.set_seed(RANDOM_STATE) def assert_no_nans(arr, name="array"): if np.isnan(arr).any(): raise ValueError(f"NaN detected in {name} (shape={arr.shape})") if np.isinf(arr).any(): raise ValueError(f"Inf detected in {name} (shape={arr.shape})") def safe_inverse_transform(scaler, arr): return scaler.inverse_transform(np.asarray(arr).reshape(-1,1)).reshape(-1) df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL]) df = df.sort_values(DATE_COL).reset_index(drop=True) df.columns = [c.strip() for c in df.columns] if DATE_COL not in df.columns: raise ValueError(f"{DATE_COL} not present in CSV") df["date"] = df[DATE_COL].dt.date df["hour"] = df[DATE_COL].dt.hour df["doy"] = df[DATE_COL].dt.dayofyear df["doy_sin"] = np.sin(2*np.pi * df["doy"] / 365.25) df["doy_cos"] = np.cos(2*np.pi * df["doy"] / 365.25) df["hour_sin"] = np.sin(2*np.pi * df["hour"] / 24.0) df["hour_cos"] = np.cos(2*np.pi * df["hour"] / 24.0) if {"TMP", "DEWP"}.issubset(df.columns): a = 17.625 b = 243.04 alpha_dewp = (a * df["DEWP"]) / (b + df["DEWP"]) alpha_tmp = (a * df["TMP"]) / (b + df["TMP"]) with np.errstate(all='ignore'): rh = 100.0 * (np.exp(alpha_dewp) / np.exp(alpha_tmp)) df["RH"] = rh.clip(0, 100).fillna(np.nan) df = df.sort_values(DATE_COL) df["TMP_roll24_mean"] = df.groupby("date")["TMP"].transform(lambda x: x.rolling(window=min(24, max(1, len(x))), min_periods=1).mean()) df["TMP_rolling_24h"] = df["TMP"].rolling(window=24, min_periods=1).mean() X_list, y_list, meta_dates = [], [], [] for col in FEATURE_COLS + ["RH", "TMP_rolling_24h", "TMP_roll24_mean", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]: if col in df.columns: df[col] = df.groupby("date")[col].transform(lambda s: s.interpolate(limit_direction='both')) for col in FEATURE_COLS + ["RH", "TMP_rolling_24h", "TMP_roll24_mean", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]: if col in df.columns: if df[col].isna().any(): median_val = df[col].median(skipna=True) df[col] = df[col].fillna(median_val) for date, g in df.groupby("date"): g = g.sort_values("hour") if g.shape[0] < INPUT_HOURS: continue seq = g[g["hour"] < INPUT_HOURS] if seq.shape[0] != INPUT_HOURS: continue seq_feats = seq[FEATURE_COLS].values prev_day = date - timedelta(days=1) prev_day_rows = df[df["date"] == prev_day] prev_max = prev_day_rows["TMP"].max() if not prev_day_rows.empty else np.nan prev_min = prev_day_rows["TMP"].min() if not prev_day_rows.empty else np.nan prev_mean = prev_day_rows["TMP"].mean() if not prev_day_rows.empty else np.nan if np.isnan(prev_max): prev_max = df.loc[df["date"] < date, "TMP"].tail(7).max() if not df.loc[df["date"] < date, "TMP"].empty else np.nan if np.isnan(prev_min): prev_min = df.loc[df["date"] < date, "TMP"].tail(7).min() if not df.loc[df["date"] < date, "TMP"].empty else np.nan if np.isnan(prev_mean): prev_mean = df.loc[df["date"] < date, "TMP"].tail(7).mean() if not df.loc[df["date"] < date, "TMP"].empty else np.nan prev_max = prev_max if not np.isnan(prev_max) else df["TMP"].median() prev_min = prev_min if not np.isnan(prev_min) else df["TMP"].median() prev_mean = prev_mean if not np.isnan(prev_mean) else df["TMP"].median() x_hours = np.arange(INPUT_HOURS) y_tmp = seq["TMP"].values A = np.vstack([x_hours, np.ones_like(x_hours)]).T try: slope, intercept = np.linalg.lstsq(A, y_tmp, rcond=None)[0] except Exception: slope = 0.0 flat_seq = seq_feats.flatten() static_feats = np.array([ prev_max, prev_min, prev_mean, seq["TMP_rolling_24h"].mean(), seq["TMP_roll24_mean"].mean(), slope, seq["hour_sin"].mean(), seq["hour_cos"].mean(), seq["doy_sin"].mean(), seq["doy_cos"].mean() ], dtype=float) X_list.append(np.concatenate([flat_seq, static_feats], axis=0)) y_list.append(g["TMP"].max()) meta_dates.append(date) X = np.array(X_list) y = np.array(y_list) meta_dates = np.array(meta_dates) print("Built dataset: X.shape =", X.shape, "y.shape =", y.shape) assert_no_nans(X, "X after feature build") assert_no_nans(y, "y after feature build") if SPLIT_DATE is None: n = len(meta_dates) train_end_idx = int(n * 0.8) train_idx = np.arange(0, train_end_idx) test_idx = np.arange(train_end_idx, n) else: split_date_obj = pd.to_datetime(SPLIT_DATE).date() train_idx = np.where(meta_dates < split_date_obj)[0] test_idx = np.where(meta_dates >= split_date_obj)[0] if len(train_idx) == 0 or len(test_idx) == 0: raise ValueError("Bad SPLIT_DATE results in empty train or test") X_train = X[train_idx] X_test = X[test_idx] y_train = y[train_idx] y_test = y[test_idx] print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}") scaler_X = StandardScaler() scaler_y = StandardScaler() X_train_scaled = scaler_X.fit_transform(X_train) X_test_scaled = scaler_X.transform(X_test) y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).reshape(-1) y_test_scaled = scaler_y.transform(y_test.reshape(-1,1)).reshape(-1) assert_no_nans(X_train_scaled, "X_train_scaled") assert_no_nans(X_test_scaled, "X_test_scaled") assert_no_nans(y_train_scaled, "y_train_scaled") assert_no_nans(y_test_scaled, "y_test_scaled") print("\nTraining LightGBM baseline (point estimator)...") lgb_model = lgb.LGBMRegressor(**LGB_PARAMS, random_state=RANDOM_STATE) lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(10)]) lgb_pred = lgb_model.predict(X_test_scaled) lgb_mse = mean_squared_error(y_test, lgb_pred) lgb_mae = mean_absolute_error(y_test, lgb_pred) print(f"LightGBM - MSE: {lgb_mse:.3f}, MAE: {lgb_mae:.3f}") print("\nTraining LightGBM quantile models (0.1, 0.5, 0.9)...") quantiles = [0.1, 0.5, 0.9] lgb_quant_preds = {} for q in quantiles: params = LGB_PARAMS.copy() params["objective"] = "quantile" params["alpha"] = q model_q = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE) model_q.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(10)]) lgb_quant_preds[q] = model_q.predict(X_test_scaled) print("\nBuilding heteroscedastic Keras model...") n_input = X_train_scaled.shape[1] tf.keras.backend.set_floatx('float32') def nll_heteroscedastic(y_true, y_pred): mu = y_pred[:, 0] log_sigma = y_pred[:, 1] sigma = tf.exp(log_sigma) + 1e-6 return tf.reduce_mean(0.5 * tf.math.log(2.0 * np.pi) + log_sigma + 0.5 * ((y_true - mu)**2) / (sigma**2)) inp = Input(shape=(n_input,), dtype="float32") x = layers.Dense(256, activation="relu")(inp) x = layers.BatchNormalization()(x) x = layers.Dropout(0.3)(x) x = layers.Dense(128, activation="relu")(x) x = layers.BatchNormalization()(x) x = layers.Dropout(0.2)(x) mu = layers.Dense(1, activation="linear")(x) log_sigma = layers.Dense(1, activation="linear")(x) out = layers.Concatenate()([mu, log_sigma]) model = models.Model(inputs=inp, outputs=out) opt = tf.keras.optimizers.Adam(learning_rate=TF_LEARNING_RATE, clipnorm=1.0) model.compile(optimizer=opt, loss=nll_heteroscedastic, metrics=[]) model.summary() callbacks = [ TerminateOnNaN(), EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1), ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-6, verbose=1) ] history = model.fit( X_train_scaled.astype(np.float32), y_train_scaled.astype(np.float32), validation_data=(X_test_scaled.astype(np.float32), y_test_scaled.astype(np.float32)), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2 ) y_pred_het = model.predict(X_test_scaled.astype(np.float32)) mu_scaled = y_pred_het[:, 0] log_sigma = y_pred_het[:, 1] sigma_scaled = np.exp(log_sigma) mu_pred = safe_inverse_transform(scaler_y, mu_scaled) sigma_pred = sigma_scaled * scaler_y.scale_[0] assert_no_nans(mu_pred, "mu_pred (keras)") assert_no_nans(sigma_pred, "sigma_pred (keras)") mse_het = mean_squared_error(y_test, mu_pred) mae_het = mean_absolute_error(y_test, mu_pred) print(f"Heteroscedastic NN - MSE: {mse_het:.3f}, MAE: {mae_het:.3f}") ensemble_pred = 0.5 * lgb_pred + 0.5 * mu_pred mse_ens = mean_squared_error(y_test, ensemble_pred) mae_ens = mean_absolute_error(y_test, ensemble_pred) print(f"\nEnsemble (50/50 LightGBM + NN) - MSE: {mse_ens:.3f}, MAE: {mae_ens:.3f}") q_preds = { 0.1: lgb_quant_preds[0.1], 0.5: 0.5 * lgb_quant_preds[0.5] + 0.5 * mu_pred, 0.9: lgb_quant_preds[0.9] } errors = ensemble_pred - y_test print("\nSUMMARY:") print(f" LightGBM MAE: {lgb_mae:.3f}") print(f" Heteroscedastic NN MAE: {mae_het:.3f}") print(f" Ensemble MAE: {mae_ens:.3f}") print(f" Ensemble Bias (mean error): {np.mean(errors):.3f}, Std error: {np.std(errors):.3f}") abs_err = np.abs(errors) p50, p75, p90, p95 = np.percentile(abs_err, [50, 75, 90, 95]) print(f" Ensemble absolute error percentiles: 50% {p50:.3f}, 75% {p75:.3f}, 90% {p90:.3f}, 95% {p95:.3f}") idx_sorted = np.argsort(np.abs(errors)) print("\nBest predictions (top 5):") for i in idx_sorted[:5]: print(f" Pred={ensemble_pred[i]:.2f}, True={y_test[i]:.2f}, Err={errors[i]:+.2f}") print("\nWorst predictions (top 5):") for i in idx_sorted[-5:]: print(f" Pred={ensemble_pred[i]:.2f}, True={y_test[i]:.2f}, Err={errors[i]:+.2f}") joblib.dump(scaler_X, "scaler_X.joblib") joblib.dump(scaler_y, "scaler_y.joblib") lgb_model.booster_.save_model("lgb_model.txt") model.save("hetero_nn.keras") errors = ensemble_pred - y_test abs_errors = np.abs(errors) mean_err = np.mean(errors) median_err = np.median(errors) mae = np.mean(abs_errors) std_err = np.std(errors) min_err = np.min(errors) max_err = np.max(errors) q25 = np.percentile(abs_errors, 25) q50 = np.percentile(abs_errors, 50) q75 = np.percentile(abs_errors, 75) q90 = np.percentile(abs_errors, 90) q95 = np.percentile(abs_errors, 95) fig, axes = plt.subplots(3, 2, figsize=(15, 12)) axes[0, 0].hist(errors, bins=20, color='skyblue', edgecolor='black') axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=1) axes[0, 0].axvline(mean_err, color='green', linestyle='-', linewidth=2, label=f"Mean: {mean_err:.2f}") axes[0, 0].axvline(median_err, color='orange', linestyle='-', linewidth=2, label=f"Median: {median_err:.2f}") axes[0, 0].set_title("Error Distribution (Predicted - Actual)") axes[0, 0].set_xlabel("Error (°C)") axes[0, 0].set_ylabel("Frequency") axes[0, 0].legend() axes[0, 0].text(0.95, 0.95, f"Mean: {mean_err:.2f}\nMedian: {median_err:.2f}\nStd: {std_err:.2f}", transform=axes[0, 0].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')) axes[0, 1].boxplot(errors, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen')) axes[0, 1].set_title("Boxplot of Prediction Errors") axes[0, 1].set_xlabel("Error (°C)") axes[0, 1].text(0.95, 0.95, f"Min: {min_err:.2f}\nQ1: {np.percentile(errors, 25):.2f}\nMedian: {median_err:.2f}\nQ3: {np.percentile(errors, 75):.2f}\nMax: {max_err:.2f}", transform=axes[0, 1].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')) axes[1, 0].scatter(y_test, ensemble_pred, alpha=0.7) axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) axes[1, 0].set_title("Predicted vs Actual High Temperatures") axes[1, 0].set_xlabel("Actual (°C)") axes[1, 0].set_ylabel("Predicted (°C)") axes[1, 0].text(0.05, 0.95, f"MAE: {mae:.2f}\nMSE: {np.mean(errors**2):.2f}\nR² ~ {1 - np.sum(errors**2)/np.sum((y_test - np.mean(y_test))**2):.2f}", transform=axes[1, 0].transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='