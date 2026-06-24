# =============================================================================
# ARMA Modelling of US Business Cycle
# =============================================================================
# This script performs the following steps:
#   1. Load and plot log GDP per capita for USA (1960–2022)
#   2. Apply HP filter (lambda=6.25) and extract residual cycle component
#   3. Plot the cyclical residual series
#   4. Plot ACF and PACF of the residual
#   5. Fit ARMA(p,q) for p,q in {0,1,2,3,4} and build a 5x5 AIC matrix
#   6. Select the best model based on AIC
#   7. Discuss and carry out ARIMA modelling on the raw log-GDP series
# =============================================================================

# ── 0. Imports ────────────────────────────────────────────────────────────────
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")   # suppress convergence warnings for cleaner output

# ── 1. Load data and extract log GDP per capita for USA (1960–2022) ──────────

DATA_PATH = r"C:\Users\Lenovo\Desktop\Projects\IIMA_Assignment\API_NY.GDP.PCAP.KD_DS2_en_excel_v2_5607591.xls"

# The World Bank file has 3 header rows before the actual column names
df_raw = pd.read_excel(DATA_PATH, sheet_name="Data", engine="xlrd", header=3)

# Filter for the United States (Country Code = "USA")
usa_row = df_raw[df_raw["Country Code"] == "USA"]

# Year columns are stored as STRING column names '1960', '1961', ..., '2022'
years = [str(y) for y in range(1960, 2023)]  # 1960 to 2022 inclusive → 63 observations
gdp_series = usa_row[years].values.flatten().astype(float)

# Create a pandas Series with a DatetimeIndex (annual, year-end)
date_index = pd.date_range(start="1960", periods=len(years), freq="YE")
y = pd.Series(np.log(gdp_series), index=date_index, name="log_GDP_per_capita")

# ── Plot 1: Log GDP per capita ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y.index.year, y.values, color="steelblue", linewidth=1.8, label=r"$y_t = \ln(\text{GDP per capita})$")
ax.set_title("USA Log GDP per Capita (1960–2022)", fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel(r"$y_t$ (log scale, constant 2015 USD)", fontsize=12)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(r"C:\Users\Lenovo\Desktop\Projects\IIMA_Assignment\plot1_log_gdp.png", dpi=150)
plt.show()
print("Plot 1 saved: Log GDP per capita")

# ── 2. HP Filter (lambda = 6.25 for annual data) ─────────────────────────────
# hpfilter returns: (cycle, trend)
# The 'cycle' is the residual ε_t = y_t − τ_t  (business cycle component)
epsilon, trend = hpfilter(y.values, lamb=6.25)

epsilon_series = pd.Series(epsilon, index=date_index, name="HP_cycle")

print(f"\nHP Filter applied with λ = 6.25")
print(f"  Trend range  : {trend.min():.4f} – {trend.max():.4f}")
print(f"  Cycle std dev: {epsilon.std():.4f}")

# ── 3. Plot the cyclical residual ε_t ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(date_index.year, epsilon_series.values, color="crimson", linewidth=1.6,
        label=r"$\epsilon_t$ (HP cycle)")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.fill_between(date_index.year, epsilon_series.values, 0,
                where=(epsilon_series.values < 0), alpha=0.25, color="crimson", label="Recessions (approx.)")
ax.set_title("USA Business Cycle Component — HP Filter Residual", fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel(r"$\epsilon_t$ (deviation from trend)", fontsize=12)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(r"C:\Users\Lenovo\Desktop\Projects\IIMA_Assignment\plot2_hp_residual.png", dpi=150)
plt.show()
print("Plot 2 saved: HP cycle residual")

# ── 4. ACF and PACF of ε_t ───────────────────────────────────────────────────
# We use 20 lags (about 1/3 of 63 observations — a standard rule of thumb)
N_LAGS = 20

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

plot_acf(epsilon_series, lags=N_LAGS, ax=axes[0], alpha=0.05,
         title="ACF of HP-Filter Residual $\\epsilon_t$")
axes[0].set_xlabel("Lag (years)", fontsize=12)
axes[0].set_ylabel("Autocorrelation", fontsize=12)
axes[0].axhline(0, color="black", linewidth=0.6)

plot_pacf(epsilon_series, lags=N_LAGS, ax=axes[1], alpha=0.05,
          method="ywm",   # Yule-Walker (modified) — numerically stable for small samples
          title="PACF of HP-Filter Residual $\\epsilon_t$")
axes[1].set_xlabel("Lag (years)", fontsize=12)
axes[1].set_ylabel("Partial Autocorrelation", fontsize=12)
axes[1].axhline(0, color="black", linewidth=0.6)

plt.suptitle("Autocorrelation Diagnostics for $\\epsilon_t$", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(r"C:\Users\Lenovo\Desktop\Projects\IIMA_Assignment\plot3_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot 3 saved: ACF / PACF")

# ── 5. ARMA(p,q) grid search — AIC matrix ─────────────────────────────────────
# ARMA(p,q) is equivalent to ARIMA(p,0,q); we set d=0 because the HP-residual
# is already stationary by construction.

P_MAX = 4
Q_MAX = 4

aic_matrix = np.full((P_MAX + 1, Q_MAX + 1), np.nan)  # NaN = failed to converge

print("\nFitting ARMA(p,q) models on HP residual ε_t ...\n")
print(f"{'Model':<15} {'AIC':>12}")
print("-" * 28)

for p in range(P_MAX + 1):
    for q in range(Q_MAX + 1):
        if p == 0 and q == 0:
            # White noise: no AR or MA terms; AIC = n*log(σ²) + 2*1 (constant)
            sigma2 = np.var(epsilon_series.values, ddof=1)
            n = len(epsilon_series)
            aic_wn = n * np.log(sigma2) + 2 * 1
            aic_matrix[p, q] = round(aic_wn, 3)
            print(f"  ARMA(0,0)    :  {aic_wn:>10.3f}   [white noise baseline]")
            continue
        try:
            model = SARIMAX(
                epsilon_series,
                order=(p, 0, q),
                trend="c",                      # include intercept (mean)
                enforce_stationarity=False,     # allow exploration of unit-root region
                enforce_invertibility=False
            )
            result = model.fit(disp=False, maxiter=500)
            aic_matrix[p, q] = round(result.aic, 3)
            print(f"  ARMA({p},{q})      :  {result.aic:>10.3f}")
        except Exception as e:
            # Convergence failure: leave as NaN
            print(f"  ARMA({p},{q})      :  [convergence failed — {type(e).__name__}]")

# Display 5×5 AIC matrix as a labelled DataFrame
index_labels = [f"p={p}" for p in range(P_MAX + 1)]
col_labels   = [f"q={q}" for q in range(Q_MAX + 1)]
aic_df = pd.DataFrame(aic_matrix, index=index_labels, columns=col_labels)

print("\n── 5×5 AIC Matrix (rows = p, cols = q) ──")
print(aic_df.to_string())

# ── 6. Best model selection ────────────────────────────────────────────────────
best_idx = np.unravel_index(np.nanargmin(aic_matrix), aic_matrix.shape)
best_p, best_q = best_idx
best_aic = aic_matrix[best_p, best_q]

print(f"\n── Best Model (lowest AIC) ──")
print(f"  ARMA({best_p},{best_q})  →  AIC = {best_aic:.3f}")

# Re-fit the best model and print full summary
best_model = SARIMAX(
    epsilon_series,
    order=(best_p, 0, best_q),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False, maxiter=500)

print("\nBest-Model Summary:")
print(best_model.summary())

# ── Plot the AIC matrix as a heatmap ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
masked = np.where(np.isnan(aic_matrix), np.nanmax(aic_matrix) * 1.1, aic_matrix)
im = ax.imshow(masked, cmap="RdYlGn_r", aspect="auto")
plt.colorbar(im, ax=ax, label="AIC")
ax.set_xticks(range(P_MAX + 1))
ax.set_yticks(range(Q_MAX + 1))
ax.set_xticklabels([f"q={q}" for q in range(Q_MAX + 1)])
ax.set_yticklabels([f"p={p}" for p in range(P_MAX + 1)])
ax.set_xlabel("MA order q", fontsize=12)
ax.set_ylabel("AR order p", fontsize=12)
ax.set_title("AIC Heatmap for ARMA(p,q) Models", fontsize=13, fontweight="bold")

# Annotate each cell
for i in range(P_MAX + 1):
    for j in range(Q_MAX + 1):
        val = aic_matrix[i, j]
        txt = f"{val:.1f}" if not np.isnan(val) else "—"
        color = "white" if (i == best_p and j == best_q) else "black"
        weight = "bold" if (i == best_p and j == best_q) else "normal"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                color=color, fontweight=weight)

# Highlight best cell
rect = plt.Rectangle((best_q - 0.5, best_p - 0.5), 1, 1,
                      linewidth=2.5, edgecolor="blue", facecolor="none")
ax.add_patch(rect)

plt.tight_layout()
plt.savefig(r"C:\Users\Lenovo\Desktop\Projects\IIMA_Assignment\plot4_aic_heatmap.png", dpi=150)
plt.show()
print("Plot 4 saved: AIC heatmap")

# ── 7. ARIMA on the raw log-GDP series ────────────────────────────────────────
# Discussion:
# The raw log-GDP series {y_t} is clearly non-stationary (it has a persistent
# upward trend). ARIMA(p,d,q) with d≥1 is designed precisely for this case.
# Taking d=1 first-differences {y_t}, removing the stochastic trend.
# We do NOT need to HP-filter first — differencing is the ARIMA way to
# achieve stationarity.
#
# A unit-root test (ADF) helps decide d.

from statsmodels.tsa.stattools import adfuller

print("\n── Step 7: ARIMA on raw log-GDP ──")
adf_level = adfuller(y.values, autolag="AIC")
print(f"  ADF test on y_t (levels):")
print(f"    Test statistic : {adf_level[0]:.4f}")
print(f"    p-value        : {adf_level[1]:.4f}")
print(f"    → {'Non-stationary (fail to reject H₀)' if adf_level[1] > 0.05 else 'Stationary (reject H₀)'}")

dy = y.diff().dropna()
adf_diff = adfuller(dy.values, autolag="AIC")
print(f"\n  ADF test on Δy_t (first difference):")
print(f"    Test statistic : {adf_diff[0]:.4f}")
print(f"    p-value        : {adf_diff[1]:.4f}")
print(f"    → {'Non-stationary (fail to reject H₀)' if adf_diff[1] > 0.05 else 'Stationary (reject H₀)'}")

# Fit ARIMA(p, 1, q) grid — d=1 because y_t ~ I(1)
D = 1   # degree of differencing

aic_arima = np.full((P_MAX + 1, Q_MAX + 1), np.nan)

print(f"\nFitting ARIMA(p,{D},q) models on raw log-GDP y_t ...\n")
print(f"{'Model':<15} {'AIC':>12}")
print("-" * 28)

for p in range(P_MAX + 1):
    for q in range(Q_MAX + 1):
        try:
            model = SARIMAX(
                y,
                order=(p, D, q),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            result = model.fit(disp=False, maxiter=500)
            aic_arima[p, q] = round(result.aic, 3)
            print(f"  ARIMA({p},{D},{q})    :  {result.aic:>10.3f}")
        except Exception as e:
            print(f"  ARIMA({p},{D},{q})    :  [convergence failed — {type(e).__name__}]")

aic_arima_df = pd.DataFrame(aic_arima, index=index_labels, columns=col_labels)
print(f"\n── 5×5 AIC Matrix for ARIMA(p,{D},q) ──")
print(aic_arima_df.to_string())

best_arima_idx = np.unravel_index(np.nanargmin(aic_arima), aic_arima.shape)
bp, bq = best_arima_idx
print(f"\n── Best ARIMA Model (lowest AIC) ──")
print(f"  ARIMA({bp},{D},{bq})  →  AIC = {aic_arima[bp, bq]:.3f}")

best_arima_model = SARIMAX(
    y,
    order=(bp, D, bq),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False, maxiter=500)

print("\nBest ARIMA Model Summary:")
print(best_arima_model.summary())

# Heatmap for ARIMA AIC
fig, ax = plt.subplots(figsize=(7, 5))
masked2 = np.where(np.isnan(aic_arima), np.nanmax(aic_arima) * 1.1, aic_arima)
im2 = ax.imshow(masked2, cmap="RdYlGn_r", aspect="auto")
plt.colorbar(im2, ax=ax, label="AIC")
ax.set_xticks(range(P_MAX + 1))
ax.set_yticks(range(P_MAX + 1))
ax.set_xticklabels([f"q={q}" for q in range(Q_MAX + 1)])
ax.set_yticklabels([f"p={p}" for p in range(P_MAX + 1)])
ax.set_xlabel("MA order q", fontsize=12)
ax.set_ylabel("AR order p", fontsize=12)
ax.set_title(f"AIC Heatmap for ARIMA(p,{D},q) Models", fontsize=13, fontweight="bold")

for i in range(P_MAX + 1):
    for j in range(Q_MAX + 1):
        val = aic_arima[i, j]
        txt = f"{val:.1f}" if not np.isnan(val) else "—"
        color = "white" if (i == bp and j == bq) else "black"
        weight = "bold" if (i == bp and j == bq) else "normal"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                color=color, fontweight=weight)

rect2 = plt.Rectangle((bq - 0.5, bp - 0.5), 1, 1,
                       linewidth=2.5, edgecolor="blue", facecolor="none")
ax.add_patch(rect2)

plt.tight_layout()
plt.savefig(r"C:\Users\Lenovo\Desktop\Projects\IIMA_Assignment\plot5_arima_aic_heatmap.png", dpi=150)
plt.show()
print("Plot 5 saved: ARIMA AIC heatmap")

print("\n✅  All steps completed successfully.")