# ARMA Modelling of the US Business Cycle

## Overview

This project analyzes the business cycle dynamics of the United States using annual GDP per capita data from 1960 to 2022. The study applies the Hodrick-Prescott (HP) filter to extract cyclical fluctuations from the long-run growth trend and uses ARMA modeling to characterize the resulting business cycle component.

A comprehensive grid search is performed over ARMA(p,q) models with p,q ∈ {0,1,2,3,4}, and the optimal specification is selected using the Akaike Information Criterion (AIC). Additionally, an ARIMA-based approach is implemented on the raw GDP series to compare trend-cycle decomposition with direct time-series modeling.

---

## Project Objectives

* Extract U.S. GDP per capita data (1960–2022).
* Transform the series using natural logarithms.
* Apply the Hodrick-Prescott filter (λ = 6.25) to separate trend and cyclical components.
* Analyze business cycle fluctuations.
* Examine autocorrelation patterns using ACF and PACF.
* Estimate 25 ARMA(p,q) models.
* Select the optimal model using Akaike Information Criterion (AIC).
* Conduct stationarity testing using the Augmented Dickey-Fuller (ADF) test.
* Fit ARIMA models directly on the original GDP series.
* Compare ARMA and ARIMA frameworks for macroeconomic time-series analysis.

---

## Dataset

**Source:** World Bank Open Data

**Indicator:** GDP per Capita (Constant 2015 US$)

**Country:** United States

**Period:** 1960–2022

**Observations:** 63 Annual Data Points

---

## Methodology

### 1. Data Preprocessing

* Imported GDP per capita data from Excel.
* Selected observations for the United States.
* Extracted annual data from 1960–2022.
* Applied logarithmic transformation:

yₜ = ln(GDP per capita)

### 2. Hodrick-Prescott Filter

The HP filter decomposes the series into:

yₜ = τₜ + εₜ

where:

* τₜ = Long-run trend
* εₜ = Business cycle component

For annual data:

λ = 6.25

The residual series εₜ is used for ARMA modeling.

### 3. Business Cycle Analysis

The extracted cyclical component was analyzed to identify:

* Economic expansions
* Economic contractions
* Recession periods
* Deviations from long-run growth

### 4. Autocorrelation Diagnostics

Generated:

* Autocorrelation Function (ACF)
* Partial Autocorrelation Function (PACF)

These diagnostics provide insights into the potential AR and MA orders.

### 5. ARMA Model Selection

Estimated all combinations:

ARMA(p,q)

where:

p = 0,1,2,3,4

q = 0,1,2,3,4

Total Models Estimated:

25

AIC values were computed for every specification and stored in a 5×5 matrix.

### 6. ARIMA Modeling

To model the original GDP series directly:

* Performed Augmented Dickey-Fuller (ADF) Test.
* Checked stationarity.
* Applied first differencing.
* Estimated ARIMA(p,1,q) models.
* Selected the optimal model using AIC.

---

## Results

### HP Filter + ARMA Framework

* Business cycle component successfully extracted.
* Residual series exhibited weak persistence.
* Best model according to AIC:

ARMA(0,0)

AIC = -539.6

* Best non-trivial specification:

ARMA(2,1)

AIC = -376.8

### ARIMA Framework

ADF Test Results:

| Series           | ADF Statistic | p-value |
| ---------------- | ------------- | ------- |
| Log GDP          | -2.251        | 0.188   |
| First Difference | -6.651        | <0.001  |

Result:

* Log GDP is non-stationary.
* First differenced series is stationary.
* GDP follows an I(1) process.

Best Model:

ARIMA(0,1,0)

AIC = -295.1

Interpretation:

* U.S. GDP per capita behaves like a random walk with drift.
* Estimated annual growth rate ≈ 1.94%.

---

## Visualizations

The project generates:

* Log GDP per Capita Time Series
* HP Filter Residual Plot
* ACF Plot
* PACF Plot
* ARMA AIC Heatmap
* ARIMA AIC Heatmap

---

## Technologies Used

* Python 3.12
* Pandas
* NumPy
* Matplotlib
* Statsmodels
* OpenPyXL

---

## Repository Structure

```text
ARMA-US-Business-Cycle/
│
├── data/
│   └── GDP_Per_Capita.xlsx
│
├── figures/
│   ├── log_gdp.png
│   ├── hp_cycle.png
│   ├── acf_pacf.png
│   ├── arma_heatmap.png
│   └── arima_heatmap.png
│
├── report/
│   └── IIMA.pdf
│
└── ARMA_US.py
│
└── README.md
```

---

## Key Learnings

* Time Series Analysis
* Business Cycle Modeling
* Hodrick-Prescott Filtering
* ARMA Model Estimation
* ARIMA Modeling
* Stationarity Testing
* AIC-Based Model Selection
* Macroeconomic Data Analysis
* Econometrics with Python

---

## Author

### Shreya Sidabache

