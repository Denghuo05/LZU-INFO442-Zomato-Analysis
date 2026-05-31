# Data Quality Summary — Zomato Restaurant Dataset

**Project:** What Makes a Restaurant Successful? — Data-Driven Analysis Based on Zomato Ratings  
**Module:** M2 & M3 — Data Acquisition & Preprocessing  
**Team:** T21 — Wenxuan Wu · Jiarui Zhai · Zilin Wang · Jiayang Li · Yuchen Zhang  
**Date:** 2026-05-31  
**Cleaned file:** `data/zomato_clean_all.csv`

---

## 1. Null Rates

### 1.1 Raw Dataset Null Rates

The table below shows the null count and null percentage for every column in the **raw** dataset (9,552 rows × 23 columns).

| Column | Null Count | Null Rate (%) |
|---|---|---|
| Unnamed: 0 | 0 | 0.000 |
| Restaurant ID | 0 | 0.000 |
| Restaurant Name | 0 | 0.000 |
| Country Name | 0 | 0.000 |
| Country Code | 0 | 0.000 |
| City | 0 | 0.000 |
| Address | 0 | 0.000 |
| Locality | 1 | 0.010 |
| Locality Verbose | 1 | 0.010 |
| Longitude | 1 | 0.010 |
| Latitude | 1 | 0.010 |
| Cuisines | 10 | 0.105 |
| Average Cost for two | 1 | 0.010 |
| Currency | 1 | 0.010 |
| Has Table booking | 1 + 1 anomaly | 0.021 |
| Has Online delivery | 1 | 0.010 |
| Is delivering now | 1 | 0.010 |
| Switch to order menu | 2 | 0.021 |
| Price range | 2 | 0.021 |
| Aggregate rating | 2 | 0.021 |
| Rating color | 2 | 0.021 |
| Rating text | 2 | 0.021 |
| Votes | 2 | 0.021 |

**Overall raw null rate:** < 0.11% across all columns. The dataset was already largely complete; no column exceeded 0.11% missing values.

### 1.2 Cleaned Dataset Null Rates

After preprocessing:

| Metric | Value |
|---|---|
| Total cells | 6,567 × 24 = **157,608** |
| Null cells | **0** |
| Overall null rate | **0.000%** |

All columns in `zomato_clean_all.csv` have a null rate of exactly **0%**. Complete data coverage is guaranteed for every analytical task.

---

## 2. Class Balance

### 2.1 Target Variable — `Aggregate rating`

The cleaned dataset filters out "Not rated" (0.0) entries via the IQR outlier step. The resulting distribution covers the range **[1.8, 4.9]**.

| Statistic | Value |
|---|---|
| Count | 6,567 |
| Mean | 3.397 |
| Std | 0.542 |
| Min | 1.800 |
| 25th percentile | 3.000 |
| Median | 3.400 |
| 75th percentile | 3.800 |
| Max | 4.900 |

The distribution is approximately unimodal and slightly left-skewed, centred around 3.4. No extreme class imbalance exists for regression tasks. For classification tasks (e.g., binning into Low / Medium / High), stratified sampling is recommended.

Zomato `Rating text` distribution in the cleaned dataset:

| Rating Text | Count | % |
|---|---|---|
| Average | ~2,800 | ~43% |
| Good | ~2,100 | ~32% |
| Very Good | ~900 | ~14% |
| Poor | ~500 | ~8% |
| Excellent | ~280 | ~4% |

*(Approximate; "Not rated" rows were removed by IQR filtering.)*

---

### 2.2 Binary Features

#### `Has Table booking`

| Value | Count | % |
|---|---|---|
| 0 (No) | 6,012 | 91.6% |
| 1 (Yes) | 555 | 8.5% |

**Assessment:** Heavily imbalanced (~11:1 ratio). Reflects the real-world scarcity of table-booking facilities in the dataset (dominated by Indian restaurants). The natural distribution is **retained** without resampling — it is a genuine business characteristic, not a data collection artefact. For classification models predicting table-booking availability, SMOTE or class-weight adjustment may be needed.

#### `Has Online delivery`

| Value | Count | % |
|---|---|---|
| 0 (No) | 4,413 | 67.2% |
| 1 (Yes) | 2,154 | 32.8% |

**Assessment:** Moderate imbalance (~2:1 ratio). Distribution retained as-is; no resampling applied.

#### `Is delivering now`

| Value | Count | % |
|---|---|---|
| 0 (No) | ~6,400+ | ~97%+ |
| 1 (Yes) | ~100 | ~1-2% |

**Assessment:** Severely imbalanced. This column reflects a snapshot of real-time delivery status, which is inherently near-zero during most data collection windows. Retained as-is for informational use; excluded from primary predictive features.

#### `Switch to order menu`

| Value | Count | % |
|---|---|---|
| 0 (No) | ~6,560+ | ~99.9% |
| 1 (Yes) | ~5 | ~0.1% |

**Assessment:** Near-constant (< 0.1% positive). Retained in schema for completeness but should be excluded from modelling due to near-zero variance.

---

### 2.3 Engineered Feature — `Price_Level`

| Tier | Condition | Count | % |
|---|---|---|---|
| Budget | cost < 600 | 4,333 | 66.0% |
| Mid-range | 600 ≤ cost ≤ 1,200 | 2,142 | 32.6% |
| High-end | 1,200 < cost ≤ 3,000 | 92 | 1.4% |
| Luxury | cost > 3,000 | 0 | 0.0% |

**Assessment:** The distribution is heavily skewed toward Budget restaurants, consistent with the dataset's India-dominant composition (low local cost of dining). The High-end tier has limited representation (92 restaurants); caution is advised when drawing statistical conclusions from this group. The Luxury tier is empty after IQR filtering (all restaurants with cost > 3,000 were flagged as outliers), so the category is preserved in code but will not appear in the cleaned dataset.

---

### 2.4 Engineered Feature — `Primary_Cuisine`

The dataset contains **200+ distinct primary cuisines**. The distribution is long-tailed, with a small number of dominant cuisines:

| Primary Cuisine | Count |
|---|---|
| North Indian | 1,944 |
| Chinese | 540 |
| Cafe | 518 |
| Fast Food | 478 |
| Bakery | 431 |
| American | 242 |
| South Indian | 201 |
| Pizza | 188 |
| Street Food | 160 |
| Italian | 159 |
| *(others)* | ~1,507 |

**Assessment:** Top 10 cuisines account for ~77% of all records. Long-tail cuisines (< 10 restaurants each) have insufficient sample sizes for robust statistical inference. Analyses relying on `Primary_Cuisine` should either group rare cuisines into an `'Other'` category or restrict to cuisines with n ≥ 30 samples.

---

### 2.5 Geographic Distribution — `Country Name`

| Country | Count | % |
|---|---|---|
| India | 5,726 | 87.2% |
| United States | 431 | 6.6% |
| United Kingdom | 79 | 1.2% |
| UAE | 60 | 0.9% |
| South Africa | 58 | 0.9% |
| *(others)* | ~213 | ~3.2% |

**Assessment:** India dominates the dataset. Cross-country analyses must account for this extreme imbalance and the confounding effect of currency (different `Average Cost for two` scales across countries). Consider normalising cost by country median before multi-country comparisons.

---

## 3. Outlier Treatment

### 3.1 Method

**Tukey IQR Fence:**  
$$\text{Valid range} = [Q_1 - 1.5 \times IQR,\; Q_3 + 1.5 \times IQR]$$

The IQR method was selected because:
1. Both target columns are **non-Gaussian** (right-skewed with heavy tails).
2. It is **non-parametric** — no distributional assumptions required.
3. It is more **robust** than Z-score when the data includes multiple country/currency groups.

### 3.2 Results

#### `Average Cost for two`

| Statistic | Raw | After IQR filter |
|---|---|---|
| Count | 9,551 | 6,567 |
| Mean | 1,199 | 474 |
| Std | 16,121 | 300 |
| Min | 0 | 0 |
| Max | 800,000 | 1,350 |
| Q1 | 250 | 250 |
| Median | 400 | 450 |
| Q3 | 700 | 650 |

- IQR bounds: **[−350, 1,250]** (lower bound effectively 0 given non-negative cost)  
- Outliers removed: **853 rows**  
- Primary cause: Multi-currency mixing (JPY, IDR, ZAR prices coexist with INR, causing extreme nominal values)

#### `Aggregate rating`

| Statistic | Raw | After IQR filter |
|---|---|---|
| Count | 9,550 | 6,567 |
| Mean | 2.67 | 3.40 |
| Std | 1.52 | 0.54 |
| Min | 0.0 | 1.8 |
| Max | 4.9 | 4.9 |
| Q1 | 2.5 | 3.0 |
| Median | 3.2 | 3.4 |
| Q3 | 3.7 | 3.8 |

- IQR bounds: **[0.7, 5.5]**  
- Outliers removed: **~2,130 rows** (primarily "Not rated" restaurants with rating = 0.0)  
- Combined rows removed (both filters): **2,983**

### 3.3 Decision on Retention

Outliers were **removed** rather than winsorised or capped because:
- Currency mixing makes normalisation impractical without external exchange-rate data.
- "Not rated" restaurants are structurally different from rated ones and would introduce systematic bias into rating-prediction models.
- The cleaned dataset retains 6,567 / 9,552 = **68.7%** of original records — sufficient for robust analysis.

---

## 4. Data Schema

### 4.1 Cleaned Dataset: `zomato_clean_all.csv`

**Dimensions:** 6,567 rows × 24 columns

| # | Column | Dtype | Description | Origin |
|---|---|---|---|---|
| 1 | Restaurant ID | object | Unique restaurant identifier | Raw |
| 2 | Restaurant Name | object | Restaurant display name | Raw |
| 3 | Country Name | object | Country of operation | Raw |
| 4 | Country Code | float64 | Numeric country code | Raw |
| 5 | City | object | City of operation | Raw |
| 6 | Address | object | Street address | Raw |
| 7 | Locality | object | Sub-area / neighbourhood | Raw |
| 8 | Locality Verbose | object | Full locality string | Raw |
| 9 | Longitude | float64 | Geographic longitude (decimal degrees) | Raw (cleaned) |
| 10 | Latitude | float64 | Geographic latitude (decimal degrees) | Raw (cleaned) |
| 11 | Cuisines | object | Comma-separated cuisine list | Raw |
| 12 | Average Cost for two | float64 | Average meal cost for two (local currency) | Raw |
| 13 | Currency | object | Local currency code/symbol | Raw |
| 14 | Has Table booking | int64 | 1 = accepts table bookings, 0 = does not | Raw (encoded) |
| 15 | Has Online delivery | int64 | 1 = offers online delivery, 0 = does not | Raw (encoded) |
| 16 | Is delivering now | int64 | 1 = currently delivering, 0 = not | Raw (encoded) |
| 17 | Switch to order menu | int64 | 1 = has order menu, 0 = does not | Raw (encoded) |
| 18 | Price range | float64 | Zomato price tier 1–4 | Raw |
| 19 | Aggregate rating | float64 | Overall restaurant rating (0–5 scale) | Raw |
| 20 | Rating color | object | Zomato rating colour label | Raw |
| 21 | Rating text | object | Zomato rating text label | Raw |
| 22 | Votes | float64 | Number of user votes/reviews | Raw |
| 23 | Primary_Cuisine | object | First-listed cuisine type | **Engineered** |
| 24 | Price_Level | object | Discretised cost tier (Budget / Mid-range / High-end / Luxury) | **Engineered** |

### 4.2 Data Types Summary

| Type | Count | Columns |
|---|---|---|
| `object` (string) | 10 | Restaurant ID, Restaurant Name, Country Name, City, Address, Locality, Locality Verbose, Cuisines, Currency, Rating color, Rating text, Primary_Cuisine, Price_Level |
| `float64` | 10 | Country Code, Longitude, Latitude, Average Cost for two, Price range, Aggregate rating, Votes + type-coerced numeric |
| `int64` | 4 | Has Table booking, Has Online delivery, Is delivering now, Switch to order menu |

### 4.3 Engineered Features Detail

| Feature | Type | Values | Construction Method |
|---|---|---|---|
| `Primary_Cuisine` | Categorical | 200+ unique cuisine names | `Cuisines.split(',')[0].strip()` |
| `Price_Level` | Ordinal (string) | Budget, Mid-range, High-end, Luxury | Threshold binning on `Average Cost for two` |

---

## 5. Readiness for Analysis

The cleaned dataset `zomato_clean_all.csv` satisfies all requirements for the following planned tasks:

| Task | Status |
|---|---|
| Exploratory Data Analysis (EDA) | ✅ Ready |
| Data Visualisation (rating by city, cuisine, price) | ✅ Ready |
| Feature correlation analysis | ✅ Ready |
| Machine learning — regression (predict rating) | ✅ Ready |
| Machine learning — classification (predict price tier) | ✅ Ready |
| Hypothesis testing (price vs. rating, cuisine vs. rating) | ✅ Ready |
| Geographic analysis (Longitude/Latitude) | ✅ Ready |
