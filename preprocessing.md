# Preprocessing Log — Zomato Restaurant Dataset

**Project:** What Makes a Restaurant Successful? — Data-Driven Analysis Based on Zomato Ratings  
**Module:** M2 & M3 — Data Acquisition & Preprocessing  
**Team:** T21 — Wenxuan Wu · Jiarui Zhai · Zilin Wang · Jiayang Li · Yuchen Zhang  
**Date:** 2026-05-31  
**Notebook:** `part1.ipynb`  
**Input file:** `data/zomato_restaurants.csv` (9,552 rows × 23 columns)  
**Output file:** `data/zomato_clean_all.csv` (6,567 rows × 24 columns)

---

## Overview

This document records every data transformation applied to the raw Zomato dataset, together with the justification for each decision. Steps are numbered to match the corresponding cells in `part1.ipynb`. The pipeline is fully reproducible: running all notebook cells in sequence regenerates `zomato_clean_all.csv` from the raw CSV.

---

## Step 1 — Environment Setup & Raw Data Loading

**Action:** Import `pandas`, `numpy`, and `warnings`; configure display settings; resolve file paths relative to the notebook directory; load `zomato_restaurants.csv`.

**Justification:**  
Using `os.path` for path resolution ensures the notebook runs correctly on any machine regardless of local directory structure, removing the hard-coded absolute path that was present in the original version. Suppressing `SettingWithCopyWarning` keeps output readable without masking real errors.

**Result:**  
- Raw shape: **9,552 rows × 23 columns**

---

## Step 2 — Drop Redundant Index Column

**Action:** Drop column `Unnamed: 0` if present.

**Justification:**  
`Unnamed: 0` is an auto-generated default integer index written to the CSV on export. It duplicates the DataFrame's built-in row index and provides no business meaning. Retaining it would add noise during correlation analysis and inflate feature counts.

**Result:**  
- Columns reduced from 23 → 22.

---

## Step 3 — Deduplication

**Action:**  
1. Remove fully identical rows (`df.drop_duplicates()`).  
2. Remove rows with duplicate `Restaurant ID`, keeping the first occurrence.

**Justification:**  
Duplicate records would cause double-counting in aggregations (e.g., per-country average rating), inflate vote totals, and bias model training. `Restaurant ID` is the natural primary key of this dataset; any restaurant appearing more than once is a data artefact. First-occurrence retention preserves the most likely original entry.

**Result:**  
- Fully duplicated rows removed: **0**  
- Duplicate-by-ID rows removed: **0**  
- Shape unchanged at **9,552 rows × 22 columns**

*(Note: The raw Zomato dataset is already de-duplicated at source. The deduplication step is retained in the pipeline as a defensive guard against future data refreshes.)*

---

## Step 4 — Missing Value Handling

### 4a — Anomalous Value Correction

**Action:** Replace the value `'Orange'` in `Has Table booking` with `'No'`.

**Justification:**  
`Has Table booking` is a boolean Yes/No field. The single occurrence of `'Orange'` is a clear data entry error (likely a mis-paste from a color code field). It is corrected to `'No'` — the conservative default — before downstream binary encoding.

**Result:** 1 anomalous value corrected.

---

### 4b — Text / Categorical Fields (fill with `'Unknown'`)

**Columns:** `Restaurant Name`, `Cuisines`, `Locality`, `Locality Verbose`, `Address`

**Action:** Fill NaN with the string `'Unknown'`.

**Justification:**  
These fields are used for grouping and display, not arithmetic. Using `'Unknown'` preserves the row for non-text analyses (e.g., rating models) while making the missingness explicit and filterable. Dropping rows for missing locality would unnecessarily reduce the dataset, as restaurant ratings and cost figures are still valid.

**Result:**  
| Column | Nulls filled |
|---|---|
| Locality | 1 |
| Locality Verbose | 1 |
| Cuisines | 10 |
| Restaurant Name, Address | 0 |

---

### 4c — Binary Yes/No Fields (fill with `'No'`)

**Columns:** `Has Table booking`, `Has Online delivery`, `Is delivering now`, `Switch to order menu`

**Action:** Fill NaN with `'No'`.

**Justification:**  
These binary flags indicate feature availability. A missing value most plausibly means the feature is not offered (i.e., not available = No). A conservative default avoids inflating the count of feature-enabled restaurants.

**Result:**  
| Column | Nulls filled |
|---|---|
| Has Table booking | 1 |
| Has Online delivery | 1 |
| Is delivering now | 1 |
| Switch to order menu | 2 |

---

### 4d — Numerical Fields (median imputation)

**Columns:** `Average Cost for two`, `Votes`

**Action:** Fill NaN with the column's median value.

**Justification:**  
Median is robust to the heavy-tailed, right-skewed distributions observed in both `Average Cost for two` (raw max: 800,000; raw std: 16,121) and `Votes`. Using the mean would distort the imputed value upward due to extreme outliers. Only 1–2 rows were affected; the overall distributional impact is negligible.

**Result:**  
| Column | Nulls filled | Imputed value |
|---|---|---|
| Average Cost for two | 1 | 400.0 |
| Votes | 2 | 74.0 |

---

### 4e — Other Categorical Fields

**Action:**  
- `Currency` → fill with `'Unknown'`  
- `Rating color` → fill with `'White'` (Zomato's color code for "Not rated")  
- `Rating text` → fill with `'Not rated'`  
- `Country Code` → fill with `0` (placeholder for unknown)  
- `Price range` → fill with column median

**Justification:**  
Each fill value is chosen to be internally consistent with the existing coding scheme of the respective field, preserving data integrity without introducing fabricated information.

**Result:** All remaining structural nulls cleared.

---

### 4f — Drop Rows with Missing `Aggregate rating`

**Action:** Drop rows where `Aggregate rating` is NaN.

**Justification:**  
`Aggregate rating` is the primary target variable for regression and classification tasks throughout this project. A restaurant with no rating cannot contribute to any analysis objective. Imputing ratings is inadvisable as it would introduce artificial data into the response variable.

**Result:** 2 rows dropped. Shape: **9,550 rows × 22 columns**.

---

## Step 5 — Format Correction & Data Type Conversion

### 5a — Longitude / Latitude String Cleaning

**Action:** Strip leading/trailing single-quotes from `Longitude` and `Latitude` string values; convert to `float64`.

**Justification:**  
Several rows store coordinate values as quoted strings (e.g., `"'120.98'"`) — a formatting artefact likely caused by a spreadsheet export. Numeric type is required for geographic distance calculations, spatial clustering (e.g., DBSCAN), and map visualisations.

**Result:** Both columns converted to `float64`; no coercion failures.

### 5b — Yes/No Binary Encoding

**Columns:** `Has Table booking`, `Has Online delivery`, `Is delivering now`, `Switch to order menu`

**Action:** Map `'Yes' → 1`, `'No' → 0`.

**Justification:**  
Scikit-learn and other ML libraries require numeric input. Integer encoding is appropriate here because the fields are nominally binary with no ordinal meaning — one-hot encoding would produce identical information.

**Result:** All four columns converted to `int64`.

### 5c — Enforce Numeric Types

**Action:** Apply `pd.to_numeric(..., errors='coerce')` to `Aggregate rating`, `Average Cost for two`, `Votes`, `Price range`, `Country Code`.

**Justification:**  
Some columns were loaded as `object` dtype due to mixed-type entries in the raw CSV. Explicit conversion guarantees that downstream arithmetic and statistical operations behave correctly.

---

## Step 6 — Outlier Removal (IQR Method)

**Action:** For `Average Cost for two` and `Aggregate rating`, remove rows outside the range `[Q1 − 1.5 × IQR, Q3 + 1.5 × IQR]`.

**Justification:**  
The raw `Average Cost for two` distribution has a maximum of 800,000 (currency-mixed; e.g., Japanese restaurants priced in JPY coexist with Indian restaurants priced in INR). These extreme values would dominate distance-based models and distort regression coefficients. For `Aggregate rating`, the "Not rated" group (rating = 0) was already partially handled in Step 4f; IQR filtering removes any remaining statistical outliers.

The IQR (Tukey fence) method is chosen over Z-score because the distributions are non-Gaussian and right-skewed. It is a non-parametric method that makes no distributional assumptions.

**Result:**

| Column | Q1 | Q3 | IQR | Lower bound | Upper bound | Rows removed |
|---|---|---|---|---|---|---|
| Average Cost for two | 250 | 650 | 400 | −350 | 1,250 | ~853 |
| Aggregate rating | 2.5 | 3.7 | 1.2 | 0.7 | 5.5 | ~2,148 |

*(The Aggregate rating filter primarily removes "Not rated" (0.0) restaurants.)*

- Rows removed (combined): **2,983**  
- Shape after outlier removal: **6,567 rows × 22 columns**

---

## Step 7 — Feature Engineering

### 7a — `Primary_Cuisine`

**Action:** Split `Cuisines` on `','` and extract the first token; strip whitespace.

**Justification:**  
`Cuisines` stores multi-valued strings (e.g., `"North Indian, Chinese, Fast Food"`), making direct use in categorical models impractical. The first-listed cuisine is treated as the restaurant's primary offering, a common convention in restaurant data analysis. This enables grouped analysis by cuisine type (research question: does cuisine type influence rating?).

**Result:** New column `Primary_Cuisine` with 200+ unique values; top value: `North Indian` (1,944 restaurants).

### 7b — `Price_Level`

**Action:** Map `Average Cost for two` to one of four ordered string labels: `Budget`, `Mid-range`, `High-end`, `Luxury`.

**Thresholds:**  
| Tier | Condition |
|---|---|
| Budget | cost < 600 |
| Mid-range | 600 ≤ cost ≤ 1,200 |
| High-end | 1,200 < cost ≤ 3,000 |
| Luxury | cost > 3,000 |

**Justification:**  
Discretisation converts a heavily skewed continuous variable into an interpretable ordinal feature. The thresholds are informed by the IQR-filtered quartile distribution (Q1 ≈ 250, Q3 ≈ 650) and aligned with the research hypothesis that price tier mediates the relationship between restaurant features and rating. Categorical tiers are also more robust to residual currency-mixing artefacts than raw numeric cost.

**Result:**  
| Tier | Count |
|---|---|
| Budget | 4,333 |
| Mid-range | 2,142 |
| High-end | 92 |
| Luxury | 0 |

---

## Step 8 — Export Cleaned Dataset

**Action:** Write `df_clean` to `data/zomato_clean_all.csv` with `encoding='utf-8-sig'` and `index=False`.

**Justification:**  
A single unified file maintains global data integrity for all downstream analyses. City-level splits (e.g., Bangalore-only) can be produced on demand in analysis notebooks without loss of generalisability. `utf-8-sig` encoding ensures the BOM marker is included, preventing encoding issues in Excel and Windows tools.

**Result:**  
- Output file: `data/zomato_clean_all.csv`  
- Final shape: **6,567 rows × 24 columns** (22 original + 2 engineered features)  
- Total null cells: **0**

---

## Summary of Row / Column Changes

| Stage | Rows | Columns |
|---|---|---|
| Raw data loaded | 9,552 | 23 |
| After drop index column | 9,552 | 22 |
| After deduplication | 9,552 | 22 |
| After missing value treatment | 9,550 | 22 |
| After type conversion | 9,550 | 22 |
| After outlier removal | 6,567 | 22 |
| After feature engineering | 6,567 | 24 |
| **Final output** | **6,567** | **24** |
