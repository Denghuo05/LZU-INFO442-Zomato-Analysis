# M6: Model Card - Restaurant High-Rating Classifier

## 1. Model Description

**Model Type:** Binary Classification  
**Algorithm:** XGBoost Classifier (with Logistic Regression as baseline)  
**Target Variable:** `HighRating` — 1 if `Aggregate rating ≥ 4.0`, else 0  

**Input Features:**

| Feature | Type | Description |
|---------|------|-------------|
| `log_Votes` | Numeric | Log‑transformed number of user reviews |
| `Price range` | Ordinal | 1–4 scale |
| `Has Table booking` | Binary | Table booking availability |
| `Has Online delivery` | Binary | Online delivery availability |
| Cuisine dummies | Binary | Top‑10 cuisines + "Other" |

**Training Data:** 6,567 cleaned restaurant records from Zomato (global, multiple countries)  
**Train/Test Split:** 80/20 stratified  
**Class Distribution (high‑rated):** 16.2%

---

## 2. Intended Use

**Primary Use Case:**  
Identify key drivers of high restaurant ratings to support business decisions for restaurant owners and platform managers.

**Who Should Use This Model:**
- Restaurant owners seeking actionable improvement strategies
- Platform managers (Zomato) for feature prioritisation and user engagement
- Data analysts exploring restaurant reputation dynamics

**Out‑of‑Scope Uses:**
- Predicting exact numerical ratings (the model is optimised for binary classification)
- Causal inference — the model identifies correlations, not causal relationships
- Generalisation to restaurants not listed on Zomato or to other geographic regions without validation

---

## 3. Performance

**Cross‑Validation Results (Stratified 5‑fold):**

| Model | Accuracy (±std) | F1‑score (±std) | AUC‑ROC (±std) |
|-------|----------------|-----------------|----------------|
| XGBoost | 0.8884 ± 0.0077 | 0.6187 ± 0.0203 | 0.9104 ± 0.0058 |
| Logistic Regression | 0.8852 ± 0.0084 | 0.5927 ± 0.0136 | 0.9126 ± 0.0093 |

**Interpretation:**  
The model achieves excellent discriminative power (AUC > 0.91). F1‑score is moderate (0.62) due to class imbalance (only 16.2% high‑rated), but the high AUC indicates reliable ranking of restaurants by likelihood of being high‑rated.

**Feature Importance (Permutation Importance):**

| Feature | Relative Importance |
|---------|---------------------|
| `log_Votes` | ~70% |
| `Price range` / `Average Cost` | ~20% |
| Cuisine dummies | ~5–10% |
| `Has Online delivery` / `Has Table booking` | ~2% each |

---

## 4. Limitations

| Limitation | Description |
|------------|-------------|
| **Static Data** | The dataset is a snapshot; trends, seasonality, and temporal dynamics are not captured. |
| **Confounded Features** | `Votes` is correlated with visibility and platform engagement — increasing votes may not cause higher ratings. |
| **Arbitrary Threshold** | The binary cutoff (≥ 4.0) may not suit all business contexts; ordinal classification could offer finer granularity. |
| **Missing Variables** | Review sentiment, service quality, and precise location data are absent. |
| **Sampling Bias** | Data includes only Zomato‑listed restaurants; findings may not generalise to unlisted establishments. |
| **Geographic Bias** | Country‑level differences suggest local rating behaviours that the model does not explicitly account for. |

---

## 5. Recommendations

**For Deployment:**
- Use probability outputs rather than binary predictions to allow threshold tuning.
- Monitor model performance over time as consumer ptions.

---
