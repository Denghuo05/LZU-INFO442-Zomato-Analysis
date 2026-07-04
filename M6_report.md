# M6 : What Makes a Restaurant Successful?
## A Data-Driven Analysis Based on Zomato Ratings

**Project:** Data Science Project  
**Team:** Group 21  
**Date:** 2026-07-02  

---

## Table of Contents

1. [Project Overview and Problem Definition](#1-project-overview-and-problem-definition)
   - 1.1 Background
   - 1.2 Problem Definition and Research Questions
   - 1.3 Research Hypotheses
2. [Data Acquisition and Preprocessing (M1–M3)](#2-data-acquisition-and-preprocessing-m1m3)
   - 2.1 Raw Dataset Overview
   - 2.2 Data Quality Assessment
   - 2.3 Preprocessing Pipeline
   - 2.4 Feature Engineering
   - 2.5 Data Quality Summary
3. [Exploratory Data Analysis (M4)](#3-exploratory-data-analysis-m4)
   - 3.1 Univariate Analysis
   - 3.2 Bivariate Analysis
   - 3.3 Multivariate Analysis
   - 3.4 Correlation and Normality Assessment
   - 3.5 Random Forest Feature Importance (Preliminary)
   - 3.6 Key Insights from EDA
4. [Predictive Modeling (M5)](#4-predictive-modeling-m5)
   - 4.1 Modeling Objective and Target Definition
   - 4.2 Feature Engineering for Modeling
   - 4.3 Model Selection
   - 4.4 Evaluation Metrics
   - 4.5 Cross-Validation Results
   - 4.6 Feature Importance and Decision Tree
   - 4.7 Model Limitations
5. [Discussion and Business Implications](#5-discussion-and-business-implications)
   - 5.1 Summary of Key Findings
   - 5.2 Actionable Recommendations
6. [Conclusion](#6-conclusion)

---

## 1. Project Overview and Problem Definition

### 1.1 Background

The restaurant industry is highly competitive, and online platforms like Zomato have become essential channels for customer acquisition and brand visibility. Ratings on these platforms serve as powerful signals that influence consumer choice, directly impacting restaurant revenue and reputation. Understanding what drives high ratings is therefore crucial for both restaurant owners seeking to improve their establishments and platform managers aiming to optimize user experience.

Zomato, one of the world's largest restaurant discovery platforms, aggregates millions of user reviews and ratings across global markets. The platform's aggregate rating functions as a de facto quality signal, yet the determinants of these ratings — whether they reflect objective food quality, popularity, pricing strategy, or service features — remain incompletely understood.

### 1.2 Problem Definition and Research Questions

This project addresses the following primary research question:

> **"What factors distinguish high-performing restaurants (aggregate rating ≥ 4.0) from others on the Zomato platform?"**

The original regression problem (predicting exact numerical ratings) is reframed as a binary classification task. This decision is justified by two key observations from the exploratory analysis:

| Rationale | Explanation |
|-----------|-------------|
| **Statistical** | The distribution of ratings is non‑normal (Shapiro‑Wilk test: p < 0.001), violating key assumptions of ordinary least squares regression. |
| **Practical** | Business stakeholders are more interested in identifying what makes a restaurant "excellent" rather than predicting minor rating variations. |
| **Actionability** | Classification enables clear decision‑making and targeted improvement strategies. |

### 1.3 Research Hypotheses

Based on domain knowledge and preliminary data inspection, the following hypotheses were formulated:

| ID | Hypothesis |
|----|------------|
| H1 | Higher‑priced restaurants receive higher aggregate ratings. |
| H2 | Restaurants offering online delivery have higher ratings. |
| H3 | Restaurants with more votes (greater popularity) receive higher ratings. |
| H4 | Cuisine type significantly moderates the price‑rating relationship. |

---

## 2. Data Acquisition and Preprocessing (M1–M3)

### 2.1 Raw Dataset Overview

The raw dataset contains **9,552 restaurant records** with **23 attributes** per restaurant. Key variables include:

| Variable | Type | Description |
|----------|------|-------------|
| Restaurant ID | Integer | Unique identifier |
| Country Name / City | Categorical | Geographic location |
| Cuisines | Text | Comma‑separated cuisine list |
| Average Cost for two | Numeric | Cost in local currency |
| Has Table booking / Has Online delivery | Binary | Feature availability flags |
| Aggregate rating | Numeric (0–5) | Primary target variable |
| Votes | Numeric | Number of user reviews |

The dataset exhibits several data quality issues requiring systematic treatment: missing values, format inconsistencies, outliers, and redundant columns.

### 2.2 Data Quality Assessment

The raw data quality assessment revealed the following issues:

| Issue | Details |
|-------|---------|
| Redundant column | `Unnamed: 0` — auto‑generated index with no business meaning |
| Missing values | 31 null cells across multiple columns |
| Text columns | `Cuisines` (10 nulls), `Locality` (1 null), `Locality Verbose` (1 null) |
| Binary columns | `Has Table booking`, `Has Online delivery`, `Is delivering now`, `Switch to order menu` have nulls |
| Numerical columns | `Average Cost for two` (1 null), `Votes` (2 nulls) |
| Format issues | `Longitude` and `Latitude` stored as strings with extra quotes |
| Anomalous value | `Has Table booking` contains `"Orange"` instead of `"Yes"`/`"No"` |

### 2.3 Preprocessing Pipeline

A fully reproducible preprocessing pipeline was implemented in Python using pandas and numpy. The major steps are documented below:

| Step | Action | Rationale |
|------|--------|-----------|
| 1 | Drop `Unnamed: 0` index column | Redundant auto‑generated index with no business meaning. |
| 2 | Remove duplicate rows and duplicate `Restaurant ID`s | Ensure one unique record per restaurant. |
| 3 | Fill missing text fields with `"Unknown"` | Preserve rows for numeric analyses; make missingness explicit. |
| 4 | Fill binary flags with `"No"` | Conservative default — assume feature is absent. |
| 5 | Impute numeric fields (Cost, Votes) with median | Median is robust to heavy‑tailed, right‑skewed distributions. |
| 6 | Drop rows missing `Aggregate rating` | Target variable cannot be imputed. |
| 7 | Convert Yes/No to 0/1; clean longitude/latitude strings | Ensure correct data types for modeling and geospatial use. |
| 8 | Remove outliers via IQR (Cost & Rating) | Non‑parametric method appropriate for non‑Gaussian distributions. |

### 2.4 Feature Engineering

**Primary_Cuisine**
- Extracted the first cuisine from the comma‑separated `Cuisines` string.
- This simplifies cuisine‑based grouping and reduces dimensionality from hundreds of unique combinations to a more manageable set of primary cuisines.

**Price_Level**
- Discretized `Average Cost for two` into four ordered tiers based on the quartile distribution after outlier removal:

| Tier | Condition |
|------|-----------|
| Budget | cost < 600 |
| Mid-range | 600 ≤ cost ≤ 1,200 |
| High-end | 1,200 < cost ≤ 3,000 |
| Luxury | cost > 3,000 |

- The thresholds are aligned with the research hypotheses on price‑rating relationships.

### 2.5 Data Quality Summary

| Metric | Raw Data | Processed Data |
|--------|----------|----------------|
| Rows | 9,552 | 6,567 |
| Columns | 23 | 24 |
| Null cells | 31 | 0 |
| Duplicate rows | 0 | 0 |
| Outliers removed | — | 2,983 rows |
| New features | — | Primary_Cuisine, Price_Level |

The cleaned dataset (`zomato_clean_all.csv`) serves as the foundation for all downstream analysis and modeling.

---

## 3. Exploratory Data Analysis (M4)

### 3.1 Univariate Analysis

#### 3.1.1 Distribution of Ratings

The histogram of `Aggregate rating` reveals a **left‑skewed distribution** concentrated in the 3.0–4.5 range, with a peak around 3.5. The boxplot confirms the skew: the median is approximately 3.5, with mild outliers below 2.0. Very few restaurants achieve ratings below 2.0.

**Key insights from the distribution:**
- The scarcity of low ratings means binary classification (high‑rated vs. others) requires handling class imbalance.
- The non‑normal distribution supports reframing the problem as classification rather than regression.
- The target distribution supports a threshold of ≥ 4.0 for "high rating" — aligned with Zomato's "Excellent" category.

#### 3.1.2 Price Range Frequency

Most restaurants fall into price ranges 2 (medium) and 3 (high), comprising approximately 80% of the dataset. Range 4 (very high) is sparse (<10%). This concentration suggests that high‑end restaurants are under‑represented, potentially limiting statistical power for that segment.

### 3.2 Bivariate Analysis

| Feature Pair | Observation |
|--------------|-------------|
| **Cost vs. Rating** | Weak positive correlation (Pearson r ≈ 0.22, R² ≈ 0.05). Many low‑cost restaurants achieve high ratings (>4.0), while some high‑cost restaurants score poorly. Cost explains very little variance. |
| **Online delivery vs. Rating** | Delivery‑enabled restaurants have slightly higher median rating (≈3.7 vs 3.4). Distributions overlap substantially; many non‑delivery restaurants achieve high ratings. |
| **Table booking vs. Rating** | Similar pattern to online delivery, with a slightly larger median gap (≈3.8 vs 3.5). Still overlapping distributions. |
| **Votes (log) vs. Rating** | Clear positive trend; correlation (log votes) ≈ 0.35, higher than cost. Some high‑rated restaurants have very few votes (new or niche establishments). |

### 3.3 Multivariate Analysis

#### 3.3.1 Average Rating by Country

India, the United States, and the United Kingdom show the highest average ratings (>3.8). Lower‑rated countries have fewer observations.

**Interpretation:** Geographic differences may reflect rating behavior bias, quality standards, or sampling disparities. Country-level features may need to be incorporated in modeling.

#### 3.3.2 Cuisine × Price Range Interaction

A heatmap of average rating by cuisine and price range reveals a **significant interaction effect**:

| Cuisine | Price Range | Average Rating |
|---------|-------------|----------------|
| Japanese | 3 | ≈ 4.2 |
| Japanese | 4 | drops noticeably |
| Fast Food | 2 | ≈ 3.8 |
| Fast Food | 3+ | declines |
| North Indian | All | relatively stable |

**Interpretation:** The optimal price range varies by cuisine. Premium pricing works for some categories (Japanese, Italian) but not for others (Fast Food). This interaction is one of the most actionable findings from the analysis.

#### 3.3.3 Three‑Dimensional Relationship (Cost, Votes, Rating)

For a selected cuisine (North Indian), bubble size (votes) consistently clusters with high ratings, while cost shows no clear gradient. This reinforces the importance of **popularity over price** within the same cuisine.

### 3.4 Correlation and Normality Assessment

#### 3.4.1 Correlation Matrix

| Feature Pair | Correlation |
|--------------|-------------|
| Votes – Rating | 0.31 (strongest) |
| Price range – Cost | 0.73 (multicollinearity) |
| Cost – Rating | 0.19 (weak) |
| Price range – Rating | 0.20 (weak‑to‑moderate) |

**Key takeaways:**
- Cost and price range are nearly redundant — using both in linear models would cause multicollinearity.
- Votes have the strongest individual correlation with the target.
- Overall correlations are modest, suggesting ratings are influenced by complex interactions not captured by simple linear relationships.

#### 3.4.2 Normality Check

The Q‑Q plot shows significant deviation from the diagonal. The Shapiro‑Wilk test confirms non‑normality (p < 0.001). This further justifies the classification approach over standard linear regression.

#### 3.4.3 Cumulative Distribution Function (CDF)

- ~0.1% of restaurants have rating ≤ 2.0
- ~3.6% have rating ≥ 4.5
- ~60% of restaurants fall between 3.0 and 4.0

### 3.5 Random Forest Feature Importance (Preliminary)

A preliminary Random Forest regressor was trained on the cleaned numerical and binary features. The importance ranking is:

| Feature | Importance |
|---------|------------|
| Votes | ~0.70 |
| Average Cost | ~0.21 |
| Price range | ~0.05 |
| Online delivery | ~0.02 |
| Table booking | ~0.02 |

This strongly suggests that **votes (popularity) are the dominant predictor**, with price features playing a secondary role. Binary service features contribute minimally.

### 3.6 Key Insights from EDA

1. **Votes are the strongest predictor** of ratings, with a correlation of 0.31 and high feature importance (~0.70) in the Random Forest model.

2. **Cost and price range have weak individual effects** (r ≈ 0.19–0.20) but may interact with other features.

3. **Cuisine × price interaction** is significant and actionable — the optimal pricing strategy depends on cuisine type.

4. **Online delivery and table booking** contribute little to rating prediction when other features are controlled.

5. **Geographic variation exists** — country-level differences may need to be accounted for in modeling.

---

## 4. Predictive Modeling (M5)

### 4.1 Modeling Objective and Target Definition

We build a binary classifier to predict **HighRating = 1 if Aggregate rating ≥ 4.0, else 0**. This target aligns with Zomato's "Excellent" category and is more actionable for business decisions.

**Class distribution:** Only 16.2% of restaurants are high‑rated, indicating significant class imbalance that must be addressed in modeling.

### 4.2 Feature Engineering for Modeling

The final feature set for modeling includes:

| Feature | Type | Rationale |
|---------|------|-----------|
| `log_Votes` | Numeric | Log‑transform to linearise relationship with rating |
| `Price range` | Ordinal | Original 1–4 scale |
| `Has Table booking` | Binary | Feature availability |
| `Has Online delivery` | Binary | Feature availability |
| Cuisine dummies | Binary | Top‑10 cuisines + "Other" category |

Numerical features were standardised for logistic regression. The dataset was split 80/20 using stratified sampling to preserve class distribution.

### 4.3 Model Selection

Two models were chosen to balance interpretability and performance:

| Model | Purpose | Strengths | Weaknesses |
|-------|---------|-----------|------------|
| **Logistic Regression** | Interpretable baseline | Coefficients directly interpretable as log‑odds; clear business explanations | Assumes linear decision boundary |
| **XGBoost Classifier** | High‑performance prediction | Captures non‑linear interactions; robust to outliers; state‑of‑the‑art accuracy | Less interpretable; requires tuning |

### 4.4 Evaluation Metrics

Given the class imbalance, accuracy alone would be misleading. Three complementary metrics were used:

| Metric | Purpose |
|--------|---------|
| **Accuracy** | Overall correctness; baseline for comparison |
| **F1‑score** | Harmonic mean of precision and recall; more informative for imbalanced classes |
| **AUC‑ROC** | Measures discriminative ability across all thresholds |

### 4.5 Cross‑Validation Results

Stratified 5‑fold cross‑validation results are:

| Model | Accuracy (±std) | F1‑score (±std) | AUC (±std) |
|-------|----------------|-----------------|------------|
| Logistic Regression | 0.8852 ± 0.0084 | 0.5927 ± 0.0136 | 0.9126 ± 0.0093 |
| XGBoost | 0.8884 ± 0.0077 | 0.6187 ± 0.0203 | 0.9104 ± 0.0058 |

**Interpretation:**
- Both models achieve AUC > 0.91, indicating **excellent discriminative power**.
- XGBoost slightly outperforms Logistic Regression across all metrics, particularly on F1‑score, due to its ability to capture non‑linear interactions.
- The modest F1‑scores (around 0.59–0.62) reflect the challenge of predicting the minority class; however, the high AUC values confirm the models are reliable discriminators.

### 4.6 Feature Importance and Decision Tree

#### 4.6.1 Permutation Importance (XGBoost)

Permutation importance confirms the EDA findings:

| Feature | Relative Importance |
|---------|---------------------|
| `log_Votes` | ~70% |
| `Price range` / `Average Cost` | ~20% |
| Cuisine dummies | ~5–10% |
| `Has Online delivery` / `Has Table booking` | ~2% each |

#### 4.6.2 Decision Tree (Interpretable Rules)

A shallow decision tree (max depth = 3) provides a simple, actionable rule set:

1. **If `log_Votes > 6.84` (≈ 900 votes):** → **74.8%** chance of being high‑rated.
2. **Else if `Cuisine = Chinese` and `Price range > 2.5`:** → **41.3%** chance.
3. **Else:** → **~8%** chance.

**Business insight:** The tree provides a clear prioritisation:
- **First priority:** Increase visibility to drive votes.
- **Second priority:** Align pricing strategy with cuisine type.
- **Third priority:** Consider cuisine‑specific marketing campaigns.

### 4.7 Model Limitations

| Limitation | Description | Impact |
|------------|-------------|--------|
| **Data is static** | Snapshot data; trends, seasonality, and temporal dynamics not captured. | Model may become outdated as consumer preferences evolve. |
| **Votes are confounded** | Vote count is correlated with visibility, platform engagement, and restaurant age. | Causal inference is not established — increasing votes may not guarantee higher ratings. |
| **Arbitrary threshold** | The binary cutoff (≥ 4.0) may not be optimal for all contexts. | A more nuanced ordinal classification would be more precise. |
| **Cuisine encoding** | One‑hot encoding loses mixed‑cuisine information. | Loss of nuanced information may reduce accuracy. |
| **Missing variables** | No review sentiment, service quality, or exact location granularity. | Key factors may be missing. |

---

## 5. Discussion and Business Implications

### 5.1 Summary of Key Findings

| Finding | Detail |
|---------|--------|
| **Votes dominate** | Popularity (measured by review volume) is the strongest predictor of high ratings. Restaurants with >900 votes are 74.8% likely to be high‑rated. |
| **Price × Cuisine interaction** | Premium pricing works for Japanese and Italian cuisines but not for Fast Food or Street Food. Pricing strategy must be cuisine‑tailored. |
| **Binary service features are weak** | Online delivery and table booking availability contribute minimally when other features are controlled. |
| **Geographic variation exists** | Countries differ significantly in average ratings, suggesting cultural or sampling biases. |
| **Model performance** | Both models achieve AUC > 0.91, confirming strong discriminative ability. |

### 5.2 Actionable Recommendations

#### For Restaurant Owners

| Priority | Recommendation | Rationale |
|----------|----------------|-----------|
| 1 | Increase review volume via customer engagement | Votes are the strongest predictor of high ratings |
| 2 | Tailor pricing strategy to cuisine type | Premium pricing works for some cuisines, not all |
| 3 | Focus on cuisine‑specific marketing | Some cuisines consistently outperform others |
| 4 | Consider local context | Country‑level differences matter |

#### For Platform Managers (Zomato)

| Priority | Recommendation | Rationale |
|----------|----------------|-----------|
| 1 | Boost user engagement to drive review volume | Increases rating signals and platform quality |
| 2 | Provide cuisine‑specific benchmarks | Different cuisines have different dynamics |
| 3 | Support new restaurants' visibility | Cold‑start problem disproportionately affects new entries |
| 4 | Incorporate country‑specific features | Geographic variation requires localised strategies |

---

## 6. Conclusion

This project successfully identified the key drivers of high restaurant ratings on the Zomato platform through a systematic, end‑to‑end data science workflow. The main findings are:

1. **Popularity (votes) is the most important predictor**, with a correlation of 0.31 with ratings and dominant feature importance in both Random Forest and XGBoost models.

2. **Price strategy must be aligned with cuisine type** — the cuisine × price interaction is significant and actionable.

3. **Binary service features (online delivery and table booking) contribute minimally** when other features are controlled.

4. **Both classification models achieve AUC > 0.91**, confirming excellent discriminative power.

5. **The decision tree offers a simple rule set** for immediate business use: prioritize driving votes, then align price with cuisine.

The analysis provides practical guidance for restaurant owners seeking to improve their ratings and for platform managers optimizing user experience. Future work could extend this analysis with richer data (e.g., review sentiment, temporal trends) and explore more nuanced ordinal classification approaches.

---


```python

```
