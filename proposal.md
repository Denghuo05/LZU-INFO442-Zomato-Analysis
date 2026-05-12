# Project Proposal: What Makes a Restaurant Successful? A Data-Driven Analysis of Zomato Ratings

## 1. Domain and Motivation

The restaurant industry is highly competitive. Globally, more than half of new restaurants close within three to five years. Entrepreneurs and investors often rely on intuition when making decisions about location, pricing, and service type (dine-in, takeaway, or delivery). This leads to high uncertainty and frequent failures.

Online platforms like Zomato collect large amounts of structured data on restaurants, including ratings, price levels, location, and service options. These data offer an opportunity to move from intuition‑based decisions to data‑driven insights. Understanding which factors drive higher customer ratings can help new restaurant owners allocate resources more effectively, and can assist investors in evaluating potential opportunities.

Moreover, recommendation systems on food platforms suffer from the cold‑start problem for newly opened restaurants. A model that predicts a restaurant’s expected rating based only on its observable attributes (price, location, cuisine type, etc.) could help surface promising new venues earlier. Therefore, this project has both practical business value and algorithmic relevance.

## 2. Dataset Description

We will use the **Zomato Bangalore Restaurants** dataset, available on Kaggle (user: yashsahu02). The data were originally collected from the Zomato platform.

- **Source**: Kaggle (publicly available)
- **Size**: ~9,000 restaurant records
- **Format**: CSV, approximately 5–10 MB
- **Access method**: Direct download from Kaggle or via Kaggle API.
- **Key attributes**: restaurant name, location (neighbourhood), restaurant type (e.g., casual dining, quick bites), average cost for two persons, aggregate rating (scale 0–5), online delivery availability, table booking option, and more.

**Ethical considerations**: The dataset contains no personally identifiable information (PII). It is aggregated at the restaurant level. However, we acknowledge potential biases: the data are limited to Bangalore, India, so findings may not generalise to other cities or cultures. Ratings might be influenced by self‑selection (users who rate are often those with strong opinions). Also, the original collector is a Kaggle user, not an official Zomato source; we will cross‑check key statistics with any publicly available Zomato API data if feasible. We will not attempt to de‑anonymise any entity, and our analysis will be used only for academic learning purposes.

## 3. Scientific Question

**Primary research question:**  
How do average price, location (neighbourhood), restaurant type, and online delivery availability jointly influence the aggregate rating of a restaurant? Specifically, how much variance in ratings can each factor explain after controlling for the others?

This main question can be decomposed into three sub‑questions:

1. Is the relationship between average price and rating non‑linear (e.g., very low or very high prices associated with lower ratings)?
2. After controlling for price and restaurant type, do significant rating differences persist across neighbourhoods (e.g., Indiranagar vs. less central areas)?
3. Do the factors that determine ratings differ between restaurants that offer online delivery and those that do not?

**Measurable outcome**: We will build a regression model (and optionally tree‑based models) that predicts a restaurant’s rating. Performance will be measured using RMSE (root mean squared error) and R² on a held‑out test set. The final model will be interpretable, allowing us to quantify the marginal effect of each feature.

**Real‑world decision tie‑in**: The model can be used by a prospective restaurant owner to estimate the expected rating of a new restaurant given its planned attributes, and to identify which attribute changes (e.g., moving to a different neighbourhood) could yield the largest rating improvement.

## 4. Preliminary Hypotheses

Based on domain knowledge and prior literature, we propose three hypotheses:

- **H1 (Price‑rating curve)**: Restaurants with a medium price (around 600–1200 INR for two persons) receive the highest average ratings. Very cheap restaurants may compromise on quality, while very expensive restaurants raise customer expectations, making it harder to achieve a high rating.
- **H2 (Location effect)**: Restaurants located in busy commercial neighbourhoods (e.g., Indiranagar, Koramangala) have significantly higher ratings than those in residential areas, even after controlling for price and restaurant type. Intense competition in commercial centres drives out low‑quality establishments.
- **H3 (Delivery‑only vs. hybrid)**: Delivery‑only restaurants exhibit higher variance in ratings and a lower mean rating compared to restaurants that offer both dine‑in and delivery. The latter have more control over the full customer experience.

**Expected validation**: If supported, these hypotheses will offer actionable advice: open in a commercial district, set prices in the medium range, and offer both dine‑in and delivery to increase the probability of achieving a high rating.

## 5. Roles

Our team has five members. The responsibilities are divided as follows to ensure full coverage of the data science pipeline and smooth project management.

| Role                           | Responsible member | Main duties                                                                                   |
|--------------------------------|--------------------|-----------------------------------------------------------------------------------------------|
| **Data Engineering Lead**      | Yuchen Zhang (zhy1188) | Data acquisition, cleaning, deduplication, missing value handling, type conversion; build a reproducible preprocessing pipeline. |
| **Exploratory Analysis Lead**  | Zilin Wang (versta1ppen) | Visualisation, correlation analysis, outlier detection; generate descriptive statistics and initial insights; support hypothesis testing. |
| **Feature Engineering Lead**   | Wenxuan Wu (Denghuo05) | Construct features (e.g., price bins, location aggregates), encode categorical variables, select/reduce features; collaborate with the modelling lead. |
| **Modelling & Evaluation Lead**| Jiayang Li (jiangdachui495) | Select models (linear regression, random forest, XGBoost, etc.), perform train/test splits, hyperparameter tuning; compute metrics (RMSE, R²) and compare performance. |
| **Communication & Delivery Lead** | Jiarui Zhai (JiaruiZhai) | Manage project timeline, write proposal and final report, maintain GitHub repository (README, code organisation), prepare presentation slides; ensure results are interpretable and tell a coherent story. |

**Collaboration mechanism**: Every team member will also review or assist in at least one other area (e.g., Feature Engineering Lead helps with model interpretation, Communication Lead contributes to visualisation). All members participate in formulating and validating the scientific hypotheses.