Run notebook cells sequentially to reproduce the entire preprocessing and generate the clean file.
What Makes a Restaurant Successful? 
— Data-Driven Analysis Based on Zomato RatingsModule: 
Data Acquisition & Preprocessing
## 1. Module Overview
This project conducts data analysis and modeling based on the public Zomato global restaurant dataset. 
A fully reproducible data preprocessing pipeline is implemented in Jupyter Notebook. 
We systematically fix common data quality issues including missing values, duplicate records, format errors and outliers. 
Feature engineering is also completed to produce standardized, analysis-ready data, which supports subsequent exploratory analysis, feature selection, model training and hypothesis testing.

## 2. Raw Dataset & Reproducible Acquisition Script2.1 Raw Dataset Information
Raw file: zomato_restaurants.csv
Raw data size: 9552 restaurant records with 23 original attributes
Content: Restaurant name, address, location, cuisines, average cost, aggregate rating, votes, table booking & online delivery status and other business fields.
### 2.2 Reproducible Data Loading ScriptRun the following code in Jupyter Notebook to initialize environment and load raw data:python运行import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#Set display format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#Load raw dataset (update file path for local use)
df = pd.read_csv("zomato_restaurants.csv")
print(f"Raw data shape: {df.shape}")

## 3. Preprocessing Log (Transformations & Justification)All data transformation steps and corresponding reasons are documented below:

### 3.1 Remove useless index column
Action: Drop the redundant column Unnamed: 0
Justification: This auto-generated index column has no business meaning and increases data redundancy.

### 3.2 Deduplication
Action: Remove fully duplicated rows and duplicate records by Restaurant ID
Justification: Ensure each restaurant has only one unique record, avoiding biased statistics and modeling results.

### 3.3 Missing value handling
Action: Fill text fields with Unknown; impute numerical fields with median values; drop records with missing core rating values
Justification: Unknown text content is uniformly marked; median is robust to outliers;
records without valid ratings have no analytical value.

### 3.4 Format correction & data type conversion
Action: Remove extra quotes in longitude & latitude and convert to float; convert Yes/No binary fields to 0/1
Justification: Cleaned numeric coordinates support mathematical calculation; 
binary numeric format meets the input requirements of machine learning models.

### 3.5 Outlier removal
Action: Apply IQR method to remove extreme outliers from Average Cost for two and Aggregate rating
Justification: Extreme outliers will distort data distribution and reduce model accuracy; 
IQR is a reliable method for outlier filtering.

### 3.6 Feature engineering
Action 1: Extract Primary_Cuisine from combined cuisine strings
Justification: Split concatenated text into single cuisine for grouped analysis.
Action 2: Create Price_Level by dividing average cost into 4 categories
Justification: Discretize continuous values into categorical levels to match research hypotheses and layered analysis.

### 3.7 Export integrated dataset
Action: Output one complete dataset without city segmentation
Justification: Maintain global data integrity for overall analysis.

## 4. Cleaned Analysis-Ready Dataset
Output file: zomato_clean_all.csv
Description: Fully preprocessed dataset with derived features, ready for formal data analysis.
Usage: Exploratory Data Analysis (EDA), data visualization, feature engineering, machine learning modeling and hypothesis validation.

## 5. Data Quality Summary5.1 Null Rates
Raw data: Multiple columns contained missing values, including core rating field.
Processed data: Overall null rate = 0%, no missing values across all columns. Data completeness is guaranteed.
### 5.2 Class Balance
Binary fields (table booking, online delivery): Natural unbalanced distribution, retained as real business status without resampling.
Price_Level: 4 price categories with reasonable distribution matching real catering consumption.
Primary_Cuisine: Head cuisines have more samples while long-tail cuisines have fewer samples. Original distribution is reserved for real-scenario analysis.
### 5.3 Outlier Treatment
Method: Interquartile Range (IQR) rule: \(Q1-1.5\times IQR \sim Q3+1.5\times IQR\)
Target fields: Average Cost for two, Aggregate rating
Result: Extreme outliers are removed, and the overall data distribution becomes stable.
### 5.4 Data Schema
Total columns: 25 (23 original columns + 2 newly derived features)
Data types: String, float, integer and category with unified format.
New features:

Primary_Cuisine: Main cuisine of each restaurant (categorical feature)
Price_Level: Consumption tier of each restaurant (categorical feature)


## 6. How to Run
Place zomato_restaurants.csv and Jupyter Notebook in the same folder.
Modify the file path in code according to your local directory.
Execute notebook cells in sequence to generate the final cleaned dataset automatically.