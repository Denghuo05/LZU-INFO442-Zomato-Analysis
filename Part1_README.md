Data Engineering Preprocessing Module
Project: What Makes a Restaurant Successful? — Data-Driven Analysis Based on Zomato Ratings
Role: Data Engineering Lead (Yuchen Zhang)

## 1. Module Overview
This module completes the full data cleaning and preprocessing pipeline for the raw Zomato restaurant dataset using Jupyter Notebook. The workflow is fully reproducible. It fixes raw data errors, standardizes data formats, and produces high-quality cleaned data for subsequent exploratory analysis, feature engineering, statistical modeling and hypothesis testing.

## 2. Dataset Information
Raw file: zomato_restaurants.csv
Original size: 9552 restaurant records with 23 attributes
Contents: Restaurant basic information, location, cuisines, average cost, rating, table booking, online delivery and other attributes.

## 3. Preprocessing Workflow
Environment setup & data loadingImport libraries, configure display settings, load and inspect raw data structure.
Basic cleaningRemove useless index columns; drop duplicate rows; deduplicate by Restaurant ID to ensure unique records.
Missing value handlingFill text fields with "Unknown"; impute numerical fields with median values; drop records with missing key ratings.
Format fixing & type conversionClean and fix longitude/latitude string format issues with extra quotes, convert to numeric type; convert all Yes/No columns to binary 0/1 for modeling.
Outlier removalUse IQR method to remove extreme outliers in average cost and aggregate rating.
Feature engineeringExtract primary cuisine from cuisine text; categorize restaurants into four price levels for research hypothesis validation.
Final outputGenerate one single standardized cleaned dataset without extra city splitting.

## 4. Output File
zomato_clean_all.csv
Fully cleaned, standardized dataset with new derived features, ready for EDA and predictive modeling.

## 5. Environment
Python 3.x
pandas / numpy
Jupyter Notebook

## 6. How to Run
Place the raw CSV in the same folder;
Modify the file path in the code;
Run notebook cells sequentially to reproduce the entire preprocessing and generate the clean file.
