# Body Fat Percentage Prediction

## Project Overview
This project focuses on predicting **body fat percentage** using machine learning techniques based on physical and physiological measurements such as age, weight, height, and body circumferences.

The goal is to build a **regression model** that can accurately estimate body fat percentage, which is an important indicator of overall health and fitness.

---

## Problem Statement
Traditional methods of measuring body fat (like underwater weighing) are expensive and inconvenient.  
This project uses **data-driven machine learning models** to predict body fat percentage using easily measurable attributes.

---

## Dataset
- The dataset contains body measurements of individuals.
- Target variable: **Body Fat Percentage**
- Features include:
  - Age
  - Weight
  - Height
  - Thigh, Chest, Abdomen, Hip circumferences
  - BMI-related measurements

> The dataset is cleaned and preprocessed before model training.

---

## Exploratory Data Analysis (EDA)
The following steps were performed:
- Checking missing values
- Statistical summary of features
- Correlation analysis
- Distribution plots
- Outlier detection

EDA helped in understanding:
- Which features strongly influence body fat
- Data skewness and variability
- Feature relationships

---

## Machine Learning Models Used
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

---

## Model Evaluation Metrics
Models were evaluated using:
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

The best-performing model was selected based on **overall accuracy and generalization ability**.

---

## Tech Stack
- **Programming Language:** Python
- **Libraries:**
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn
- **Environment:** Jupyter Notebook / VS Code

---



