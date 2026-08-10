# Titanic Data Analysis and Machine Learning

## About the Project

This project is about analysing the Titanic dataset and building machine learning models to predict whether a passenger survived or not.

The project contains data analysis, visualisation, classification models and a regression model.

## Dataset

The Titanic dataset is used for this project.

The cleaned dataset is saved as:

titanic.csv

## Part A - Data Analysis

In Part A, I performed:

- Data profiling
- Missing value checking
- Data cleaning
- Outlier detection
- Univariate analysis
- Bivariate analysis
- Correlation analysis
- Data visualisation
- Standardisation of age and fare

## Part B - Machine Learning

In Part B, I performed:

- Train and test split
- Data preprocessing
- One-hot encoding
- Feature scaling
- Logistic Regression
- Decision Tree
- Random Forest
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve and AUC
- Class imbalance comparison
- SMOTE
- Random Forest hyperparameter tuning
- OOB score
- Linear Regression for fare prediction
- MAE, RMSE, R² and Adjusted R²
- Residual plot

## Models Used

### Classification

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Regression

1. Linear Regression

## Final Result

Among the classification models, Logistic Regression performed well on this dataset.

It achieved an accuracy of about 78.34%, precision of 76.79%, recall of 67.19% and F1 score of 71.67%.

Based on these results, I selected Logistic Regression as the final model.

## Saved Model

The final model is saved using Joblib.

The saved model contains the preprocessing steps and the Logistic Regression model together.

## How to Run

Install the required libraries using:

bash
pip install -r requirements.txt