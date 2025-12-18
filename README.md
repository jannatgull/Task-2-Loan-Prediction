# Task-2-Loan-Prediction
Loan Default Prediction using Logistic Regression and Decision Tree models with data visualization and evaluation metrics.

## 📌 Objective
The objective of this project is to predict whether a customer will default on a loan or not using machine learning classification models.

## 📂 Dataset
The dataset (`loanpred.csv`) contains information about customers including:
- Index
- Employment Status
- Bank Balance
- Annual Salary
- Loan Default Status (Target Variable)

There is no missing data in the dataset.

## 🛠️ Approach
1. Loaded the dataset using Pandas.
2. Performed basic data exploration:
   - Dataset shape
   - First few rows
   - Column names
3. Visualized data using Seaborn and Matplotlib:
   - Histogram for Annual Salary distribution
   - Box plot for Bank Balance distribution
   - Count plot for Employment Status
4. Selected features and target variable.
5. Split the data into training (80%) and testing (20%) sets.
6. Trained two machine learning models:
   - Logistic Regression
   - Decision Tree Classifier
7. Evaluated model performance using:
   - Accuracy Score
   - Confusion Matrix

## 📊 Models Used
- Logistic Regression
- Decision Tree Classifier

## 📈 Results & Insights
- Logistic Regression provided good accuracy in predicting loan defaults.
- Decision Tree also performed well but showed different classification behavior.
- Salary, bank balance, and employment status were important factors in prediction.

## 🧰 Libraries Used
- pandas
- seaborn
- matplotlib
- scikit-learn

## ✅ Conclusion
This project demonstrates how machine learning models can be used to predict loan default risk effectively using customer financial data.
