# Car-Insurance-Fraud-Detection
📌 Project Overview
This project aims to detect fraudulent car insurance claims using Machine Learning. It processes a dataset of 30,000 records containing both numerical and categorical features to identify patterns that distinguish legitimate claims from fraudulent ones.

---> Key Features & Workflow
Data Preprocessing: Handled missing values and removed outliers using the Z-score method.

Feature Engineering: Treated data skewness using Yeo-Johnson Transformation to achieve a Normal Distribution.

Class Balancing: Applied SMOTE (Synthetic Minority Over-sampling Technique) to handle the imbalance in fraud cases.

Feature Scaling: Used StandardScaler to normalize numerical inputs for better model performance.

---> Model Performance
Multiple models were compared to find the best fit for this classification task:

Logistic Regression

Decision Tree

Random Forest (AUC: 0.72)

XGBoost (Final Model): Achieved the highest AUC score of 0.75.

---> Deployment Files
The following files are generated for frontend integration:

xgb_model.pkl: The trained final XGBoost model.

scaler.pkl: The fitted scaler to transform real-time user inputs.
