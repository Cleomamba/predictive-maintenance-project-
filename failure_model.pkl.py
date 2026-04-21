# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 12:26:39 2025

@author: Thandeka Cleo
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib

#Load the dataset
data = pd.read_csv("motherboard_minmaxed_with_anomalies.csv")

# Split into features and target
X = data.drop(columns=["isolation_forest_anomaly"])
y = data["isolation_forest_anomaly"]

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#  Train the XGBoost model
xgboost_model = XGBClassifier(random_state=42)
xgboost_model.fit(X_train, y_train)

#Evaluate the model
y_pred = xgboost_model.predict(X_test)
y_proba = xgboost_model.predict_proba(X_test)[:, 1]

#Evaluate with precision, recall, F1-score, and ROC-AUC
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

#Save the trained XGBoost model
joblib.dump(xgboost_model, "failured_pred_model.pkl")
print("Model saved as 'failured_pred_model.pkl'")
