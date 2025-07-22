# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 11:54:31 2025

@author: Thandeka Cleo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("motherboard_minmaxed_with_anomalies.csv")

#Define features and label
X = data.drop(columns=["isolation_forest_anomaly"])  
y = data["isolation_forest_anomaly"]  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split data: 70% train, 15% test, 15% validation
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

#Train XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

#Evaluate on test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

#Metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n Classification Report:")
print(classification_report(y_test, y_pred))
print(f" Precision: {precision:.4f}")
print(f" Recall: {recall:.4f}")
print(f" F1 Score: {f1:.4f}")
print(f" ROC-AUC: {roc_auc:.4f}")

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix:")
print(cm)

#ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost Failure Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
