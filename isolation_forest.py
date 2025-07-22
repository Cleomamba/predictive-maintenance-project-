# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 11:14:34 2025

@author: Thandeka Cleo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("motherboard_minmaxed.csv")

# Explore the data
print(data.head())  

#Use all columns 
X = data.values  

# Split the data into training and test sets 
X_train, X_test = train_test_split(X, test_size=0.15, random_state=42)

# Train Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_forest.fit(X_train)

#Get anomaly scores
anomaly_scores_train = iso_forest.decision_function(X_train)
anomaly_scores_test = iso_forest.decision_function(X_test)

#Predict anomalies 
train_preds = iso_forest.predict(X_train)
test_preds = iso_forest.predict(X_test)

# Convert -1 -> 1 for anomalies 
train_anomalies = np.where(train_preds == -1, 1, 0)
test_anomalies = np.where(test_preds == -1, 1, 0)
# Plot the anomaly scores distribution
plt.figure(figsize=(10, 5))
#Training data anomaly scores 
plt.subplot(1, 2, 1)
plt.hist(anomaly_scores_train, bins=30, color='blue', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', label="Decision Boundary (0)")
plt.title("Training Data - Anomaly Scores")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.legend()
#Testing data anomaly scores plot
plt.subplot(1, 2, 2)
plt.hist(anomaly_scores_test, bins=30, color='green', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', label="Decision Boundary (0)")
plt.title("Testing Data - Anomaly Scores")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()

# Plot anomaly scores
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], anomaly_scores_train, c=train_anomalies, cmap='coolwarm', edgecolors='k', s=100, alpha=0.7)
plt.title("Anomaly Scores vs. Feature 1 (Training Data)")
plt.xlabel("Feature 1")
plt.ylabel("Anomaly Score")
plt.colorbar(label="Anomaly (1 = Anomaly, 0 = Normal)")
plt.show()

# Scatter plot for test data
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], anomaly_scores_test, c=test_anomalies, cmap='coolwarm', edgecolors='k', s=100, alpha=0.7)
plt.title("Anomaly Scores vs. Feature 1 (Test Data)")
plt.xlabel("Feature 1")
plt.ylabel("Anomaly Score")
plt.colorbar(label="Anomaly (1 = Anomaly, 0 = Normal)")
plt.show()
