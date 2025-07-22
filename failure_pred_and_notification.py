# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 13:16:58 2025

@author: Thandeka Cleo
"""
import psutil
import pandas as pd
import joblib
import time
import random
import datetime
import tkinter as tk
from tkinter import messagebox


MODEL_NAME = 1

# Load models
iso_forest = joblib.load("iso_model.pkl")
xgb_model = joblib.load("failured_pred_model.pkl")
scaler = joblib.load("scalere.pkl")

# Track time since startup
startup_time = time.time()
last_notification_time = startup_time
first_boot_flag = True
notification_interval = 30  

# Collect custom metrics 
def collect_custom_metrics():
    metrics = {
            "model_name":MODEL_NAME,
            "CPUUsage": psutil.cpu_percent(interval=1),
             "RAMUsage": psutil.virtual_memory().percent,
        "Temperature": random.uniform(35, 90), 
        "Voltage": 1,  # Placeholder
        "DiskUsage": psutil.disk_usage('/').percent,
        "FanSpeed": random.uniform(1000, 5000),  
        "ProblemDetected": 0, #placeholder
        
    }
    return metrics

# Analyze metrics with anomaly detection + failure prediction
def analyze_metrics(metrics):
    df = pd.DataFrame([metrics])

    anomaly_flag = iso_forest.predict(df)[0]
    df["anomaly_detected"] = 1 if anomaly_flag == -1 else 0

    scaled = scaler.transform(df.drop(columns=["anomaly_detected"]))
    failure_proba = xgb_model.predict_proba(scaled)[0][1]

    return df, failure_proba

# Log to file
def log_notification(msg):
    with open("system_alerts.log", "a") as log_file:
        log_file.write(msg + "\n" + ("-" * 60) + "\n")

# Show alert notification using tkinter (forced to front)
def send_notification(metrics, proba, boot=False):
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    header = "Boot" if boot else "System Alert"

    # Define 6 key metrics
    important_metrics = ['Temperature', 'CPUUsage', 'RAMUsage', 'DiskUsage', 'FanSpeed']
    top_metrics = {k: round(metrics[k], 2) for k in important_metrics}

    # Determine system status and likely cause
    status = "NORMAL"
    likely_cause = "None"
    if proba > 0.6:
        status = "WARNING: Potential failure"
        sorted_metrics = sorted(top_metrics.items(), key=lambda item: item[1], reverse=True)
        likely_cause = sorted_metrics[0][0]

    # Build notification message
    msg = f"{header} @ {time_now}\n"
    msg += f"Risk Score: {proba:.4f}\n"
    msg += f"System Status: {status}\n"
    msg += f"Likely Cause: {likely_cause}\n\n"

    for key, val in top_metrics.items():
        msg += f"{key}: {val}\n"

    # Show using tkinter and force it to front
    root = tk.Tk()
    root.withdraw()
    root.after(0, lambda: root.attributes('-topmost', True))
    messagebox.showinfo(header, msg, parent=root)
    root.destroy()

    # Log the message
    log_notification(msg)

#Main monitoring loop
while True:
    metrics = collect_custom_metrics()
    analyzed_df, proba = analyze_metrics(metrics)
    now = time.time()

    

    # Notify immediately if high risk and anomaly detected
    if analyzed_df["anomaly_detected"].iloc[0] == 1 and proba > 0.6:
        send_notification(metrics, proba)

    time.sleep(30)

