# algorithmic-trading-system-portfolio

## End-to-End Algorithmic Trading System

This repository contains key modules from a full-cycle, high-frequency algorithmic trading system I architected and built from the ground up. The system is designed to be a robust, production-grade application that handles everything from real-time data ingestion and feature engineering to model training, backtesting, and automated trade execution.

The project demonstrates a comprehensive skill set in Python, data engineering, machine learning (with Keras/TensorFlow), and DevOps on a cloud-native stack.

## System Architecture

The system is designed as a decoupled, event-driven pipeline orchestrated via Google Cloud Pub/Sub. This architecture ensures scalability, resilience, and maintainability.

```
[ Interactive Brokers API ]
         |
         | (Real-time Order Book Data)
         v
[ 1. Data Collection & Monitoring (Linux VM on GCP) ]
   |
   |--> [ Python Data Collector ] -> Stores raw tick data (.joblib)
   |
   '--> [ Python Health Monitor ] -> Validates data, publishes status
         |
         | (Pub/Sub Topic: "data-files-created")
         v
[ 2. ML Inference & Order Management (Linux VM on GCP) ]
   |
   |--> [ Python Prediction Service ] -> Consumes data, loads model, generates signals
         |
         | (Pub/Sub Topic: "order-management")
         v
   '--> [ Python Order Management Service ] -> Consumes signals, executes trades
```
