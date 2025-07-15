# Austen Gary's Portfolio

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

## Core Components & Key Features

### 1. Data Engineering & Pipeline

A fault-tolerant data pipeline streams high-frequency order book data, capturing over 54 million records (~2.1B data points) monthly.

- **Real-Time Data Collector** (datafeed_orderbook.py): A multi-process Python service that connects to the Interactive Brokers API, subscribes to Level 2 order book data for multiple currency pairs, and persists tick data to disk in one-minute batches. Features robust error handling and automated reconnection logic.

- **Health & Integrity Monitor** (datafeed_monitor.py): A companion service that runs every minute to validate the integrity of the collected data. It checks for file existence, completeness, and timeliness, publishing health status and volume metrics to dedicated GCP Pub/Sub topics.

- **Offline Batch Processing** (agg_tick_data_batch_job.py): A scalable batch job orchestrator that systematically processes months of raw tick data through the feature engineering engine to create datasets for model training.

### 2. Feature Engineering & Modeling

The system includes a sophisticated pipeline to transform raw, noisy tick data into powerful predictive features.

- **Feature Engineering Engine** (agg_tick_data.ipynb): A comprehensive Jupyter Notebook that serves as the core transformation engine. It programmatically engineers thousands of features from raw order book data, including:
  - **Market Microstructure**: Order book imbalance, depth change, volume-weighted average price (VWAP), and order flow toxicity (VPIN).
  - **Time-Series Features**: Rolling statistical measures, EMA/SMA, and cyclical time-based encodings (sin/cos).
  - **Intra-Minute Dynamics**: Tick activity distribution, liquidity evolution in 10-second windows, and price-distance-weighted liquidity.
 
- **Model Training** (model_training.py): A complete training pipeline built with Keras/TensorFlow. It defines a deep learning (LSTM) model to predict multi-class market direction, handles data preprocessing (scaling, normalization), and uses callbacks for checkpointing, early stopping, and logging.
- **Strategy Validation** (backtest.ipynb): A rigorous backtesting framework to simulate strategy performance on out-of-sample data. It accounts for trading costs (bid-ask spread) and provides detailed PnL analysis and visualization.

### 3. Deployment & Operations (DevOps)
The entire system is designed for production and deployed as a set of automated, resilient services.

- **Automated & Idempotent Deployment** (deploy_*.sh): Deployed and updated services using idempotent Bash scripts, which check the system's state before acting, ensuring reliable and repeatable deployments.
- **Resilient Service Management** (*.service): Engineered all Python applications to run as background systemd services on Linux, configured with Restart=on-failure policies to ensure high availability and automatic recovery from application or system failures.
- **Centralized Logging**: Service logs are directed to journald for easy monitoring and debugging, with critical alerts and metrics forwarded to Google Cloud Logging.

---

### Technologies Used

- **Languages**: Python, Bash
- **Machine Learning**: Keras, TensorFlow, Scikit-learn, Pandas, NumPy
- **Cloud & Infrastructure**: Google Cloud Platform (GCP), Google Cloud Pub/Sub, Google Cloud Logging
- **System & DevOps**: Linux (Ubuntu), systemd, joblib, asyncio
- **Data Source**: Interactive Brokers API (via ib_insync)
