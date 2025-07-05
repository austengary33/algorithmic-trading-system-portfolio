# Machine Learning & Modeling

This directory contains the core machine learning components of the system, covering the entire lifecycle from feature engineering to model training and validation.

## Modules

### `feature_engineering_sample.py`

**Purpose:** This file contains a selection of functions extracted from the main feature engineering notebook. It showcases the translation of complex financial concepts into predictive signals from raw, high-frequency data.

**Key Skills Demonstrated:**
*   **Quantitative Feature Engineering:** Implementation of market microstructure features like order book imbalance, price-distance-weighted liquidity, and order flow toxicity (VPIN).
*   **Advanced Pandas/NumPy:** Use of `pandas` for time-series resampling, rolling window calculations, and efficient data manipulation.
*   **Domain Knowledge:** Translating financial theory into practical, code-based features for a predictive model.

### `model_training.py`

**Purpose:** This script defines and trains a deep learning model to predict market direction based on the engineered features.

**Key Skills Demonstrated:**
*   **Deep Learning with Keras/TensorFlow:** Definition of a sequential model using LSTM layers, along with `Dropout` and `BatchNormalization` for regularization.
*   **End-to-End Training Pipeline:** A complete pipeline that handles data loading, preprocessing (scaling, normalization), splitting into training/validation sets, and model fitting.
*   **ML Best Practices:** Implementation of essential callbacks like `ModelCheckpoint` for saving the best models, `EarlyStopping` to prevent overfitting, and custom callbacks for logging and metrics.
*   **Handling Class Imbalance:** Use of `class_weight` to manage imbalanced datasets, a common challenge in financial modeling.

### `backtesting_sample.ipynb`

**Purpose:** This Jupyter Notebook provides a framework for rigorously backtesting the trained model's performance on out-of-sample data.

**Key Skills Demonstrated:**
*   **Strategy Validation:** Simulating trade execution to evaluate the profitability and risk profile of the model's signals.
*   **Performance Analysis:** Calculating and visualizing key metrics like PnL curves and accounting for realistic trading costs (bid-ask spread).
*   **Data Visualization:** Using `matplotlib` and `pandas` to plot results and gain insights into the strategy's behavior over time.
