"""
feature_engineering_sample.py

This module contains a selection of  feature engineering functions designed
to extract predictive signals from high-frequency order book data. These functions
implement concepts from market microstructure literature
to create features for a machine learning model.

The functions are designed to be applied to a pandas DataFrame of tick data
for a single financial instrument.
"""

import pandas as pd
import numpy as np

# A small constant to prevent division by zero in calculations.
EPSILON = 1e-10

def calculate_weighted_order_book_imbalance(df: pd.DataFrame, depth: int = 5) -> pd.Series:
    """
    Calculates the order book imbalance, weighted by price distance.

    This feature captures not just the volume of bids vs. asks, but gives more
    weight to orders closer to the mid-price, providing a more sensitive
    measure of immediate price pressure.

    Args:
        df (pd.DataFrame): DataFrame containing order book tick data with columns
                           like 'bid_price_1', 'bid_size_1', 'ask_price_1', etc.
        depth (int): The number of order book levels to consider.

    Returns:
        pd.Series: A series representing the weighted imbalance for each tick.
    """
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    
    weighted_bid_volume = pd.Series(0.0, index=df.index)
    weighted_ask_volume = pd.Series(0.0, index=df.index)

    for i in range(1, depth + 1):
        bid_price = df[f'bid_price_{i}']
        bid_size = df[f'bid_size_{i}']
        ask_price = df[f'ask_price_{i}']
        ask_size = df[f'ask_size_{i}']

        # Weight is inversely proportional to distance from mid-price.
        # Closer orders have more weight.
        bid_weight = 1 / (np.abs(mid_price - bid_price) + EPSILON)
        ask_weight = 1 / (np.abs(ask_price - mid_price) + EPSILON)

        weighted_bid_volume += bid_size * bid_weight
        weighted_ask_volume += ask_size * ask_weight

    imbalance = (weighted_bid_volume - weighted_ask_volume) / (weighted_bid_volume + weighted_ask_volume + EPSILON)
    
    return imbalance.clip(-1, 1)


def calculate_vpin(df: pd.DataFrame, volume_bucket_size: int = 100000) -> pd.Series:
    """
    Calculates the Volume-Synchronized Probability of Informed Trading (VPIN).

    VPIN is a sophisticated measure of order flow toxicity. High VPIN values
    suggest a higher probability of informed traders in the market, which can
    precede periods of high volatility. This implementation uses a simplified
    approach based on volume buckets.

    Args:
        df (pd.DataFrame): DataFrame with tick data. Must contain 'bid_size_1'
                           and 'ask_size_1' columns.
        volume_bucket_size (int): The total trade volume that defines one "bucket"
                                  or time bar.

    Returns:
        pd.Series: A series representing the VPIN value at the end of each volume bucket.
    """
    # Calculate trade direction based on change in best bid/ask sizes
    bid_vol_change = df['bid_size_1'].diff().fillna(0)
    ask_vol_change = df['ask_size_1'].diff().fillna(0)
    
    # Simple heuristic for order flow: increase in bid size = buy, increase in ask size = sell
    buy_volume = bid_vol_change.clip(lower=0)
    sell_volume = ask_vol_change.clip(lower=0)
    
    total_volume = buy_volume + sell_volume
    
    # Create volume buckets
    cumulative_volume = total_volume.cumsum()
    bucket_indices = (cumulative_volume // volume_bucket_size).astype(int)
    
    # Group by volume bucket and calculate order flow imbalance
    bucket_df = pd.DataFrame({
        'bucket': bucket_indices,
        'buy_vol': buy_volume,
        'sell_vol': sell_volume
    })
    
    bucket_imbalance = bucket_df.groupby('bucket').apply(
        lambda x: (x['buy_vol'] - x['sell_vol']).abs().sum()
    )
    
    total_bucket_volume = bucket_df.groupby('bucket').apply(
        lambda x: (x['buy_vol'] + x['sell_vol']).sum()
    )
    
    # VPIN is the sum of absolute imbalances divided by total volume over N buckets
    vpin = bucket_imbalance.rolling(window=50, min_periods=10).sum() / \
           (50 * volume_bucket_size)
           
    # Map results back to the original DataFrame index
    vpin_series = vpin.reindex(bucket_indices).ffill().fillna(0)
    vpin_series.index = df.index
    
    return vpin_series


if __name__ == '__main__':
    # This block demonstrates how to use the functions.
    # It creates a sample DataFrame and applies the feature engineering.
    
    print("--- Demonstrating Feature Engineering Functions ---")

    # 1. Create a sample DataFrame mimicking real orderbook data
    data = {
        'bid_price_1': [1.1000, 1.1001, 1.1000, 1.1002],
        'bid_size_1':  [100000, 150000, 120000, 200000],
        'ask_price_1': [1.1002, 1.1003, 1.1002, 1.1004],
        'ask_size_1':  [120000, 130000, 110000, 180000],
        'bid_price_2': [1.0999, 1.1000, 1.0999, 1.1001],
        'bid_size_2':  [200000, 250000, 220000, 300000],
        'ask_price_2': [1.1003, 1.1004, 1.1003, 1.1005],
        'ask_size_2':  [220000, 230000, 210000, 280000],
    }
    # Add levels 3-5 for the imbalance calculation
    for i in range(3, 6):
        data[f'bid_price_{i}'] = data['bid_price_2'] - (i-2) * 0.0001
        data[f'bid_size_{i}'] = data['bid_size_2']
        data[f'ask_price_{i}'] = data['ask_price_2'] + (i-2) * 0.0001
        data[f'ask_size_{i}'] = data['ask_size_2']

    sample_df = pd.DataFrame(data)
    print("\nSample Input DataFrame:")
    print(sample_df.head())

    # 2. Apply the feature engineering functions
    sample_df['w_imbalance'] = calculate_weighted_order_book_imbalance(sample_df)
    sample_df['vpin'] = calculate_vpin(sample_df)

    print("\nDataFrame with Engineered Features:")
    print(sample_df[['w_imbalance', 'vpin']].head())
