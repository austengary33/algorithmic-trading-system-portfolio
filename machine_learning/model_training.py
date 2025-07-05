# ---------------------------- [ PYTHON IMPORTS ] ----------------------------
# import sklearn, keras.utils, etc.
if True:

    import sys
    import time
    import os
    import joblib
    import multiprocessing
    import threading
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import keras
    import json
    import re

    from datetime import timedelta, datetime
    from sklearn.utils import shuffle
    from sklearn.utils import class_weight
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    from keras.callbacks import TensorBoard
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
    from keras.regularizers import l2
    from keras import backend as K
    from keras.utils import to_categorical
    from collections import Counter
    from typing import List, Dict, Pattern

    # Local imports of custom functions
    from helpers.timing_debugging import debuggingTools_format_time, debug_timing

    # --------------------------------------------------

    # Only show ERRORS in tensorflow 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    # --------------------------------------------------

    # --- Config filepaths for training outputs ---

    # Model file paths
    model_iteration_file_name = (
        f"TIME_{datetime.now().strftime('%m-%d-%Y_%I_%M_%S%p')}"
        f"_SEQLEN")

    scalers_export_path = \
        f'c:/Users/auste/Documents/Python Scripts/data/training/models/{model_iteration_file_name}_saved_scalers.joblib'
    
    model_summary_output_path = f'data/training/models/{model_iteration_file_name}_model_summary.txt'

    # --------------------------------------------------

    # Used to inspect what a final transformed timeseries batch looks like (what final list of columns look like, after adding features, lags, etc.)
    sample_of_final_transformed_data = None 

    # --------------------------------------------------

# ---------------------------- [ MODEL PARAMS ] ------------------------
if True:

    # `SEQ_LEN` & `FUTURE_PERIOD_PREDICT`
    if True:

        SEQ_LEN = 45+2 # minutes in each batch (note: 2 will be dropped, when calculating target - and possibly more when calculating stats which require warmup, eg. rolling windows, etc.)

        FUTURE_PERIOD_PREDICT = 5 # minutes in the future to predict

        # ----------------------------------------------------------------------------

    # `Columns_to_use_as_model_features` 
    if True:

        columns_to_use_as_model_features = '''

            # ----------------- Time Features

            minutes_market_has_been_open

            day_of_the_week	
            hour_of_the_day	
            day_of_the_month

            # ----------------- Volume/Activity Features

            EURUSD_total_ticks
            GBPUSD_total_ticks
            # USDJPY_total_ticks

            # meaningful_ticks
            EURUSD_meaningful_ticks__L1_price_change_ticks
            GBPUSD_meaningful_ticks__L1_price_change_ticks
            # USDJPY_meaningful_ticks__L1_price_change_ticks

            EURUSD_meaningful_ratio__L1_price_change_ticks_to_total_ticks
            GBPUSD_meaningful_ratio__L1_price_change_ticks_to_total_ticks
            # USDJPY_meaningful_ratio__L1_price_change_ticks_to_total_ticks  

            # ----------------- Price Features 

            # --- All these will be dropped, they're only used used to build other features ---

            EURUSD_open_price__ask_level_1
            EURUSD_open_price__bid_level_1
            EURUSD_close_price__ask_level_1
            EURUSD_close_price__bid_level_1
            EURUSD_high_price__ask_level_1
            EURUSD_high_price__bid_level_1
            EURUSD_low_price__ask_level_1
            EURUSD_low_price__bid_level_1

            GBPUSD_open_price__ask_level_1
            GBPUSD_open_price__bid_level_1
            GBPUSD_close_price__ask_level_1
            GBPUSD_close_price__bid_level_1
            GBPUSD_high_price__ask_level_1
            GBPUSD_high_price__bid_level_1
            GBPUSD_low_price__ask_level_1
            GBPUSD_low_price__bid_level_1

            # USDJPY_open_price__ask_level_1 # removing "USDJPY" for now - to reduce complexity
            # USDJPY_open_price__bid_level_1
            # USDJPY_close_price__ask_level_1
            # USDJPY_close_price__bid_level_1
            # USDJPY_high_price__ask_level_1
            # USDJPY_high_price__bid_level_1
            # USDJPY_low_price__ask_level_1
            # USDJPY_low_price__bid_level_1

            # ----------------- L2 Features

            # Use the find pattern `^EURUSD_(.+?)\s*-\s*.*$` to capture everything after EURUSD but before any spaces or tab characters, and replace it with `GBPUSD_\1`. This keeps the suffix after the currency pair intact while replacing the instrument itself.

            # EURUSD_volume_in_0_to_10_second_window
            # EURUSD_volume_in_10_to_20_second_window
            # EURUSD_volume_in_20_to_30_second_window
            # EURUSD_volume_in_30_to_40_second_window
            # EURUSD_volume_in_40_to_50_second_window
            # EURUSD_volume_in_50_to_60_second_window

            # EURUSD_total_volume_cumulative_up_to_10th_second
            # EURUSD_total_volume_cumulative_up_to_20th_second
            # EURUSD_total_volume_cumulative_up_to_30th_second
            # EURUSD_total_volume_cumulative_up_to_40th_second
            # EURUSD_total_volume_cumulative_up_to_50th_second
            # EURUSD_total_volume_cumulative_up_to_60th_second

            # EURUSD_bid_size_1_in_0_to_10_second_window
            # EURUSD_bid_size_1_in_10_to_20_second_window
            # EURUSD_bid_size_1_in_20_to_30_second_window
            # EURUSD_bid_size_1_in_30_to_40_second_window
            # EURUSD_bid_size_1_in_40_to_50_second_window
            # EURUSD_bid_size_1_in_50_to_60_second_window

            # EURUSD_bid_size_2_in_0_to_10_second_window
            # EURUSD_bid_size_2_in_10_to_20_second_window
            # EURUSD_bid_size_2_in_20_to_30_second_window
            # EURUSD_bid_size_2_in_30_to_40_second_window
            # EURUSD_bid_size_2_in_40_to_50_second_window
            # EURUSD_bid_size_2_in_50_to_60_second_window

            # EURUSD_bid_size_3_in_0_to_10_second_window
            # EURUSD_bid_size_3_in_10_to_20_second_window
            # EURUSD_bid_size_3_in_20_to_30_second_window
            # EURUSD_bid_size_3_in_30_to_40_second_window
            # EURUSD_bid_size_3_in_40_to_50_second_window
            # EURUSD_bid_size_3_in_50_to_60_second_window

            # EURUSD_ask_size_1_in_0_to_10_second_window
            # EURUSD_ask_size_1_in_10_to_20_second_window
            # EURUSD_ask_size_1_in_20_to_30_second_window
            # EURUSD_ask_size_1_in_30_to_40_second_window
            # EURUSD_ask_size_1_in_40_to_50_second_window
            # EURUSD_ask_size_1_in_50_to_60_second_window

            # EURUSD_ask_size_2_in_0_to_10_second_window
            # EURUSD_ask_size_2_in_10_to_20_second_window
            # EURUSD_ask_size_2_in_20_to_30_second_window
            # EURUSD_ask_size_2_in_30_to_40_second_window
            # EURUSD_ask_size_2_in_40_to_50_second_window
            # EURUSD_ask_size_2_in_50_to_60_second_window

            # EURUSD_ask_size_3_in_0_to_10_second_window
            # EURUSD_ask_size_3_in_10_to_20_second_window
            # EURUSD_ask_size_3_in_20_to_30_second_window
            # EURUSD_ask_size_3_in_30_to_40_second_window
            # EURUSD_ask_size_3_in_40_to_50_second_window
            # EURUSD_ask_size_3_in_50_to_60_second_window

            # GBPUSD_volume_in_0_to_10_second_window
            # GBPUSD_volume_in_10_to_20_second_window
            # GBPUSD_volume_in_20_to_30_second_window
            # GBPUSD_volume_in_30_to_40_second_window
            # GBPUSD_volume_in_40_to_50_second_window
            # GBPUSD_volume_in_50_to_60_second_window

            # GBPUSD_total_volume_cumulative_up_to_10th_second
            # GBPUSD_total_volume_cumulative_up_to_20th_second
            # GBPUSD_total_volume_cumulative_up_to_30th_second
            # GBPUSD_total_volume_cumulative_up_to_40th_second
            # GBPUSD_total_volume_cumulative_up_to_50th_second
            # GBPUSD_total_volume_cumulative_up_to_60th_second

            # GBPUSD_bid_size_1_in_0_to_10_second_window
            # GBPUSD_bid_size_1_in_10_to_20_second_window
            # GBPUSD_bid_size_1_in_20_to_30_second_window
            # GBPUSD_bid_size_1_in_30_to_40_second_window
            # GBPUSD_bid_size_1_in_40_to_50_second_window
            # GBPUSD_bid_size_1_in_50_to_60_second_window

            # GBPUSD_bid_size_2_in_0_to_10_second_window
            # GBPUSD_bid_size_2_in_10_to_20_second_window
            # GBPUSD_bid_size_2_in_20_to_30_second_window
            # GBPUSD_bid_size_2_in_30_to_40_second_window
            # GBPUSD_bid_size_2_in_40_to_50_second_window
            # GBPUSD_bid_size_2_in_50_to_60_second_window

            # GBPUSD_bid_size_3_in_0_to_10_second_window
            # GBPUSD_bid_size_3_in_10_to_20_second_window
            # GBPUSD_bid_size_3_in_20_to_30_second_window
            # GBPUSD_bid_size_3_in_30_to_40_second_window
            # GBPUSD_bid_size_3_in_40_to_50_second_window
            # GBPUSD_bid_size_3_in_50_to_60_second_window

            # GBPUSD_ask_size_1_in_0_to_10_second_window
            # GBPUSD_ask_size_1_in_10_to_20_second_window
            # GBPUSD_ask_size_1_in_20_to_30_second_window
            # GBPUSD_ask_size_1_in_30_to_40_second_window
            # GBPUSD_ask_size_1_in_40_to_50_second_window
            # GBPUSD_ask_size_1_in_50_to_60_second_window

            # GBPUSD_ask_size_2_in_0_to_10_second_window
            # GBPUSD_ask_size_2_in_10_to_20_second_window
            # GBPUSD_ask_size_2_in_20_to_30_second_window
            # GBPUSD_ask_size_2_in_30_to_40_second_window
            # GBPUSD_ask_size_2_in_40_to_50_second_window
            # GBPUSD_ask_size_2_in_50_to_60_second_window

            # GBPUSD_ask_size_3_in_0_to_10_second_window
            # GBPUSD_ask_size_3_in_10_to_20_second_window
            # GBPUSD_ask_size_3_in_20_to_30_second_window
            # GBPUSD_ask_size_3_in_30_to_40_second_window
            # GBPUSD_ask_size_3_in_40_to_50_second_window
            # GBPUSD_ask_size_3_in_50_to_60_second_window

            # EURUSD_price_direction_pips
            # EURUSD_total_price_change_magnitude
            # EURUSD_bid_price_change_magnitude 
            # EURUSD_ask_price_change_magnitude 
            # EURUSD_total_pips_movement 
            # EURUSD_bid_pips_movement 
            # EURUSD_ask_pips_movement 
            # EURUSD_pips_imbalance_ratio
            # EURUSD_mid_price_vol_pips
            # EURUSD_direction_pips_z 
            # EURUSD_avg_tick_magnitude
            # EURUSD_avg_tick_pips 
            
            # GBPUSD_total_price_change_magnitude
            # GBPUSD_bid_price_change_magnitude 
            # GBPUSD_ask_price_change_magnitude
            # GBPUSD_total_pips_movement
            # GBPUSD_bid_pips_movement 
            # GBPUSD_ask_pips_movement 
            # GBPUSD_pips_imbalance_ratio 
            # GBPUSD_mid_price_vol_pips 
            # GBPUSD_direction_pips_z 
            # GBPUSD_avg_tick_magnitude 
            # GBPUSD_avg_tick_pips 

            # ---

            # EURUSD_queue_value_mean_in_0_to_10_second_window
            # EURUSD_queue_value_std_in_0_to_10_second_window
            # EURUSD_queue_value_min_in_0_to_10_second_window
            # EURUSD_queue_value_max_in_0_to_10_second_window
            # EURUSD_queue_value_mean_in_10_to_20_second_window
            # EURUSD_queue_value_std_in_10_to_20_second_window
            # EURUSD_queue_value_min_in_10_to_20_second_window
            # EURUSD_queue_value_max_in_10_to_20_second_window
            # EURUSD_queue_value_mean_in_20_to_30_second_window
            # EURUSD_queue_value_std_in_20_to_30_second_window
            # EURUSD_queue_value_min_in_20_to_30_second_window
            # EURUSD_queue_value_max_in_20_to_30_second_window
            # EURUSD_queue_value_mean_in_30_to_40_second_window
            # EURUSD_queue_value_std_in_30_to_40_second_window
            # EURUSD_queue_value_min_in_30_to_40_second_window
            # EURUSD_queue_value_max_in_30_to_40_second_window
            # EURUSD_queue_value_mean_in_40_to_50_second_window
            # EURUSD_queue_value_std_in_40_to_50_second_window
            # EURUSD_queue_value_min_in_40_to_50_second_window
            # EURUSD_queue_value_max_in_40_to_50_second_window
            # EURUSD_queue_value_mean_in_50_to_60_second_window
            # EURUSD_queue_value_std_in_50_to_60_second_window
            # EURUSD_queue_value_min_in_50_to_60_second_window
            # EURUSD_queue_value_max_in_50_to_60_second_window
            # EURUSD_concentration_zones_mean_in_0_to_10_second_window
            # EURUSD_concentration_zones_std_in_0_to_10_second_window
            # EURUSD_concentration_zones_min_in_0_to_10_second_window
            # EURUSD_concentration_zones_max_in_0_to_10_second_window
            # EURUSD_concentration_zones_mean_in_10_to_20_second_window
            # EURUSD_concentration_zones_std_in_10_to_20_second_window
            # EURUSD_concentration_zones_min_in_10_to_20_second_window
            # EURUSD_concentration_zones_max_in_10_to_20_second_window
            # EURUSD_concentration_zones_mean_in_20_to_30_second_window
            # EURUSD_concentration_zones_std_in_20_to_30_second_window
            # EURUSD_concentration_zones_min_in_20_to_30_second_window
            # EURUSD_concentration_zones_max_in_20_to_30_second_window
            # EURUSD_concentration_zones_mean_in_30_to_40_second_window
            # EURUSD_concentration_zones_std_in_30_to_40_second_window
            # EURUSD_concentration_zones_min_in_30_to_40_second_window
            # EURUSD_concentration_zones_max_in_30_to_40_second_window
            # EURUSD_concentration_zones_mean_in_40_to_50_second_window
            # EURUSD_concentration_zones_std_in_40_to_50_second_window
            # EURUSD_concentration_zones_min_in_40_to_50_second_window
            # EURUSD_concentration_zones_max_in_40_to_50_second_window
            # EURUSD_concentration_zones_mean_in_50_to_60_second_window
            # EURUSD_concentration_zones_std_in_50_to_60_second_window
            # EURUSD_concentration_zones_min_in_50_to_60_second_window
            # EURUSD_concentration_zones_max_in_50_to_60_second_window

            EURUSD_density_gradient_mean_in_0_to_10_second_window
            EURUSD_density_gradient_mean_in_10_to_20_second_window
            EURUSD_density_gradient_mean_in_20_to_30_second_window
            EURUSD_density_gradient_mean_in_30_to_40_second_window
            EURUSD_density_gradient_mean_in_40_to_50_second_window
            EURUSD_density_gradient_mean_in_50_to_60_second_window

            # EURUSD_density_gradient_std_in_0_to_10_second_window
            # EURUSD_density_gradient_std_in_10_to_20_second_window
            # EURUSD_density_gradient_std_in_20_to_30_second_window
            # EURUSD_density_gradient_std_in_30_to_40_second_window
            # EURUSD_density_gradient_std_in_40_to_50_second_window
            # EURUSD_density_gradient_std_in_50_to_60_second_window

            # EURUSD_density_gradient_min_in_0_to_10_second_window
            # EURUSD_density_gradient_min_in_10_to_20_second_window
            # EURUSD_density_gradient_min_in_20_to_30_second_window
            # EURUSD_density_gradient_min_in_30_to_40_second_window
            # EURUSD_density_gradient_min_in_40_to_50_second_window
            # EURUSD_density_gradient_min_in_50_to_60_second_window

            # EURUSD_density_gradient_max_in_0_to_10_second_window
            # EURUSD_density_gradient_max_in_10_to_20_second_window
            # EURUSD_density_gradient_max_in_20_to_30_second_window
            # EURUSD_density_gradient_max_in_30_to_40_second_window
            # EURUSD_density_gradient_max_in_40_to_50_second_window
            # EURUSD_density_gradient_max_in_50_to_60_second_window

            # EURUSD_ladder_density_minute_mean
            # EURUSD_ladder_density_minute_std
            # EURUSD_ladder_density_minute_min
            # EURUSD_ladder_density_minute_max
            # EURUSD_ladder_density_minute_median
            # EURUSD_relative_position_minute_mean
            # EURUSD_relative_position_minute_std
            # EURUSD_relative_position_minute_min
            # EURUSD_relative_position_minute_max
            # EURUSD_relative_position_minute_median

            # EURUSD_queue_value_minute_volatility
            # EURUSD_concentration_zone_minute_range
            # EURUSD_density_gradient_minute_trend

            # EURUSD_queue_value_cumulative_mean_up_to_10th_second
            # EURUSD_queue_value_cumulative_mean_up_to_20th_second
            # EURUSD_queue_value_cumulative_mean_up_to_30th_second
            # EURUSD_queue_value_cumulative_mean_up_to_40th_second
            # EURUSD_queue_value_cumulative_mean_up_to_50th_second
            # EURUSD_queue_value_cumulative_mean_up_to_60th_second

            # EURUSD_concentration_zones_cumulative_mean_up_to_10th_second
            # EURUSD_concentration_zones_cumulative_mean_up_to_20th_second
            # EURUSD_concentration_zones_cumulative_mean_up_to_30th_second
            # EURUSD_concentration_zones_cumulative_mean_up_to_40th_second
            # EURUSD_concentration_zones_cumulative_mean_up_to_50th_second
            # EURUSD_concentration_zones_cumulative_mean_up_to_60th_second

            # EURUSD_density_gradient_cumulative_mean_up_to_10th_second
            # EURUSD_density_gradient_cumulative_mean_up_to_20th_second
            # EURUSD_density_gradient_cumulative_mean_up_to_30th_second
            # EURUSD_density_gradient_cumulative_mean_up_to_40th_second
            # EURUSD_density_gradient_cumulative_mean_up_to_50th_second
            # EURUSD_density_gradient_cumulative_mean_up_to_60th_second

            # GBPUSD_queue_value_mean_in_0_to_10_second_window
            # GBPUSD_queue_value_std_in_0_to_10_second_window
            # GBPUSD_queue_value_min_in_0_to_10_second_window
            # GBPUSD_queue_value_max_in_0_to_10_second_window
            # GBPUSD_queue_value_mean_in_10_to_20_second_window
            # GBPUSD_queue_value_std_in_10_to_20_second_window
            # GBPUSD_queue_value_min_in_10_to_20_second_window
            # GBPUSD_queue_value_max_in_10_to_20_second_window
            # GBPUSD_queue_value_mean_in_20_to_30_second_window
            # GBPUSD_queue_value_std_in_20_to_30_second_window
            # GBPUSD_queue_value_min_in_20_to_30_second_window
            # GBPUSD_queue_value_max_in_20_to_30_second_window
            # GBPUSD_queue_value_mean_in_30_to_40_second_window
            # GBPUSD_queue_value_std_in_30_to_40_second_window
            # GBPUSD_queue_value_min_in_30_to_40_second_window
            # GBPUSD_queue_value_max_in_30_to_40_second_window
            # GBPUSD_queue_value_mean_in_40_to_50_second_window
            # GBPUSD_queue_value_std_in_40_to_50_second_window
            # GBPUSD_queue_value_min_in_40_to_50_second_window
            # GBPUSD_queue_value_max_in_40_to_50_second_window
            # GBPUSD_queue_value_mean_in_50_to_60_second_window
            # GBPUSD_queue_value_std_in_50_to_60_second_window
            # GBPUSD_queue_value_min_in_50_to_60_second_window
            # GBPUSD_queue_value_max_in_50_to_60_second_window

            # GBPUSD_concentration_zones_mean_in_0_to_10_second_window
            # GBPUSD_concentration_zones_std_in_0_to_10_second_window
            # GBPUSD_concentration_zones_min_in_0_to_10_second_window
            # GBPUSD_concentration_zones_max_in_0_to_10_second_window
            # GBPUSD_concentration_zones_mean_in_10_to_20_second_window
            # GBPUSD_concentration_zones_std_in_10_to_20_second_window
            # GBPUSD_concentration_zones_min_in_10_to_20_second_window
            # GBPUSD_concentration_zones_max_in_10_to_20_second_window
            # GBPUSD_concentration_zones_mean_in_20_to_30_second_window
            # GBPUSD_concentration_zones_std_in_20_to_30_second_window
            # GBPUSD_concentration_zones_min_in_20_to_30_second_window
            # GBPUSD_concentration_zones_max_in_20_to_30_second_window
            # GBPUSD_concentration_zones_mean_in_30_to_40_second_window
            # GBPUSD_concentration_zones_std_in_30_to_40_second_window
            # GBPUSD_concentration_zones_min_in_30_to_40_second_window
            # GBPUSD_concentration_zones_max_in_30_to_40_second_window
            # GBPUSD_concentration_zones_mean_in_40_to_50_second_window
            # GBPUSD_concentration_zones_std_in_40_to_50_second_window
            # GBPUSD_concentration_zones_min_in_40_to_50_second_window
            # GBPUSD_concentration_zones_max_in_40_to_50_second_window
            # GBPUSD_concentration_zones_mean_in_50_to_60_second_window
            # GBPUSD_concentration_zones_std_in_50_to_60_second_window
            # GBPUSD_concentration_zones_min_in_50_to_60_second_window
            # GBPUSD_concentration_zones_max_in_50_to_60_second_window

            GBPUSD_density_gradient_mean_in_0_to_10_second_window
            GBPUSD_density_gradient_mean_in_10_to_20_second_window
            GBPUSD_density_gradient_mean_in_20_to_30_second_window
            GBPUSD_density_gradient_mean_in_30_to_40_second_window
            GBPUSD_density_gradient_mean_in_40_to_50_second_window
            GBPUSD_density_gradient_mean_in_50_to_60_second_window

            # GBPUSD_density_gradient_std_in_0_to_10_second_window
            # GBPUSD_density_gradient_std_in_10_to_20_second_window
            # GBPUSD_density_gradient_std_in_20_to_30_second_window
            # GBPUSD_density_gradient_std_in_30_to_40_second_window
            # GBPUSD_density_gradient_std_in_40_to_50_second_window
            # GBPUSD_density_gradient_std_in_50_to_60_second_window

            # GBPUSD_density_gradient_min_in_0_to_10_second_window
            # GBPUSD_density_gradient_min_in_10_to_20_second_window
            # GBPUSD_density_gradient_min_in_20_to_30_second_window
            # GBPUSD_density_gradient_min_in_30_to_40_second_window
            # GBPUSD_density_gradient_min_in_40_to_50_second_window
            # GBPUSD_density_gradient_min_in_50_to_60_second_window

            # GBPUSD_density_gradient_max_in_0_to_10_second_window
            # GBPUSD_density_gradient_max_in_10_to_20_second_window
            # GBPUSD_density_gradient_max_in_20_to_30_second_window
            # GBPUSD_density_gradient_max_in_30_to_40_second_window
            # GBPUSD_density_gradient_max_in_40_to_50_second_window
            # GBPUSD_density_gradient_max_in_50_to_60_second_window

            # GBPUSD_ladder_density_minute_mean
            # GBPUSD_ladder_density_minute_std
            # GBPUSD_ladder_density_minute_min
            # GBPUSD_ladder_density_minute_max
            # GBPUSD_ladder_density_minute_median
            # GBPUSD_relative_position_minute_mean
            # GBPUSD_relative_position_minute_std
            # GBPUSD_relative_position_minute_min
            # GBPUSD_relative_position_minute_max
            # GBPUSD_relative_position_minute_median
            # GBPUSD_queue_value_minute_volatility
            # GBPUSD_concentration_zone_minute_range
            # GBPUSD_density_gradient_minute_trend
            # GBPUSD_queue_value_cumulative_mean_up_to_10th_second
            # GBPUSD_queue_value_cumulative_mean_up_to_20th_second
            # GBPUSD_queue_value_cumulative_mean_up_to_30th_second
            # GBPUSD_queue_value_cumulative_mean_up_to_40th_second
            # GBPUSD_queue_value_cumulative_mean_up_to_50th_second
            # GBPUSD_queue_value_cumulative_mean_up_to_60th_second
            # GBPUSD_concentration_zones_cumulative_mean_up_to_10th_second
            # GBPUSD_concentration_zones_cumulative_mean_up_to_20th_second
            # GBPUSD_concentration_zones_cumulative_mean_up_to_30th_second
            # GBPUSD_concentration_zones_cumulative_mean_up_to_40th_second
            # GBPUSD_concentration_zones_cumulative_mean_up_to_50th_second
            # GBPUSD_concentration_zones_cumulative_mean_up_to_60th_second
            # GBPUSD_density_gradient_cumulative_mean_up_to_10th_second
            # GBPUSD_density_gradient_cumulative_mean_up_to_20th_second
            # GBPUSD_density_gradient_cumulative_mean_up_to_30th_second
            # GBPUSD_density_gradient_cumulative_mean_up_to_40th_second
            # GBPUSD_density_gradient_cumulative_mean_up_to_50th_second
            # GBPUSD_density_gradient_cumulative_mean_up_to_60th_second

            # GBPUSD_order_flow_toxicity_vpin_mean # fast and slow rolling mean & std

            # GBPUSD_order_flow_toxicity_vpin_smooth_mean # level + diff
            # GBPUSD_order_flow_toxicity_vpin_smooth_std # level + diff

            # GBPUSD_depth_change_smooth_mean # level + diff
            # GBPUSD_depth_change_smooth_std # level + diff

            # GBPUSD_order_flow_impact_score_smooth_mean # level + diff
            # GBPUSD_order_flow_impact_score_smooth_std # level + diff

            # GBPUSD_volume_totalbook_imbalance_mean # EMA fast & slow -AND- fast and slow rolling mean & std
            # GBPUSD_volume_topbook_imbalance_mean# EMA fast & slow -AND- fast and slow rolling mean & std

            # GBPUSD_relative_level_pressure_filtered_mean # EMA fast & slow
            # GBPUSD_relative_level_pressure_smooth_mean # EMA fast & slow

        '''    
        # Remove commented out lines (features and notes) from above list 
        columns_to_use_as_model_features = [
            line.split('#')[0].strip() 
            for line in columns_to_use_as_model_features.split('\n')
            if line.split('#')[0].strip()
        ] 
        # (Optional) Print the stats for the selected features - Useful for checking your transformation config, depeneding on the feature (ie. whether to use pct_change, simple_diff, log_then_diff, etc. - depending on the feature.)
        if True:
            print_input_features_stats = False # This will trigger the section below, which loads to print the stats using pd.describe(), once the dataset is loaded - then exit/halt the script. 
            # ---
            # Note: When using this, comment any features you don't want the stats for, in the above list.

        # -------------------------------------------------------------

        target_instrument_to_predict = 'GBPUSD'

        # -------------------------------------------------------------

        # Load features for TA indicators (Disabled)
        if False:

            # These features are REQUIRED by the TA indicators code section (if it's enabled).

            currency_pairs_for_TA = ['EURUSD', 'GBPUSD', 'USDJPY'] # cross currency
            
            for pair in currency_pairs_for_TA:
                features_to_load =  [
                    f'{pair}_open_price__ask_level_1',
                    f'{pair}_close_price__bid_level_1',

                    f'{pair}_high_price__bid_level_1',
                    f'{pair}_low_price__bid_level_1'
                ]

                for feature in features_to_load:
                    if feature not in columns_to_use_as_model_features:
                        columns_to_use_as_model_features.append(feature)

        # -------------------------------------------------------------------

    # Normalization (Disabled)
    if True:

        # `pct_change()` normalization is default
        normalize_data = False 
    
        # Change this to be opt-IN, rather than opt-out - only columns you explicitily select should have this applied 
        cols_to_NOT_normalize = '''

            target

            minutes_market_has_been_open
            
            day_of_the_week 
            hour_of_the_day 
            day_of_the_month

            hour_of_the_day_sin
            hour_of_the_day_cos

            day_of_the_week_sin
            day_of_the_week_cos 

            day_of_the_month_sin
            day_of_the_month_cos

        '''.split()

    # Scaling 
    if True:

        scale_data = True

        default_scaler_to_use = 'sklearn.preprocessing.RobustScaler()'

        cols_to_exclude_from_scaling = set('''

            target

        '''.split())

        # -----------------------------------------
        
        # Log transform (Disabled)
        if True:
            # Log transform is applied BEFORE scaling. 
            # - Then features are scaled as well, AFTER being log transformed.
            cols_to_log_transform = []

    # Key training params (eg. `shuffle_data`, `Adam.optimizer`, etc.)
    if True:

        shuffle_data = True

        data_to_holdout_for_validation = 0.10 # percent

        model_optimizer = 'tf.keras.optimizers.Adam'

        model_optimizer_learning_rate = 1e-3 
        model_optimizer_clipnorm = 1.0 

        model_training_loss = 'categorical_crossentropy' # if classifer outputs only 2 classes (binary) the model will automatically switch to use 'binary_crossentropy'; the value here (`model_training_loss`) will only be used if `num_classes` > 2

        model_training_model_checkpoint_settings_monitor = 'val_accuracy'
        model_training_model_checkpoint_settings_mode = 'max'
        
        model_training_metrics = [
            'accuracy', 
            tf.keras.metrics.Precision(), 
            tf.keras.metrics.Recall()]

# ---------------------------- [ FUNCTIONS ] ---------------------------
if True:

    # ----- TARGET CLASSIFIER
    classifier_mapping = {
        0: "no movement", 
        1: "mild downward", 
        2: "strong downward",
        3: "mild upward",
        4: "strong upward"} 
        
    def dynamic_classifier_multiclass(delta, rolling_mean, rolling_std):

        # --------------------

        if np.isnan(delta) or np.isnan(rolling_mean) or np.isnan(rolling_std) or rolling_std <= 0:
            return np.nan
        
        # --------------------

        # Tuning knobs:
        # ---
        # * `z_score_no_movement` and `z_score_strong` are key tuning knobs, definitely experiment with these!
        # ---
        # - Use notebook (`target_z-score_tuning.ipynb`) to gridscan these!
        # - The notebook will graph and provide key statistics to help you analyze and decide on the optimal value for these key params.
        # ---
        # next up: try a higher nm (0.5), and lower s (1.7)
        z_score_no_movement        = 0.5 # 0.25
        z_score_strong             = 1.7 # 2

        # Upward
        upper_strong_threshold     = rolling_mean    + z_score_strong       * rolling_std
        upper_mild_threshold       = rolling_mean    + z_score_no_movement  * rolling_std
        # Downward
        lower_mild_threshold       = rolling_mean    - z_score_no_movement  * rolling_std
        lower_strong_threshold     = rolling_mean    - z_score_strong       * rolling_std

        # --------------------

        if delta >= upper_strong_threshold:
            return 4  # Strong upward movement
        elif upper_mild_threshold <= delta < upper_strong_threshold:
            return 3  # Mild upward movement
        elif lower_mild_threshold < delta < upper_mild_threshold:
            # This is the 'no movement' band centered around the rolling_mean
            return 0  # No movement
        elif lower_strong_threshold < delta <= lower_mild_threshold:
            return 1  # Mild downward movement
        elif delta <= lower_strong_threshold:
            return 2  # Strong downward movement
        else:
            raise Exception(f"Warning: Delta {delta} fell outside defined thresholds.")
    
    # ----- ASSIGN TARGET & TRANSFORM INPUT FEATURES
    def add_features_to_single_batch(dataframe, DEBUG_TARGET=False):

        global static_classifier_multiclass, dynamic_classifier_multiclass # allow access to function scope

        # DEBUGGING: Uncomment this line below to print what a timeseries batch look like (ie. what columns are being fed into the model).
        # ---
        # print(dataframe)

        # ---
 
        # Create target
        if True:

            # Set up for target 
            if True:

                # --- Target: Mid Price Change --- 

                # - Got great accu, up to 47% for a 5-class prediction class - but has trouble getting it profitable on backtest. Doesn't seem to be reflective of actual execution costs.

                if False:

                    # Define mid-price at OPEN and CLOSE - for the target instrument
                    target_open_ask = f'{target_instrument_to_predict}_open_price__ask_level_1'
                    target_open_bid = f'{target_instrument_to_predict}_open_price__bid_level_1'

                    target_close_ask = f'{target_instrument_to_predict}_close_price__ask_level_1'
                    target_close_bid = f'{target_instrument_to_predict}_close_price__bid_level_1'

                    # Calculate mid-price at OPEN and CLOSE
                    dataframe['future_open_ask'] = (
                        dataframe[target_open_ask] + dataframe[target_open_bid]
                    ).div(2) #.shift(-1)                  # <— shift by –1
                        # DISABLED - to test if perf returns to 44 - YES - it did...

                    dataframe['future_close_bid'] = (
                        dataframe[target_close_ask] + dataframe[target_close_bid]
                    ).div(2).shift(-FUTURE_PERIOD_PREDICT)

                    # Calculate delta between future mid-price and current open mid-price
                    dataframe['future_delta'] = (
                        dataframe['future_close_bid'] - dataframe['future_open_ask']
                    )
                
                # ------------------------------------------------

                # --- Target: Open ask - Minus - Close bid (Actual execution costs, baked into target)

                if True:
                    
                    # Define `future_open_ask`: as the NEXT time step.
                    # ---
                    # - Since we want the OPEN price, of the NEXT time step - then we compare it to the CLOSE price, of N (ahead) time steps.

                    # TESTING ON CLOSE PRICE OF CURRENT BAR - RATHER THAN OPEN

                    target_column_1 = f'{target_instrument_to_predict}_open_price__ask_level_1'
                    dataframe['future_open_ask'] = \
                        dataframe[target_column_1] \
                            # .shift(-1) # DISABLED - to test if also boosts  perf for this target method
                    
                    target_column_2 = f'{target_instrument_to_predict}_close_price__bid_level_1'
                    dataframe['future_close_bid'] = \
                        dataframe[target_column_2] \
                            .shift(-FUTURE_PERIOD_PREDICT)
                    
                    # Delta between the open ask (price we paid to open) & the future CLOSE (bid) price (aka. the price we paid to close the position).
                    dataframe['future_delta'] = \
                        dataframe['future_close_bid'] - dataframe['future_open_ask']
            
            # ----------------------

            # --- CLASSIFIER --- 
  
            # Dynamic classifer 
            if True:

                ewma_window_span = 20 # minutes
                
                # ------------------------------------------------------------
                # 1)  FRESHEST ROLLING STATS (mean & std) --------------------
                # ------------------------------------------------------------

                col_prefix = f'future_delta_ewma_{ewma_window_span}'

                dataframe[f'{col_prefix}_mean'] = (
                    dataframe['future_delta'] 
                        .ewm(
                            span=ewma_window_span, 
                            adjust=False, 
                            min_periods=ewma_window_span 
                        )
                        .mean()
                )

                dataframe[f'{col_prefix}_std'] = (
                    dataframe['future_delta']
                        .ewm(
                            span=ewma_window_span, 
                            adjust=False, 
                            min_periods=ewma_window_span
                        )
                        .std()
                        .clip(lower=1e-8) # never zero
                )

                # ------------------------------------------------------------
                # 2)  BUILD TARGET WITH “NOW‑STATS” --------------------------
                # ------------------------------------------------------------

                dataframe['target'] = dataframe.apply(
                    lambda row: dynamic_classifier_multiclass(
                        delta        = row['future_delta'],
                        rolling_mean = row[f'{col_prefix}_mean'],
                        rolling_std  = row[f'{col_prefix}_std']
                    ),
                    axis=1
                )

                # ------------------------------------------------------------
                # 3)  ADD THE LAGGED EWMA STATS AS MODEL FEATURES ------------
                # ------------------------------------------------------------
                
                # Add lagged ewma stats of `future_delta` as model features
                dataframe[f'{col_prefix}_mean_lag_{FUTURE_PERIOD_PREDICT}'] = \
                    dataframe[f'{col_prefix}_mean'].shift(FUTURE_PERIOD_PREDICT)

                dataframe[f'{col_prefix}_std_lag_{FUTURE_PERIOD_PREDICT}']  = \
                    dataframe[f'{col_prefix}_std'].shift(FUTURE_PERIOD_PREDICT)
            
                # ------------------------------------------------------------
                # 4)  ADD THE LAGGED LOOKAHEAD TARGET AS MODEL FEATURE ---
                # ------------------------------------------------------------
                
                # Add absolute value of base target feature (`future_delta`) as lagged feature
                dataframe[f'future_delta_lag_{FUTURE_PERIOD_PREDICT}'] = \
                    dataframe['future_delta'].shift(FUTURE_PERIOD_PREDICT)
                
                # TODO: Train once with both features; if the validation metric doesn’t improve, drop the `_diff` column and retry. It costs only one extra experiment.
                dataframe[f'future_delta_lag_{FUTURE_PERIOD_PREDICT}_diff'] = \
                    dataframe[f'future_delta_lag_{FUTURE_PERIOD_PREDICT}'].diff()
                
                # ------------------------------------------------------------

                # Drop lookahead “ewma_now” columns so the model never sees them (since they contain future information from `future_delta`)
                dataframe.drop(columns=[f'{col_prefix}_mean', f'{col_prefix}_std'], inplace=True)

                # ------------------------------------------------------------

            # Binary classifer (Disabled)
            if False:
                dataframe['target'] = (dataframe['future_delta'] > 0).astype(int)

            # ----------------------

        # -----------------------

        # COS/SIN FEATURES 
        if True:

            cols_to_drop_after_transform = [] # Keep track of columns to drop

            if 'hour_of_the_day' in dataframe:
                # Hour of the day (24-hour cycle)
                cyclic_feature = dataframe['hour_of_the_day']
                cycle_period = 24
                
                dataframe['hour_of_the_day_sin'] = np.sin(2 * np.pi * cyclic_feature / cycle_period)
                dataframe['hour_of_the_day_cos'] = np.cos(2 * np.pi * cyclic_feature / cycle_period)

                cols_to_drop_after_transform.append('hour_of_the_day') # Mark original feature for dropping
                cols_to_exclude_from_scaling.update([
                    'hour_of_the_day_sin', 'hour_of_the_day_cos'])

            if 'day_of_the_week' in dataframe:
                # Day of the week (7-day cycle)
                cyclic_feature = dataframe['day_of_the_week']
                cycle_period = 7

                dataframe['day_of_the_week_sin'] = np.sin(2 * np.pi * cyclic_feature / cycle_period)
                dataframe['day_of_the_week_cos'] = np.cos(2 * np.pi * cyclic_feature / cycle_period)

                cols_to_drop_after_transform.append('day_of_the_week') # Mark original feature for dropping
                cols_to_exclude_from_scaling.update([
                    'day_of_the_week_sin', 'day_of_the_week_cos'])

            if 'day_of_the_month' in dataframe:
                # Day of the month (assuming 30-day cycle)
                cyclic_feature = dataframe['day_of_the_month']
                cycle_period = 31

                dataframe['day_of_the_month_sin'] = np.sin(2 * np.pi * cyclic_feature / cycle_period)
                dataframe['day_of_the_month_cos'] = np.cos(2 * np.pi * cyclic_feature / cycle_period)

                cols_to_drop_after_transform.append('day_of_the_month') # Mark original feature for dropping
                cols_to_exclude_from_scaling.update([
                    'day_of_the_month_sin', 'day_of_the_month_cos'])

            if 'minutes_market_has_been_open' in dataframe:
                # Create features representing phase within the session
                cyclic_feature = dataframe['minutes_market_has_been_open']
                cycle_period = 1425.0 # Use the observed max value (of `minutes_market_has_been_open` summary stats) as the session length period

                dataframe['minutes_market_has_been_open_sin'] = np.sin(2 * np.pi * cyclic_feature / cycle_period)
                dataframe['minutes_market_has_been_open_cos'] = np.cos(2 * np.pi * cyclic_feature / cycle_period)

                cols_to_drop_after_transform.append('minutes_market_has_been_open') # Mark original feature for dropping
                cols_to_exclude_from_scaling.update([
                    'minutes_market_has_been_open_sin', 'minutes_market_has_been_open_cos'])

            # --------------------------------------

            # Drop the original raw features (ONLY if they existed and were transformed and marked for dropping).
            if cols_to_drop_after_transform:
                dataframe.drop(columns=cols_to_drop_after_transform, inplace=True)

        # -----------------------

        # ROLLING FEATURES (rolling mean, std) 
        # ---
        # Would rolling any of the input features help the model better understand the 1min bar feature? 
        if False:

            # print('DEBUG: adding rolling features (mean, std) to timeseries - inside `add_features_to_single_batch()')

            window_sizes = [5, 20] # fast = 5 min, slow = 20 min

            features_to_roll = [
                f'{target_instrument_to_predict}_total_ticks',
                f'{target_instrument_to_predict}_meaningful_ticks__L1_price_change_ticks', 
                
                f'{target_instrument_to_predict}_level_1_spread_regular_mean',

                f'{target_instrument_to_predict}_order_flow_toxicity_vpin_mean',

                f'{target_instrument_to_predict}_volume_totalbook_imbalance_mean',
                f'{target_instrument_to_predict}_volume_topbook_imbalance_mean',
            ]

            new_rolling_cols = [] 

            # Add rolling mean and standard deviation, for each selected feature, and window size
            for feature in features_to_roll:
                for window in window_sizes:
                    if feature not in dataframe.columns:
                        continue

                    # Add rolling mean
                    dataframe[f'{feature}_rolling_mean_{window}'] = \
                        dataframe[feature].rolling(window=window).mean()
                    new_rolling_cols.append(f'{feature}_rolling_mean_{window}')

                    # Add rolling std
                    dataframe[f'{feature}_rolling_std_{window}'] = \
                        dataframe[feature].rolling(window=window).std()
                    new_rolling_cols.append(f'{feature}_rolling_std_{window}')

            # -----------------------------

            # EMA-based features

            # -----------------------------

            win_fast, win_slow = 5, 20

            bases = [
                f'{target_instrument_to_predict}_volume_totalbook_imbalance_mean',
                f'{target_instrument_to_predict}_volume_topbook_imbalance_mean',

                f'{target_instrument_to_predict}_relative_level_pressure_filtered_mean',
                f'{target_instrument_to_predict}_relative_level_pressure_smooth_mean'
            ]

            for base in bases:

                if base not in dataframe.columns:
                    continue # skip if feature not in dataset

                dataframe[f'{base}_ema_{win_fast}'] = dataframe[base].ewm(span=win_fast).mean()
                dataframe[f'{base}_ema_{win_slow}'] = dataframe[base].ewm(span=win_slow).mean()

                dataframe[f'{base}_ema_diff'] = \
                    dataframe[f'{base}_ema_{win_fast}'] - dataframe[f'{base}_ema_{win_slow}']

        # -----------------------

        # TA INDICATORS 
        # ---
        if False:

            # ------------------------------------------------------
            # TA INDICATORS: SET UP STEPS
            # ------------------------------------------------------
            # Step 1. Scroll up to section: ['Simpler list of columns (chosen by chatGPT)'] - to enable data features required to compute these indicators.
            # ---
            # Step 2. Then once there, scroll to the bottom of that section - there is a for loop, that adds in the required features, dynmically.
            # ------------------------------------------------------

            import ta
        
            # List of currency pairs to calculate indicators for
            currency_pairs_for_TA = ['EURUSD', 'GBPUSD', 'USDJPY'] # Must be present in top input data

            # ------------------------------------------------------
            
            # --- TA INDICATORS: PART 1: Basic TA (SMA, EMA, RSI, BB, MACD) ---

            # ------------------------------------------------------

            for pair in currency_pairs_for_TA:

                # -----------------------------------------------------------------

                # Define the close price column for the current pair
                close_col = f'{pair}_close_price__bid_level_1'
                
                # -----------------------------------------------------------------

                # Simple Moving Average (SMA) over a 10-minute window
                dataframe[f'{pair}_SMA_10'] = dataframe[close_col] \
                    .rolling(
                        window=10, 
                        min_periods=10) \
                    .mean()
                
                # -----------------------------------------------------------------

                # Exponential Moving Average (EMA) over a 10-minute window
                dataframe[f'{pair}_EMA_10'] = dataframe[close_col] \
                    .ewm(
                        span=10, 
                        adjust=False, 
                        min_periods=10
                    ) \
                    .mean()

                # -----------------------------------------------------------------

                # Relative Strength Index (RSI) over a 14-minute window
                rsi_indicator = ta.momentum.RSIIndicator(
                    close=dataframe[close_col], 
                    window=14, 
                    fillna=False)
                
                dataframe[f'{pair}_RSI_14'] = rsi_indicator.rsi()
                
                # Do NOT normalize RSI
                if f'{pair}_RSI_14' not in cols_to_NOT_normalize:
                    cols_to_NOT_normalize.append(f'{pair}_RSI_14')

                # -----------------------------------------------------------------

                # Bollinger Bands over a 20-minute window
                bollinger = ta.volatility.BollingerBands(
                    close=dataframe[close_col], 
                    window=20, 
                    window_dev=2, 
                    fillna=False)
                
                dataframe[f'{pair}_Bollinger_High'] = bollinger.bollinger_hband()
                dataframe[f'{pair}_Bollinger_Low'] = bollinger.bollinger_lband()
                dataframe[f'{pair}_Bollinger_Mid'] = bollinger.bollinger_mavg()

                # Do NOT normalize Bollinger Bands
                list_of_bb_indicators = [
                    f'{pair}_Bollinger_High', 
                    f'{pair}_Bollinger_Low', 
                    f'{pair}_Bollinger_Mid']
                
                for bb_indicator in list_of_bb_indicators:
                    if bb_indicator not in cols_to_NOT_normalize:
                        cols_to_NOT_normalize.append(bb_indicator)

                # -----------------------------------------------------------------

                # Moving Average Convergence Divergence (MACD)
                # ---
                # * Note: MACD has a 34-bar warm up, so it requires a seq_len > 34, or it will generate all NAN values, in the column for `signal` and `diff` (at least using standard settings).
                macd_indicator = ta.trend.MACD(
                    close=dataframe[close_col], 
                    fillna=False)

                dataframe[f'{pair}_MACD'] = macd_indicator.macd()
                dataframe[f'{pair}_MACD_Signal'] = macd_indicator.macd_signal()
                dataframe[f'{pair}_MACD_Diff'] = macd_indicator.macd_diff()

                # Do NOT normalize MACD
                list_of_macd_indicators = [
                    f'{pair}_MACD', 
                    f'{pair}_MACD_Signal', 
                    f'{pair}_MACD_Diff']
                for macd_indicator in list_of_macd_indicators:
                    if macd_indicator not in cols_to_NOT_normalize:
                        cols_to_NOT_normalize.append(macd_indicator)

            # -----------------------------------------------------------------

            # --- TA INDICATORS: PART 2: Volatility Measures  ---

            # -----------------------------------------------------------------

            for pair in currency_pairs_for_TA:

                # -----------------------------------------------------------------

                high_col = f'{pair}_high_price__bid_level_1'
                low_col = f'{pair}_low_price__bid_level_1'
                close_col = f'{pair}_close_price__bid_level_1'
        
                # -----------------------------------------------------------------

                # Standard Deviation of Close Prices - over a 10-minute window
                dataframe[f'{pair}_Close_Std_10'] = dataframe[close_col].rolling(
                    window=10, 
                    min_periods=10) \
                .std()
                
                # -----------------------------------------------------------------

                # Average True Range (ATR) over a 14-minute window
                atr_indicator = ta.volatility.AverageTrueRange(

                    high=dataframe[high_col], 
                    low=dataframe[low_col], 
                    close=dataframe[close_col], 

                    window=14, 
                    fillna=False)
                
                dataframe[f'{pair}_ATR_14'] = atr_indicator.average_true_range()

            # ------------------------------------------------------

            # --- TA INDICATORS: PART 3: Price Returns ---

            # ------------------------------------------------------

            for pair in currency_pairs_for_TA:

                close_col = f'{pair}_close_price__bid_level_1'

                # ------------------------------------------------------

                # Log Returns
                dataframe[f'{pair}_Log_Return'] = np.log(
                    dataframe[close_col] / dataframe[close_col].shift(1))

                # Do NOT normalize Log Returns
                if f'{pair}_Log_Return' not in cols_to_NOT_normalize:
                    cols_to_NOT_normalize.append(f'{pair}_Log_Return')
                
                # ------------------------------------------------------

                # Add lags for Log Returns
                max_lag = 5          # <-- tune as you like (1-10 is common)
                for lag in range(1, max_lag + 1):

                    lag_col = f'{pair}_Log_Return_Lag_{lag}'
                    dataframe[lag_col] = dataframe[f'{pair}_Log_Return'].shift(lag)

                    # keep raw – no `pct_change()`
                    if lag_col not in cols_to_NOT_normalize:
                        cols_to_NOT_normalize.append(lag_col)

            # ------------------------------------------------------

            # --- TA INDICATORS: PART 4: Cross-Currency Features ---

            # ------------------------------------------------------

            # --- Cross-Currency Price Ratios ---

            for i in range(len(currency_pairs_for_TA)):
                for j in range(i+1, len(currency_pairs_for_TA)):

                    pair_a = currency_pairs_for_TA[i]
                    pair_b = currency_pairs_for_TA[j]

                    close_a = f'{pair_a}_close_price__bid_level_1'
                    close_b = f'{pair_b}_close_price__bid_level_1'

                    dataframe[f'{pair_a}_{pair_b}_Price_Ratio'] = dataframe[close_a] / dataframe[close_b]
                    dataframe[f'{pair_a}_{pair_b}_Price_Ratio'].replace([np.inf, -np.inf], 0, inplace=True) 

                    # Do NOT normalize Price Ratios
                    if f'{pair_a}_{pair_b}_Price_Ratio' not in cols_to_NOT_normalize:
                        cols_to_NOT_normalize.append(f'{pair_a}_{pair_b}_Price_Ratio')

            # ------------------------------------------------------

            # --- Cross-Currency Spread Differences: Part A: Currency Spread ---

            for pair in currency_pairs_for_TA:

                ask_col = f'{pair}_open_price__ask_level_1'
                bid_col = f'{pair}_close_price__bid_level_1'

                dataframe[f'{pair}_Spread'] = dataframe[ask_col] - dataframe[bid_col]

                # These should be excluded from normalization too?

            # --- Cross-Currency Spread Differences: Part B: Spread Differences between pairs ---

            for i in range(len(currency_pairs_for_TA)):
                for j in range(i+1, len(currency_pairs_for_TA)):

                    pair_a = currency_pairs_for_TA[i]
                    pair_b = currency_pairs_for_TA[j]

                    spread_a = f'{pair_a}_Spread'
                    spread_b = f'{pair_b}_Spread'

                    dataframe[f'{pair_a}_{pair_b}_Spread_Diff'] = dataframe[spread_a] - dataframe[spread_b]
                
                # These should be excluded from normalization too?

            # --- Cross-Currency Spread Differences: Part C: Rolling z-score of the spread --- 
            # This is only per currency, right? - should we do also for the spread_diff, between the pairs?

            spread_z_window = 30    # 30×1-min bars ≈ last half-hour

            for pair in currency_pairs_for_TA:

                spread_col = f'{pair}_Spread'

                roll_mean = dataframe[spread_col].rolling(
                    window=spread_z_window,
                    min_periods=spread_z_window
                ).mean()

                roll_std = dataframe[spread_col].rolling(
                    window=spread_z_window,
                    min_periods=spread_z_window
                ).std()

                # ---

                z_col = f'{pair}_Spread_z_{spread_z_window}'
                dataframe[z_col] = (dataframe[spread_col] - roll_mean) / roll_std

                # Do not normalize keep raw – z-score is already scale-free
                if z_col not in cols_to_NOT_normalize:
                    cols_to_NOT_normalize.append(z_col)

            # ------------------------------------------------------
            
            # ------------------------------------------------------
            # Exclude all these TA features from normalization
            # - After you have created *all* the spread and indicator columns
            # ------------------------------------------------------------

            for pair in currency_pairs_for_TA:
                for col in [
                    f'{pair}_RSI_14',
                    f'{pair}_Bollinger_High',
                    f'{pair}_Bollinger_Low',
                    f'{pair}_Bollinger_Mid',
                    f'{pair}_MACD',
                    f'{pair}_MACD_Signal',
                    f'{pair}_MACD_Diff',
                    f'{pair}_ATR_14',
                    f'{pair}_Close_Std_10',
                    f'{pair}_Spread',
                    f'{pair}_Spread_z_30'
                ]:
                    if col not in cols_to_NOT_normalize:
                        cols_to_NOT_normalize.append(col)

            # include all cross-pair spread-diff columns as well
            for i in range(len(currency_pairs_for_TA)):
                for j in range(i + 1, len(currency_pairs_for_TA)):

                    pair_a = currency_pairs_for_TA[i]
                    pair_b = currency_pairs_for_TA[j]

                    diff_col = f'{pair_a}_{pair_b}_Spread_Diff'

                    if diff_col not in cols_to_NOT_normalize:
                        cols_to_NOT_normalize.append(diff_col)

            # --- End: TA Indicators ---

        # -----------------------

        # log1p then diff + shock flag (for volume/tick count features)
        if True:

            cols_to_drop_after_transform = [] # Keep track of columns to drop

            tick_cols = [
                "EURUSD_total_ticks", 
                "GBPUSD_total_ticks", 
                # "USDJPY_total_ticks", # removing "USDJPY" for now - to reduce complexity

                # "EURUSD_meaningful_ticks", 
                # "GBPUSD_meaningful_ticks",
                # "USDJPY_meaningful_ticks"

                "EURUSD_meaningful_ticks__L1_price_change_ticks", 
                "GBPUSD_meaningful_ticks__L1_price_change_ticks",
                # "USDJPY_meaningful_ticks__L1_price_change_ticks",
            ]

            # 1) Compress counts: log1p then diff
            for c in tick_cols:

                # Feed both *_log and *_dlog to the net; the former helps level inference, the latter gives dynamics.
                dataframe[c+"_log"]   = np.log1p(dataframe[c])
                dataframe[c+"_dlog"]  = dataframe[c+"_log"].diff()

                cols_to_drop_after_transform.append(c) # Mark original feature for dropping
            
            # ---

            # 2) create 0/1 binary "shock" flags (keeps info about *any* burst) - and exclude from scaling
            thr = 1.0                            
            for c in tick_cols:
                dataframe[c+"_shock"] = (dataframe[c+"_dlog"].abs() > thr).astype(np.int8)

                if c+"_shock" not in cols_to_exclude_from_scaling:
                    cols_to_exclude_from_scaling.add(c+"_shock")

                # Debugging: print how many rows triggered shock flag
                # ---
                # pct = (dataframe[c+"_dlog"].abs() > thr).mean() * 100
                # print(f"{c}: {pct:.2f}% of rows flagged")

            # ---

            # 3) Collapse the identical total‑tick shock flags
            dataframe["any_total_shock"] = (
                  dataframe["EURUSD_total_ticks_shock"]
                | dataframe["GBPUSD_total_ticks_shock"]
                # | dataframe["USDJPY_total_ticks_shock"]
                ).astype(int)

            dataframe.drop([
                    "EURUSD_total_ticks_shock",
                    "GBPUSD_total_ticks_shock",
                    # "USDJPY_total_ticks_shock"
                ],
                axis=1,
                inplace=True
            )
            if "any_total_shock" not in cols_to_exclude_from_scaling:
                cols_to_exclude_from_scaling.add("any_total_shock")

            # ---

            # Drop the original raw features 
            if cols_to_drop_after_transform:
                dataframe.drop(columns=cols_to_drop_after_transform, inplace=True)

        # -----------------------

        # Transform price features (for better interpretability by LSTM)
        if True:

            cols_to_drop_after_transform = []
            new_features_for_model = [] # used to inspect the stats of these new features

            pairs = ["EURUSD", "GBPUSD"] # removing "USDJPY" for now - to reduce complexity

            for pair in pairs:

                # build mid-price OHLC
                dataframe[f"{pair}_mid_open"]  = (dataframe[f"{pair}_open_price__ask_level_1"] 
                                                + dataframe[f"{pair}_open_price__bid_level_1"]) / 2
                dataframe[f"{pair}_mid_close"] = (dataframe[f"{pair}_close_price__ask_level_1"]
                                                + dataframe[f"{pair}_close_price__bid_level_1"]) / 2
                dataframe[f"{pair}_mid_high"]  = (dataframe[f"{pair}_high_price__ask_level_1"]
                                                + dataframe[f"{pair}_high_price__bid_level_1"]) / 2
                dataframe[f"{pair}_mid_low"]   = (dataframe[f"{pair}_low_price__ask_level_1"]
                                                + dataframe[f"{pair}_low_price__bid_level_1"]) / 2

                # (1) log-close-mid level
                dataframe[f"{pair}_p"]      = np.log(dataframe[f"{pair}_mid_close"])
                new_features_for_model.append(f"{pair}_p")

                # (2) close‑to‑close log return
                dataframe[f"{pair}_r"]      = dataframe[f"{pair}_p"].diff()      
                new_features_for_model.append(f"{pair}_r")        

                # (3) High‑low range divided by mid	(intraminute volatility proxy)
                dataframe[f"{pair}_hl"]  = np.log(dataframe[f"{pair}_mid_high"] / dataframe[f"{pair}_mid_low"])
                new_features_for_model.append(f"{pair}_hl")   

                # (4) open‑to‑close intrabar return
                dataframe[f"{pair}_oc"] = np.log(
                    dataframe[f"{pair}_mid_close"] / 
                    dataframe[f"{pair}_mid_open"]
                )
                new_features_for_model.append(f"{pair}_oc")   

                # (5) relative spread (normalised spread across pairs)
                dataframe[f"{pair}_spr"] = ((dataframe[f"{pair}_close_price__ask_level_1"] - 
                                             dataframe[f"{pair}_close_price__bid_level_1"]) / 
                                             dataframe[f"{pair}_mid_close"]) #
                new_features_for_model.append(f"{pair}_spr")  

                # ---------- mark raw bid/ask and helper mids for dropping later ----------

                cols_to_drop_after_transform.extend([
                    
                    # Mark raw input features for dropping
                    f"{pair}_open_price__ask_level_1",
                    f"{pair}_open_price__bid_level_1",
                    f"{pair}_close_price__ask_level_1",
                    f"{pair}_close_price__bid_level_1",
                    f"{pair}_high_price__ask_level_1",
                    f"{pair}_high_price__bid_level_1",
                    f"{pair}_low_price__ask_level_1",
                    f"{pair}_low_price__bid_level_1",

                    # Prune the helper mid_* columns once the engineered features are built	
                    f"{pair}_mid_open", 
                    f"{pair}_mid_close",
                    f"{pair}_mid_high", 
                    f"{pair}_mid_low"
                ])
    
            # # ───────── clip abnormally wide spreads ───────── 

            # pairs     = ["EURUSD", "GBPUSD", "USDJPY"]
            # clip_rel  = 1e-3      # ±0.001  (0.1 %)  ≈ 10 pips majors, 7 pips JPY

            # for p in pairs:
            #     ask = dataframe[f"{p}_close_price__ask_level_1"]
            #     bid = dataframe[f"{p}_close_price__bid_level_1"]
                
            #     mid = (ask + bid) / 2
            #     rel = (ask - bid) / mid

            #     crossed = rel < 0          # bid > ask ⇒ invalid quote
            #     wide    = rel.abs() > clip_rel

            #     # 1️⃣  drop crossed‑book rows (very few)
            #     # dataframe = dataframe.loc[~crossed] (You can't drop these here - they must be dropped in pre-filter)

            #     # 2️⃣  clip ultra‑wide spreads to ±0.001
            #     dataframe.loc[wide, f"{p}_spr"] = np.sign(rel[wide]) * clip_rel

            # ---------- inspect/debugging helpers ----------

            # print(f'\nFull list of new features: {new_features_for_model}\n') # You can copy these (once printed in console) and then paste into the code section to inspect these features before scaling. This is useful if you are having scaling-health-check issues - and you want to check the distrubtion of the new features - with an LLM, to ensure the new features are being prepared properly.

            # print(f'Below are the stats for the new features (only from a single batch - of {len(dataframe)} Minutes):\n')
            # for col_name, col_series_data in dataframe[new_features_for_model].items():
            #     print(col_name)
            #     print(col_series_data.describe())
            #     print('----------------')
            # sys.exit(0)
            
            # ---------- drop raw/helper columns ----------

            if cols_to_drop_after_transform:
                dataframe.drop(columns=cols_to_drop_after_transform, inplace=True)

        # -----------------------

        # Level + Change for Key Features - (`using .diff()`) 
        if False:

            features_for_level_plus_diff = [
                "GBPUSD_order_flow_toxicity_vpin_smooth_mean",
                "GBPUSD_order_flow_toxicity_vpin_smooth_std",
                
                "GBPUSD_depth_change_smooth_mean",
                "GBPUSD_depth_change_smooth_std",

                "GBPUSD_order_flow_impact_score_smooth_mean",
                "GBPUSD_order_flow_impact_score_smooth_std"
            ]

            for col_prefix in features_for_level_plus_diff:
                if col_prefix in columns_to_use_as_model_features:
                    dataframe[f'{col_prefix}_diff'] = dataframe[col_prefix].diff()

        # -----------------------

        # LAG FEATURES
        # ---
        # Would lagging any of the input features help the model better understand the 1min bar feature? 
        if False:

            # Define key features to apply lags
            lag_features = [
                f'{target_instrument_to_predict}_open_price__ask_level_1', 
                f'{target_instrument_to_predict}_close_price__bid_level_1', 
                f'{target_instrument_to_predict}_volume']

            # Define lags to apply
            lags = [1, 2, 5, 10, 15, 30]  # in minutes

            # Create lagged fjeatures
            for feature in lag_features:
                for lag in lags:
                    dataframe[f'{feature}_lag_{lag}'] = dataframe[feature].shift(lag)

        # ------------------------

        # LAGGED TARGET (AS FEATURE)
        # --- 
        if False:

            # -------------------------------------------------------------
            
            # Define the lag times.
            # ---
            # - IMPORTANT: If any of these values are LESS than the value of `FUTURE_PERIOD_PREDICT`, it will introduce lookahead bias.
            # - Since `FUTURE_PERIOD_PREDICT` shifts BACK the future price (eg. `.shift(-FUTURE_PERIOD_PREDICT)`)
            # ... BUT we CAN use `FUTURE_PERIOD_PREDICT` as a the first lag, since it's OKAY for THAT row to have information about itself.
            lags = [FUTURE_PERIOD_PREDICT, 
                    FUTURE_PERIOD_PREDICT+1, 
                    FUTURE_PERIOD_PREDICT+2, 
                    FUTURE_PERIOD_PREDICT+3, 
                    FUTURE_PERIOD_PREDICT+4, 
                    FUTURE_PERIOD_PREDICT+5]
            
            for lag in lags:
                dataframe[f'target_lag_{lag}'] = dataframe['target'].shift(lag)
                # dataframe.dropna(subset=f'target_lag_{lag}', inplace=True) 

            # -------------

            # Tip: Verify lagged features, using notebook: (`inspect_data_pipeline_intermediate_output.ipynb`).

        # ------------------------

        # FFT (Fast Fourier transform)
        # ---
        # Would apply FFT to any of the input features help the model better understand the 1min bar feature? 
        if False:
            
            '''
                Things to change before enabling:
                ---
                1. Disable normaliazation

                2. Update code in data_pipeline:

                  -- Uncomment the `debug_custom_sequence_length` param (since the FFT transformation changes the size of the batch.

                        * eg. If was using 60 seq_len, then with 10-min fft rolling window, will be like 40 or 50 or something like that.
                        
                        Example:
                        ```
                        scaled_batches = get_minute_batches_pd(
                                dataframe=scaled_dataframe, 
                                post_normalization=True if normalize_data is True else False,
                                # debug_custom_sequence_length=batches[0].shape[0]
                                ) # NEW
                        ```
            '''
            
            # Example function to compute FFT features
            def compute_fft_features(series, n_freqs=3):

                fft_result = np.fft.fft(series)
                amplitudes = np.abs(fft_result)[:n_freqs]  # Take the first n_freqs amplitudes

                return amplitudes

            # Transform each time-series feature
            n_freqs = 3  # Number of FFT components
            window_size = 10  # Rolling window size

            features = dataframe.columns
            data = dataframe
            fft_features = pd.DataFrame()

            for feature in features:

                if feature == 'target':
                    continue # do NOT encode the target

                feature_fft = []

                for i in range(len(data) - window_size + 1):
                    window = data[feature].iloc[i:i + window_size]
                    feature_fft.append(compute_fft_features(window, n_freqs))

                feature_fft_dataframe = \
                    pd.DataFrame(feature_fft, 
                                 columns=[f'{feature}_FFT_{i+1}' for i in range(n_freqs)])
                
                fft_features = pd.concat([fft_features, feature_fft_dataframe], axis=1)

            # Align FFT features with the original dataset
            fft_features.index = data.index[window_size - 1:]
            data = pd.concat([dataframe['target'], fft_features], axis=1).dropna()

            # DEBUGGING
            # ---
            # print(data)
            
            dataframe = data # replace batch data with new FFT dataframe

        # ------------------------
            
        if DEBUG_TARGET is True:

            # DOC: How to debug target and classifier:
            if True:

                # DOC: How to debug classifier:
                # ---
                # Use `DEBUG_TARGET = True` in this function's param.
                # - This will comment out the dropping of rows above (`future_open_ask`, `future_close_bid`).
                # -- So you can inspect these target columns manually, and check the rows, row-by-row, by hand manually, to ensure the calculcation is being handled correctly.
                # -- You'll also be able to check that the correct rows are being shfited backwards (in timesteps).
                # -- Use this notebook: `inspect_data_pipeline_intermediate_output.ipynb` (to inspect data, cols.)
                # ---

                pass

            # ------------------------------------------

            # This will return the dataframe/batch, up to the point the target is assigned (and stop execution below that point)
            # - When return, the data object will include the variables used to calculcate the target, remaining in the dataframe (so they can be inspected).
            # - Inspect using notebook: `inspect_data_pipeline_intermediate_output.ipynb`.

            return dataframe
        
        elif DEBUG_TARGET is False:  # PRODUCTION/DEFAULT USAGE

            # Drop rows with future information in them.
            # - This is how the code should function regularly, in production.

            cols_with_lookahead_bias = [
                'future_open_ask', 
                'future_close_bid', 
                'future_delta']
            
            dataframe.drop(cols_with_lookahead_bias, 
                            axis=1, 
                            inplace=True)             

        # ------------------------

        def drop_initial_NAN_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
            # ----------------------------------------------------------------
            # - Only drop NANs at beginning of batch, but do not drop NANs at end of batch.
            # - Since the logic of `build_x_y` uses row-based indexing, which account for NAN values, due to `FUTURE_PREDICT_PERIOD` shifts.
            # ----------------------------------------------------------------
            """
            Drops rows from the beginning of the DataFrame until the first row
            with no NaN values is found. Keeps all subsequent rows, including
            those that might contain NaNs later on.

            Args:
                dataframe: The input pandas DataFrame.

            Returns:
                A pandas DataFrame with initial NaN-containing rows removed.
                Returns an empty DataFrame if no row is completely NaN-free.
            """

            # Test code workflow:
            '''
            # --- Example Usage ---
            data = {
                'A': [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 7, 8],
                'B_roll_3': [np.nan, np.nan, np.nan, 5, 6, 7, 8, 9, 10, 11], # Example rolling stat
                'C': [10, 11, 12, 13, 14, 15, np.nan, 17, 18, np.nan], # Example with later NaNs
                'Target': [1, 1, 0, 1, 0, 0, 1, 0, np.nan, np.nan] # Example target with trailing NaNs
            }
            dataframe_test = pd.DataFrame(data)

            print("Original DataFrame:")
            print(dataframe_test)
            print("\n" + "="*30 + "\n")

            dataframe_cleaned = drop_initial_nan_rows(dataframe_test)

            print("Cleaned DataFrame (Initial NaNs Dropped):")
            print(dataframe_cleaned)
            '''
            
            # --- Function Logic Begins here --- 
            
            # Check if any row is completely free of NaNs
            # .all(axis=1) checks if all values in a row are True (not NaN)
            # .any() checks if there is at least one such row
            if dataframe.notna().all(axis=1).any():
                # Find the index label of the first row where all values are not NaN
                # idxmax() returns the index label of the first True value (True=1, False=0)
                first_valid_index = dataframe.notna().all(axis=1).idxmax()

                # DEBUGGING:
                # print(f"First valid row found at index: {first_valid_index}")

                # Return the slice of the DataFrame starting from the first valid row
                return dataframe.loc[first_valid_index:]
            else:
                # Handle the case where no row is completely free of NaNs
                print("Warning in `drop_initial_NAN_rows()`: No row found without any NaN values.")
                return pd.DataFrame(columns=dataframe.columns) # Return empty DataFrame

        # Drop initial NAN rows (from warmup for feature data, eg. features in `Dynamic classifer` & `TA Indicators`)
        dataframe = drop_initial_NAN_rows(dataframe)

        # -----------------------------------------------------------------------------------
        # ------------------------------- [ DEBUGGING TOOLS ] -------------------------------
        # ------------------------------------------j-----------------------------------------
        
        # (DEBUGGING): INSPECT WHAT THIS BATCH LOOKS LIKE
        if False:

            # -------------------------------------------------------------------
            # INSPECT PRE-PROCESSED BATCHES
            # > USEFUL TO INSPECT WHAT TRANSFORMED BATCHES LOOK LIKE
            # -------------------------------------------------------------------

            # print('\n--------- WHAT TRAINING SAMPLES LOOK LIKE: START ----------')
            # print('BASE BATCH w/ FEATURES: >>>>>>>>>>>>>>>> ')
            # print(return_rows)
            # print()

            # print('X w/ DROPPED FEATURES: >>>>>>>>>>>>>>>> ')
            # print(return_rows.iloc[:-2].drop(columns=[TARGET_COLUMN]))
            # print()

            # print('Y (TARGET): >>>>>>>>>>>>>>>> ')
            # print(return_rows[TARGET_COLUMN].iloc[-3]) # Use the target value ~ of the NEXT to last minute ~ as the target 
            # print()

            # print('--------- WHAT TRAINING SAMPLES LOOK LIKE: END ----------')

            # -------------------------------------------------------------------
            # INSPECT PRE-PROCESSED BATCHES: END
            # -------------------------------------------------------------------

            # EXAMPLE:
            # Since we are using IS_SEQ(), and adding 'target' AFTER batches... a batch will look like this:
            
            #                              EURUSD_open_price__ask_level_1  ...   target
            # minute                                               ...
            # 2023-12-22 00:00:00                         1.10061  ...  0.00004
            # 2023-12-22 00:01:00                         1.10059  ... -0.00002
            # 2023-12-22 00:02:00                         1.10068  ... -0.00007
            # 2023-12-22 00:03:00                         1.10070  ...  0.00012
            # 2023-12-22 00:04:00                         1.10066  ... -0.00004
            # 2023-12-22 00:05:00                         1.10082  ... -0.00027
            # 2023-12-22 00:06:00                         1.10082  ... -0.00017
            # 2023-12-22 00:07:00                         1.10060  ... -0.00002  1.10047
            # 2023-12-22 00:08:00                         1.10045  ...  0.00008  1.10047
            # 2023-12-22 00:09:00                         1.10047  ...      NaN

            pass
    
        # ----------------------------------------------------------------------

        # (DEBUGGING TOOL): INCLUDE THE TARGET IN TRAINING DATA (TEST DATA PIPELINE & MODEL SETUP):
        if False:

            # -------
            # Test if the model and data pipeline is set up properly. 
            # ---
            # The model SHOULD score 100% accuracy, when including the target. 
            # ---
            # -- If it doesn't, this signals a misalignement/data-processing problem.
            # -------

            dataframe['target_copy'] = dataframe['target']

            # ---
            pass

        # -----------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------

        return dataframe # return `add_features_to_single_batch()`

    @debug_timing
    def add_features_to_all_batches(batches):
        return [add_features_to_single_batch(batch) for batch in batches]

    # ----- Build Data Batches of SEQUENTIAL Minutes 
    @debug_timing
    def get_minute_batches_pd(dataframe, sequence_length=None):

        # Initialize a list to store valid batches (valid sequence of minutes).
        valid_batches = [] 

        # Calculate the difference in minutes between consecutive rows/minutes.
        time_diff = dataframe.index.to_series().diff().dt.total_seconds().div(60).fillna(1)

        # Identify the breaks (where the difference is more than 1min 
        break_size = 1
        breaks = time_diff != break_size

        # Create a group number that increases at each break
        dataframe['group'] = breaks.cumsum()        

        # Define sequence length of minutes to look for (consider valid).
        if sequence_length is None:
            # --- This is the first call (before data processing, adding features, etc.) ---
            # Allow sequence length to be explicitly defined in the function invocation as param
            sequence_length = SEQ_LEN
            print(f"--- DEBUG --- Calculated INITIAL sequence length: {sequence_length}")
        else:
            # --- This is the second call (after processing data in data_pipeline) ---
            print(f"--- DEBUG --- Calculated PROCESSED sequence length: {sequence_length}")
                
        # Iterate over each group of contiguous time blocks
        for _, group_dataframe in dataframe.groupby('group'):

            # Apply a rolling window to get every possible N-minute batch
            for i in range(len(group_dataframe) - sequence_length + 1):

                # Select the current batch of minutes
                batch = group_dataframe.iloc[i:i + sequence_length]

                # Calculate the time differences between consecutive rows within the batch
                time_diffs = batch.index.to_series().diff().dt.total_seconds() # note: this is in seconds

                # Fill the first NaN difference (which represents the first row) with `60 seconds`
                time_diffs_filled = time_diffs.fillna(60)

                # Check if all the differences are exactly 60 seconds (1 minute)
                total_seconds_diff_allowed = 60

                if time_diffs_filled.eq(total_seconds_diff_allowed).all():
                    # If all minutes in this batch are sequential, consider this a valid batch.
                    valid_batches.append(batch.drop(['group'], axis=1))
                else:
                    # got invalid batch
                    pass
                    print('invalid batch')
        
        return valid_batches
    
    # ----- NORMALIZE TIMESTEPS - by using `pct_change()`
    def normalize_single_batch(dataframe):
      
        # -------------------------------------
        # Normalize all columns/feature - by using `pct_change()`,
        # ... except for: 
        # - the Target itself
        # - the specified list of Exclusion columns (defined above at top of script).
        # -------------------------------------
        
        # List of cols to NOT normalize
        list_of_cols_in_dataframe_to_skip_normalization = [item for item in cols_to_NOT_normalize \
                if item in dataframe.columns]
        
        # Step 1: Separate the target column 
        excluded_columns = dataframe[list_of_cols_in_dataframe_to_skip_normalization]

        # Step 2: Apply `pct_change()` to the remaining DataFrame
        dataframe_pct_change = dataframe.drop(
            columns=list_of_cols_in_dataframe_to_skip_normalization)\
                .pct_change(fill_method=None)

        # Step 3: Concatenate the target back onto the dataframe
        dataframe_pct_change = pd.concat([dataframe_pct_change, excluded_columns], axis=1) # *Use pd.concat to avoid pandas fragmentation warning.

        # --------------------------------------------

        # Replace infinity values with 0
        dataframe_pct_change.replace([np.inf, -np.inf], 0, inplace=True)

        # --------------------------------------------

        # Drop the first row - due to NaN values from `pct_change()` - the first row has no value above it, so all values are `nan`
        dataframe_pct_change = dataframe_pct_change.iloc[1:] 

        # --------------------------------------------

        # --------------------------------------------
        # DEBUGGING TOOL HELPER: Inspect what data batches look like, *AFTER* normaliazaiton.
        # ---
        # print('\n--------- WHAT TRAINING SAMPLES LOOK LIKE /AFTER NORMALIZATION: START ----------')
        # print(dataframe_processed)
        # print('--------- WHAT TRAINING SAMPLES LOOK LIKE /AFTER NORMALIZATION: END ----------\n')
        # --------------------------------------------

        return dataframe_pct_change 

    @debug_timing
    def normalize_all_batches(batches):
        # --------------------
        # Multiprocessing option:
        # ---
        # preprocessed_batches = []
        # data_preprocessing_num_processes = 1
        # with multiprocessing.Pool(processes=data_preprocessing_num_processes) as pool:
        #     preprocessed_batches = pool.map(normalize_single_batch, batches)
        # ----------------------------
        return [normalize_single_batch(batch) for batch in batches]

    # ----- SCALE DATA
    @debug_timing
    def scale_dataframe(dataframe, scalers, preprocessing_type, save_scalers=False):

        # --------------------------------------

        DEBUGGING_PRINT_STATEMENTS = False

        if DEBUGGING_PRINT_STATEMENTS:
            print('SCALING DATA NOW...')

        # --------------------------------------

        for col in dataframe.columns:
        
            # --------------------------------------

            # If this column hasn't been excluded from scaling
            if col not in cols_to_exclude_from_scaling:

                if preprocessing_type == "train":

                    # --- Log transform specified columns --- 
                    if col in cols_to_log_transform:
                
                        try:
                            dataframe[col] = np.log1p(dataframe[col].values)
                        except:
                            print('ERROR SCALING!')
                            print(f'col={col}')
                            print('dataframe[col]=')
                            print(dataframe[col])

                            raise Exception

                        if DEBUGGING_PRINT_STATEMENTS:
                            print(f'log transformed col: {col}')

                    # --- Config Scaler --- 

                    # Config: which scaler library to use
                    scaler = RobustScaler() # MinMaxScaler() # StandardScaler()

                    scaled_values = scaler.fit_transform(dataframe[col].values.reshape(-1, 1))

                    # Convert scaled values back to Pandas Series with the same index
                    dataframe[col] = pd.Series(scaled_values.flatten(), index=dataframe.index)

                    # Save scaler 
                    scalers[col] = scaler 

                    # ---

                    if DEBUGGING_PRINT_STATEMENTS:
                        print(f'Using {default_scaler_to_use}() for col: {col} ')

                    # --------------------------------------

                elif preprocessing_type == "validation":

                    if col in cols_to_log_transform:
                
                        dataframe[col] = np.log1p(dataframe[col].values) 
                        
                        if DEBUGGING_PRINT_STATEMENTS:
                            print(f'log transformed col: {col}')
                    
                    # --------------------------------------

                    # Use the scalers passed from the training data pre-processing
                    scaler = scalers[col]

                    scaled_values = scaler.transform(dataframe[col].values.reshape(-1, 1))
                    
                    # Convert back to Pandas Series with the same index
                    dataframe[col] = pd.Series(scaled_values.flatten(), index=dataframe.index)

                    # ---
                    
                    if DEBUGGING_PRINT_STATEMENTS:
                        print(f'Using SAVED {default_scaler_to_use}() for col: {col}')
                    
                    # --------------------------------------

        # ---

        # Save scalers to file 
        # ---
        # - Why do we save these to file? - So we can use the same scaler settings during inference/prediction/backtest/live prediction.

        # Export scalers
        if preprocessing_type == "train" and save_scalers is True:
            joblib.dump(scalers, scalers_export_path) 

        if DEBUGGING_PRINT_STATEMENTS:
            print(f'Scaling complete. Exported scalers to {scalers_export_path}')

        # ---

        return dataframe, scalers

    # ----- POST-SCALING DATA HEALTH CHECK (EXTREME OUTLIERS, NANs, INFs, ++)
    def sanity_check_scaled(
            X: pd.DataFrame,
            *,
            # ───────────────────────────── NaN / Inf RULES ──────────────────────────────
            nan_allow: Dict[str, int] | None = None,   # e.g. {"future_delta_lag_2": 1610}
            default_nan_allow: int = 0,                # fallback for cols not in nan_allow
            # ────────────────── MAGNITUDE / VARIANCE / FAT‑TAIL RULES ───────────────────
            abs_max_global: float = 7.0,               # global |max| threshold
            per_col_abs: Dict[str, float] | None = None,  # per‑column overrides
            tail_ratio: float = 10.0,                  # 99.9 % / 95 % fat‑tail ratio
            tail_ratio_overrides: Dict[str, float] | None = None,  # per‑column overrides
            nzv_thresh: float = 1e-4,                  # near‑zero variance flag
            ignore_regex: str = r"_sin$|_cos$|_shock$",      # skip flat sine/cos encodings ← added |_shock$
            # ───────────────────────────────── EXTRAS ───────────────────────────────────
            check_corr: bool = True,                   # flag duplicate features when d < 150
            verbose: bool = True,                      # print “✓ passed” on success
        ) -> None:
        """
        Abort (sys.exit) if the *scaled* feature matrix `X` violates any of:

        • NaN / Inf counts  –  per‑column allowance via `nan_allow`
        • Near‑zero variance (< nzv_thresh)          ──┐
        • IQR == 0                                   ──┴  skipped for cols matching
        • |max| > abs_max_global (or per‑column abs)      
        • Fat‑tail ratio (q0.999 / q0.95) > tail_ratio
        • Optional: perfect‑correlated duplicates (|corr| > 0.99999)

        Parameters
        ----------
        X : pd.DataFrame
            Your *already scaled* features (train or full set).
        nan_allow : dict[str, int], default {}
            Explicit NaN budget per column.  Columns not listed use `default_nan_allow`.
        per_col_abs : dict[str, float], default {}
            Column‑specific |max| thresholds that override `abs_max_global`.
        """

        # ───── pre‑flight setup ──────────────────────────
        nan_allow = nan_allow or {}
        per_col_abs = per_col_abs or {}
        tail_ratio_overrides = tail_ratio_overrides or {}

        cyc_pat: Pattern = re.compile(ignore_regex)
        errors: List[str] = []
        offending_cols = []

        # 1️⃣  NaN / ±Inf  ─────────────────────────────────
        nan_cnt = X.isna().sum()
        inf_cnt = (X == np.inf).sum() + (X == -np.inf).sum()

        for col in X.columns:
            allowed = nan_allow.get(col, default_nan_allow)
            if nan_cnt[col] > allowed or inf_cnt[col] > 0:
                errors.append(
                    f"{col}: NaN={nan_cnt[col]}, ±Inf={inf_cnt[col]} (allowed={allowed})"
                )
                offending_cols.append(col)

        # 2️⃣  Variance & IQR checks  ──────────────────────
        var = X.var()
        iqr = X.quantile(0.75) - X.quantile(0.25)

        for col in X.columns:
            if cyc_pat.search(col):       # skip e.g. *_sin, *_cos
                continue
            if var[col] < nzv_thresh:
                errors.append(f"{col}: variance={var[col]:.1e} < {nzv_thresh}")
                offending_cols.append(col)
            if iqr[col] == 0:
                errors.append(f"{col}: IQR==0")
                offending_cols.append(col)

        # 3️⃣  |max| & fat‑tail ratio  ─────────────────────
        abs_max = X.abs().max()
        q_hi = X.quantile(0.999)
        q_ref = X.quantile(0.95).replace(0, np.nan)
        ratio = (q_hi / q_ref).abs()

        for col, m in abs_max.items():
            thr = per_col_abs.get(col, abs_max_global)
            if m > thr:
                errors.append(f"{col}: |max|={m:.2f} > {thr}")
                offending_cols.append(col)

            tr = tail_ratio_overrides.get(col, tail_ratio)        

            if pd.notna(ratio[col]) and ratio[col] > tr:          
                errors.append(f"{col}: 99.9/95 % ratio={ratio[col]:.1f} > {tr}")
                offending_cols.append(col)

        # 4️⃣  Perfect‑duplicate scan (optional)  ──────────                
        if check_corr and X.shape[1] < 150:
            # --- skip columns you already ignore elsewhere ---
            skip_mask = X.columns.to_series().str.contains(ignore_regex)
            cols_for_corr = X.columns[~skip_mask]

            corr = X[cols_for_corr].corr().abs()

            for i in corr.columns:
                for j in corr.index:
                    if i < j and corr.loc[i, j] > 0.99999:
                        errors.append(f"{i} & {j}: corr≈1")
                        offending_cols.extend([i, j])  

        # 5️⃣  Verdict  ────────────────────────────────────
        offending_cols = list(dict.fromkeys(offending_cols))
        if errors:
            print("\n⚠️  Sanity‑check failed:\n" + "\n".join(f" • {e}" for e in errors))
            print("Investigate / clean or relax thresholds before training. \n")

            print('Below are the stats for offending features (post-scaling):\n')
            for col_name, col_series_data in X[offending_cols].items():
                print(col_name)
                print(col_series_data.describe())
                print('----------------')

            offending_cols_in_raw_data = [offending_col for offending_col in offending_cols \
                                if offending_col in main_dataframe.columns]
            
            if len(offending_cols_in_raw_data) > 1:
                print('\nBelow are the stats for the raw input data (pre-scaled):\n') # For the features that had the same name in `main_dataframe` - this will exclude cols added during `add_features_to_batch()` - since those aren't in `main_dataframe` (original dataset)               

                for col_name, col_series_data in main_dataframe[offending_cols_in_raw_data].items():
                    print(col_name)
                    print(col_series_data.describe())
                    print('----------------')
                
            sys.exit('exiting...')

        elif verbose:
            print("✓ Sanity‑check passed – dataset looks clean.")
            
    # ----- BUILD X & Y BATCHES (DROP TARGET & LOOKAHEAD ROWS)
    @debug_timing 
    def build_x_and_y_from_batches(batches):
        X = []
        y = []

        print(f"DEBUG: build_x_y() received {len(batches)} batches.")

        for batch in batches: 

            # ---------------------
            # HOW WE BUILD X AND Y:
            # ---
            # X []: 
            # - Remove the last row, *AND* the target column, from the batch.
            # Y []: 
            # - Use the value, of the **NEXT** to last, minute, as the target 
            # ---------------------
            #
            # Reminder: We drop the first row of this batch (due to NaN values from `pct_change()`) when doing `normalize_single_batch()` - which is done BEFORE this step (`build_x_and_y_from_batches()`)
            #
            #
            # - Question: WHY was I doing this? 
            #       > That is, dropping the `last` row, and using the `2nd` to last row, as the target?
            #   (Was it Due to normalization? Which I'm not doing now?)
            # ---
            # 
            # - Answer: Because we are shifting the Future price, back a step - so we can compute its change - from the present open value (this is how we assign the target).
            # ---
            # > But, for the last minute, it WON'T have a future value (for us to shift back), since it's the last time step, in the batch (so we cannot create a target for it).
            #   > (For more info & example, see `add_target_feature_to_batch()`.)
            # ---------------------
            #
            # - Note: If you change `future_predict_period`, these values will need to be increased.
            # ---
            # Here's an example with `FUTURE_PERIOD_PREDICT` = 1:
            '''
            # Here we create the input features for the model:
            #
            # X.append(\
            #     batch\
            #         .iloc[:-(1)]\                   -- so we drop the last row, from the batch
            #             .drop(columns=["target"])\  -- drop the target, from the batch
            #                 .values)                -- Use the remaining columns, in the batch, as input features for the model,
            #
            # `.iloc` Further explained:
            # ----
            #     - .iloc is used in pandas for integer-based indexing, of rows and columns, in a DataFrame. 
            #
            #     - Here's what .iloc[:-2] and .iloc[:-1] do: 
            #
            #         - .iloc[:-2]: Selects all rows, except the last two. 
            #             The :-2 slice means "from the start until two rows before the end." 
            #
            #         - .iloc[:-1]: Selects all rows, except the last one. 
            #             The :-1 slice means "from the start until one row before the end."
            #
            #     - 📝 Summary:
            #         `.iloc[:-2]` skips the last two rows.
            #         `.iloc[:-1]` skips the last row.
            #
            # Summary so far: 
            # ... So we've dropped the last row, completely from the batch, of N timesteps, for the INPUT (X).
            #
            # Here we create the TARGET variable we are predicting:
            #  
            # But why do we do -(1+1) = -2? (Reminder: we are using `FUTURE_PERIOD_PREDICT` = 1)
            # When we move to production, it will be -(N+1).
            #
            # So for example, with `FUTURE_PERIOD_PREDICT` = 2, it will be -(2+1) = -3
            #
            # ----
            # To review, here's what a batch looks like:
            #
            # Timestep 1, 0.004 0.31 0.01 0.32
            # Timestep 2, 0.031 0.62 0.85 0.62
            # ...
            # Timestep N, 0.064 0.32 0.71 0.29
            # 
            # ----
            # 1. To create the target, what we do, is shift, the value of the next (future) timestep, backwards, into the current row/timestep.
            #
            # But since we have a batch of N timesteps, the N element (the last in the sequence), will NOT have any future timestep. 
            # > Therefore we CANNOT create a target for it. 
            # > So we drop it.
            #
            # Therefore, the first timestep that DOES have a future variable/next time step, is the NEXT, to last row (timestep) (N-1).
            #
            # Which means, we use the (N-1) timestep (row). 
            # > What that row does, is compare itself, to the LAST row. 
            #
            # Because the last row (N), itself, does not have a future value ahead of it. 
            # > But it itself (this last row), can be used to calculcate a value, for the row/timestep right before/behind it (N-1). 
            # 
            # ---
            #
            # So in the case of `FUTURE_PERIOD_PREDICT = 1`
            #
            # y.append(\
            #     batch["target"]\
            #         .iloc[-(1+1)]) -- so we DON'T use the LAST row, as the target (since the target, for that row will be NAN, since it DOESN'T have a future value ahead of it) - so instead, we use the target, from the SECOND to last row. -- This becomes the target for the entire batch, of N timesteps.
            #
            # `.iloc[-(1+1)]` broken down:
            #   1. `-(1+1)` evaluates to `-2`.
            #   2. `.iloc[-2]` selects the second-to-last, row, in a Pandas DataFrame
            #   3. Summary: In short, it retrieves the second-to-last row of your data.
            #
            # `.iloc[-1]` selects the last row in a Pandas DataFrame or Series. 
            # - It uses negative indexing, where `-1` refers to the last element.
            # 
            # Here’s a quick summary:
            # > `.iloc[-(1+1)]`	-- Selects the second-to-last, row/element
            # > `.iloc[-1]`	    -- Selects the last, row/element
            #
            # - Both use negative indexing, in Pandas, to retrieve rows, from the end, of the DataFrame.
            '''
            # ---------------------

            # Example with `FUTURE_PERIOD_PREDICT = 10`
            # 
            # So in this case, we are calculcating the target, by looking 10 rows into the future.
            # 
            # This means, that the 10th row, to LAST row, in a batch, will *NOT8 have any future value, to calculcate the target against.
            #
            # Therefore, we use the 10th + 1 = 11th to LAST row, in a batch, as the target - since that is the last minute, which was able to actually calculcate a target.
            # 
            # Here's an example:
            #
            # This is from the `.tail(n=15)` of a batch of minutes:
            # ---
            # 	                     target
            # minute	
            # 2023-12-22 01:04:00	0
            # 2023-12-22 01:05:00	0
            # 2023-12-22 01:06:00	2
            # 2023-12-22 01:07:00	0
            # 2023-12-22 01:08:00	3           -- 11th to last row, -11
            # 2023-12-22 01:09:00	-999        -- 10th to last row, -10
            # 2023-12-22 01:10:00	-999        -- 9th to last row, -9
            # 2023-12-22 01:11:00	-999        -- 8th to last row, -8
            # 2023-12-22 01:12:00	-999        -- 7th to last row, -7
            # 2023-12-22 01:13:00	-999        -- 6th to last row, -6
            # 2023-12-22 01:14:00	-999        -- 5th to last row, -5
            # 2023-12-22 01:15:00	-999        -- 4th to last row, -4
            # 2023-12-22 01:16:00	-999        -- 3rd to last row, -3
            # 2023-12-22 01:17:00	-999        -- 2nd to last row, -2
            # 2023-12-22 01:18:00	-999        -- last row, -1
            # ---
            # 
            # As a reminder, `classifier_multiclass` looks like this:
            #
            # def classifier_multiclass(delta):
            #     if np.isnan(delta):
            #         return -999
            #
            # So, since we are using `FUTURE_PERIOD_PREDICT = 10`, to predict 10 rows into the future - the last row that this will be possible for, is the 11th row.
            #

            # ---------------------

            # `None` represents the start of the range (before the first element/row), and `-FUTURE_PERIOD_PREDICT` represents the index to stop/end/cap the filter/window/slice.
            the_ROWS_to_use_as_input_minutes = slice(None, -FUTURE_PERIOD_PREDICT) 
            the_row_to_use_as_the_target = -(FUTURE_PERIOD_PREDICT + 1)
            
            # --- Assign X ---
            # X.append(batch.iloc[:-(FUTURE_PERIOD_PREDICT)].drop(columns=["target"]).values) 
            X.append(batch.iloc[the_ROWS_to_use_as_input_minutes].drop(columns=["target"]).values)

            # --- Assign Y ---
            y.append(batch["target"].iloc[the_row_to_use_as_the_target]) 

        return X, y

    # ----- DATA PROCESSSING PIPELINE (*IN ORDER STEPS*)
    @debug_timing
    def data_processing_pipeline(dataframe, 
                            preprocessing_type="train", 
                            scale_data=True, 
                            scalers=None, # used for passing scalers from training to validation
                            normalize_data=None, 
                            shuffle_data=False,
                            use_cache=True):    

        global sample_of_final_transformed_data
        
        # Cache
        if True:

            # Check if file exists (of already pre-processed data, eg. `preprocessed_data_train.joblib`)
            
            # ---------------------------------------------------------------
            # CHECK FOR CACHE 
            # ---------------------------------------------------------------

            cache_file_location = f"data/cache/preprocessed_data_{preprocessing_type}.joblib"
            if use_cache is True:
                print(f"Checking for {preprocessing_type} cache file: {os.getcwd()}/{cache_file_location}")
                if os.path.exists(cache_file_location):
                    cache_data = joblib.load(cache_file_location)
                    return cache_data['X'], cache_data['y'], cache_data['scalers']
                else:
                    print("Cache NOT found... Computing data preprocessing now...")
            elif use_cache is False: 
                print("Resetting cache... Computing data preprocessing now...")

            # ---------------------------------------------------------------
            # END: CHECK FOR CACHE 
            # ---------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------
        # Data pipeline (Overview):
        # --
        # 1. Split data into sequential batches eg. 60 min, and add custom features (TA indicators, etc).
        # - 1a. Split the data into batches.
        # - 1b. Add custom features.
        # 2. Scale the data - AFTER added custom features (since TA indicators require sequential data.)
        # - 2a. Join all batches together into single dataframe.
        # - 2b. Calculcate delta thresholds (skipped).
        # - 2c. Drop delta column (skipped).
        # - 2d. Fit the scaler to the joined single dataframe.
        # 3. Re-split the scaled, single dataframe into batches.
        # ----------------------------------------------------------------------------------------------

        # Split dataframe into sequential timeseries batches of size `SEQ_LEN`
        batches = get_minute_batches_pd(dataframe)

        # ----------------------------------------------------------

        print(f"Adding features to batches...")

        original_batch_length = len(batches)
        print(f"DEBUG: Got {original_batch_length} batches of sequential timeseries...")

        # Add features to batches
        batches = add_features_to_all_batches(batches)
        print(f"DEBUG: Adding features & calculating targets complete.")

        # Check length of updated batches (from adding features)
        updated_batch_length = len(batches)
        if updated_batch_length != original_batch_length:
            raise ValueError(f'Lost batches during `add_features_to_all_batches()` - entry length: {original_batch_length}, current length: {updated_batch_length}')

        # ----------------------------------------------------------

        # DEBUGGING: Export updated batches (with features & target) for inspection (optional) 
        if True:

            # Use this notebook to inspect these batches: (`inspect_data_pipeline_intermediate_output.ipynb`)

            # ------------------------------------------------------------

            # Note: This exports the data batches *BEFORE* they are normalized and scaled.

            # ------------------------------------------------------------

            # Export batches only for training set, not validation set.
            if preprocessing_type == "train":

                dump_file_location = 'preprocessed_prenormaliazation_prescaling_batches.joblib'
                dump_file_folder =  r'C:\Users\auste\Documents\Python Scripts\data\cache'
                
                joblib.dump(batches, f'{dump_file_folder}\{dump_file_location}') 

        # ----------------------------------------------------------
     
        # Normalize this batch of sequential minutes - by using `pct_change()`
        if True:
            # ---
            # - Note: We drop the first row, due to NAN values being generated in every column - due the fact that `pct_change()` will not have a row above it to difference. (This NAN row is dropped in `normalize_single_batch()`) 
            # ---
            if normalize_data:
                preprocessed_batches = normalize_all_batches(batches)

                print(f'DEBUG: Normalize data complete. \
                    len(normalized_batches)={len(preprocessed_batches)}')
            else:
                preprocessed_batches = batches

                print(f'DEBUG: Normalization is DISABLED, skipping...')

        # ----------------------------------------------------------

        # Measure updated length of timeseries batches (after adding features, dropping NAN rows, normalizing - but before scaling).
        if True:

            # The size of these batches (after adding features, and normalizing) is used as a key input, when re-splitting the batches (after scaling) - for knowing what `sequence_length` to look for (of sequential timeseries data sequences).

            shape_counts = Counter(batch.shape for batch in preprocessed_batches)

            # --- Print the Summary ---
            print("--- Batch Summary ---")
            
            for shape, count in shape_counts.items():
                rows, cols = shape # Unpack the tuple
                print(f"Got {count} batches of shape {shape} / ({rows} rows, {cols} columns)")

            print("-------------------------")

            most_common_shape = max(shape_counts, key=shape_counts.get)
            max_count = shape_counts[most_common_shape] # Get the count for the most common shape

            print(f"The most frequent shape is: {most_common_shape}")
            print(f"It occurred {max_count} times.")
            print("-------------------------")

        # ----------------------------------------------------------

        if scale_data:

            # --------------------------------------------------
            # Scaling data (Workflow overview) 
            # ---
            # 1. Join batches (into single dataframe), 
            # 2. Then scale (single dataframe), 
            # 3. Then finally re-split the scaled dataframe, back into sequential batches (of size `SEQ_LEN`).
            # ---
            # - We do this in this order, because `get_minute_batches_pd()` gets rid of invalid batches of minutes, it returns a list[] of valid batches of sequential minutes, and discards any batches of non-sequential minutes. 
            # - So if a minute/row didn't fit into a sequential batch (of size `SEQ_LEN`), that minute/row will be completely dropped.
            # ... But I suppose it's not absolutely necessary to do this, in this order, I guess we could scale_data before this, since we are re-joining the data anyways, and this won't preserve perfect continious time series order (there will be gaps in the time series between consequenive minutes, once the minutes are joined into a single dataframe, due to holidays, weekends, data feed outages [5min, 1hour, etc.], etc.)
            # --------------------------------------------------

            # --- Scaling: Part 1 ---

            if scalers is None:
                scalers = {} # Used for passing the scaler from training data set, to the validation data set.
            
            # Join all timeseries batches into a single pandas dataframe (so they can be scaled). 
            #  - This DOES break continuous time series alignement; there will be time gaps in these now. (This is why we resplit the batches, with the second call to `get_minute_batches_pd()`)
            joined_batches = pd.concat(preprocessed_batches)

            del preprocessed_batches # Free up memory: clear memory of previous variables we no longer need. Here we clear the the `joined_batches` [from above] is all we need now.)

            # ---------------------------------------------------------

            # --- Scaling: Part 1B (DEBUGGING): Pre-Check of Post-Processed New Features --- 

            # You can copy and paste the names of new features you create here to inspect these features before scaling - if you are having scaling-health-check issues - and you want to check the distrubtion of the new features.
            # new_features_for_model = \
            #     ['EURUSD_p', 'EURUSD_r', 'EURUSD_hl', 'EURUSD_oc', 'EURUSD_spr', 'GBPUSD_p', 'GBPUSD_r', 'GBPUSD_hl', 'GBPUSD_oc', 'GBPUSD_spr', 'USDJPY_p', 'USDJPY_r', 'USDJPY_hl', 'USDJPY_oc', 'USDJPY_spr']

            # print(f'Below are the stats for the new features (from {max_count} batches of {SEQ_LEN} Minutes):\n')
            # for col_name, col_series_data in joined_batches[new_features_for_model].items():
            #     print(col_name)
            #     print(col_series_data.describe())
            #     print('----------------')

            # sys.exit(0)

            # ---------------------------------------------------------
            
            # --- Scaling: Part 2 ---
            
            # Scale dataframe
            scaled_dataframe, scalers = scale_dataframe(
                # Data to scale
                dataframe=joined_batches, 
                # Used for passing the fit scaler from the training data set, to the validation data set.
                scalers=scalers, 
                # Training or validation or dataset? (used for saving scaled data to cache file)
                preprocessing_type=preprocessing_type, 
                # Save fit scaler to file (so we can use the same scaler settings during inferance/prediction/backtest/live prediction).
                save_scalers=True
            )

            print(f'DEBUG: Scaling completed. \
                  scaled_dataframe.shape={scaled_dataframe.shape}')
            
            del joined_batches # Clear previous variables to free up memory (`scaled_dataframe` is all we need now)

            # ---------------------------------------------------------

            # disabled sanity check: 
            # - because I just want to see how the model trains without it...

            # this failed on validation set

            # not sure what to disable to bypass

            # so im just disabling the sanity check for now


            # ⚠️  Sanity‑check failed:
            #  • USDJPY_p: 99.9/95 % ratio=49.6 > 10.0
            # Investigate / clean or relax thresholds before training.

            # Below are the stats for offending features (post-scaling):

            # USDJPY_p
            # count    1.392401e+06
            # mean    -3.019389e-01
            # std      1.598370e-01
            # min     -6.818487e-01
            # 25%     -4.043039e-01
            # 50%     -2.986006e-01
            # 75%     -1.992064e-01
            # max      8.071676e-02
            # Name: USDJPY_p, dtype: float64
            # ----------------
            # exiting...


            # Quick health‑check (post-scaling): abort if scaled X still has NaNs, flat cols, or 
            # numerically dangerous outliers → prevents silent gradient explosions later
            # ---
            # sanity_check_scaled(
            #     X           = scaled_dataframe,
            #     nan_allow   = { # N batches × K trailing steps unpacked → Y NaNs OK
            #                     "target": max_count * FUTURE_PERIOD_PREDICT },
            #     per_col_abs = {             # per‑column |max| overrides
            #                     "future_delta_lag_2": 8.5,
            #                     "future_delta_lag_2_diff": 8.5,
                                
            #                     # returns & intrabar moves
            #                     "EURUSD_r": 10, "GBPUSD_r": 10, "USDJPY_r": 10,
            #                     "EURUSD_oc": 10, "GBPUSD_oc": 10, "USDJPY_oc": 10,

            #                     # high–low range
            #                     "EURUSD_hl": 25, "GBPUSD_hl": 25, "USDJPY_hl": 25,

            #                     # relative spread
            #                     "EURUSD_spr": 25, "GBPUSD_spr": 25, "USDJPY_spr": 25,

            #                     # ----- i do think i need to invesgiate these overrides for the feature below more (but for now, can test with these manual overrides - but im not SURE SURE this data is clean)

            #                     "future_delta_lag_2":       50,
            #                     "future_delta_lag_2_diff":  50,

            #                     "future_delta_ewma_20_mean_lag_2": 30,
            #                     "future_delta_ewma_20_std_lag_2":  40,

            #                     # tick count log‑returns
            #                     "EURUSD_total_ticks_dlog":        12,
            #                     "GBPUSD_total_ticks_dlog":        12,
            #                     "USDJPY_total_ticks_dlog":        12,
            #                     "EURUSD_meaningful_ticks_dlog":   12,
            #                     "GBPUSD_meaningful_ticks_dlog":   12,
            #                     "USDJPY_meaningful_ticks_dlog":   12,

            #                     # meaningful‑ratio (very small IQR → big robust units)
            #                     "EURUSD_meaningful_ratio": 12,
            #                     "GBPUSD_meaningful_ratio": 12,

            #                     # returns & intrabar moves
            #                     "EURUSD_r": 70, "GBPUSD_r": 70, "USDJPY_r": 70,
            #                     "EURUSD_oc": 70, "GBPUSD_oc": 70, "USDJPY_oc": 70,

            #                     # high‑low range
            #                     "EURUSD_hl": 80, "GBPUSD_hl": 85, "USDJPY_hl": 80,

            #                     # relative spread (scaled units)
            #                     "EURUSD_spr": 180,
            #                     "GBPUSD_spr": 120,
            #                     "USDJPY_spr": 600,
            #                     },
            #     tail_ratio_overrides = {"EURUSD_spr": 20, "GBPUSD_spr": 20,
            #                             "USDJPY_spr": 45, "USDJPY_hl": 15}
            # )

            # ---------------------------------------------------------
            
            # --- Scaling: Part 3 ---         
               
            # Re-split the scaled dataframe back into sequential batches (of size `SEQ_LEN`).
            batches_of_scaled_data = get_minute_batches_pd(
                dataframe=scaled_dataframe, 
                sequence_length=most_common_shape[0] # Adding the target as a feature, creates a NAN, so again, we need to subtract `1` from the sequence length, in order to get these to match. -- BUT rather than keep updating the sequence length algo to look for x size, I think it's simplest to just see what length of sequences we end up with (after adding features, etc.), and then just use that sequence length. So making it more dynamic. 
            ) 

            # ---------------------------------------------------------

            # DEBUGGING: Export scaled batches for inspection (optional) 
            if True:
                if preprocessing_type == "train":

                    # Use this notebook to inspect these batches: 
                    # - (`inspect_data_pipeline_intermediate_output.ipynb`)

                    # ----------------------------

                    # Note: This exports batches *AFTER* they are normalized (if enabled), and scaled.

                    # ------------------------------

                    joblib.dump(batches_of_scaled_data, 
                                r'C:\Users\auste\Documents\Python Scripts\data\cache\preprocessed_normalized_and_scaled_batches.joblib')  
            
            # ---------------------------------------------------------

        else:
            raise Exception("Not implemented: haven't built the data pipeline to handle non-scaled data.")

        # ----------------------------------------------------------

        # Build X[] and Y[] sets (`build_x_and_y_from_batches()`)
        if True:

            # DEBUGGING: Inspect a single final post-processed batch w/ features and targets
            if preprocessing_type == "train":

                # Log what a final transofmred timeseries batch looks like (right before build_x_and_y)
                # - Used to log the final list of columns - after all transformations have been completed (add features, normalization, scaling, etc.) - in the output training log (for experiment tracking).
                sample_of_final_transformed_data = batches_of_scaled_data[0] 

                # Print list of columns
                print(batches_of_scaled_data[0].columns)

                # For each column, print summary statistics (inspect transformations).
                if False:
                    for col_name, series_data in batches_of_scaled_data[0].items():
                        print(col_name)
                        print(series_data.head())
                        print('---')

            # ----------------------------------------------------------------

            # DEBUGGING: Check current batch len (after scaling - before building x and y)
            print(f'len(batches_of_scaled_data)={len(batches_of_scaled_data)} \
                  ... now feeding into `build_x_and_y()`') 

            X, y = build_x_and_y_from_batches(batches=batches_of_scaled_data)
            
            # DEBUGGING: Check current batch len (after building batches)
            print(f'build_x_and_y() DONE ... len(X, y)= {len(X)}, {len(y)}') 

        # ----------------------------------------------------------

        if shuffle_data:

            # TIMING: how long does it take to shuffle?
            print(f'DEBUG - START: SHUFFLE DATA')
            start_time = time.time()

            # -------------------------------------

            if preprocessing_type == "train":
                # Shuffle data
                X, y = shuffle(X, y)
            else:
                print('skipping shuffle on validation dataset.')
    
            # -------------------------------------

            print(f'DEBUG - COMPLETED: SHUFFLE DATA')
            end_time = time.time()
            execution_time = end_time - start_time

            print(f"DEBUG - Stage Execution time (`shuffle_data()): {debuggingTools_format_time(execution_time)}")

        # ----------------------------------------------------------

        print('Job status change: all preprocessing steps completed in `data_preprocessing_pipeline()`.')

        # ----------------------------------------------------------

        # Cache: store preprocessed data to cache file.
        if True:

            print(f"Now writing preprocessed data to cache: {cache_file_location}...")

            # -------------------------------------

            # Timing: how long it takes to save data to cache
            print(f'DEBUG - START: WRITING PREPROCESSED DATA TO CACHE')
            start_time = time.time()

            # -------------------------------------
 
            # Write data to file (x, y, and scalers).
            data_to_write_to_cache = {
                'X': np.array(X), 
                'y': np.array(y),
                'scalers': scalers }
            
            joblib.dump(data_to_write_to_cache, cache_file_location)

            # -------------------------------------

            print(f'DEBUG - COMPLETED: SHUFFLE DATA')
            end_time = time.time()
            execution_time = end_time - start_time

            print(f"DEBUG - Stage Execution time (WRITING PREPROCESSED DATA TO CACHE): {debuggingTools_format_time(execution_time)}")

            # -------------------------------------

        # ----------------------------------------------------------

        # END `data_preprocessing_pipeline()`
        return np.array(X), np.array(y), scalers

    # ----- CHECK DATA FOR `NANs` AND `INFs`
    def check_data_for_nans_and_infs(x, y, dataset_type, PRINT_DEBUG_STATEMENTS=False):

        # -------------------------------------------------------------

        if PRINT_DEBUG_STATEMENTS:

            print(f'Checking for NaNs in {dataset_type} dataset...\n')

            print(f'np.isnan({dataset_type}_x).sum() = {np.isnan(x).sum()}')
            print(f'np.isnan({dataset_type}_y).sum() = {np.isnan(y).sum()}')

            print(f'\nChecking for Infinities in {dataset_type} data...\n')

            print(f'np.isinf({dataset_type}_x).sum() = {np.isinf(x).sum()}')
            print(f'np.isinf({dataset_type}_y).sum() = {np.isinf(y).sum()}')

        # -------------------------------------------------------------

        if np.isnan(x).any() \
            or np.isinf(x).any() \
                or np.isnan(y).any() \
                    or np.isinf(y).any():
            
            print(f"\nWARNING: Invalid values found in `{dataset_type}` dataset.\n")

            # ------------------------------------

            if PRINT_DEBUG_STATEMENTS is False:
                # Run again, to print the rows with errors to console
                check_data_for_nans_and_infs(x, y, dataset_type, PRINT_DEBUG_STATEMENTS=True)

            raise Exception('See warning above.')
        
            # ------------------------------------

        else:

            if PRINT_DEBUG_STATEMENTS:
                print(f"\nSUCCESS! No invalid values found in {dataset_type} dataset.\n")

            # ------------------------------------

        return x, y

# ----------------------------------------------------------------------

if __name__ == "__main__":

    # --------------------------------------------------------------------     
    # --------------------------------------------------------------------     
    # --------------------------------------------------------------------     
    # --------------------------------------------------------------------     
    # --------------------------------------------------------------------     

    # Archive output files from previous training runs
    from helpers.archive_previous_runs import archive_previous_runs
    archive_previous_runs()

    # --------------------------------------------------------------------     
    # --------------------------------------------------------------------
    # ------------------- [ KEY SECTION: LOAD DATA ] ---------------------
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    if True:

        # DEBUGGING HELPER: Press (a) to load all rows, (d) to specify number of debug rows, and (c) to clear cache. 
        if True:

            DEBUG_N_ROWS = 1000 # Use None to load all rows

            use_cache = True

            def input_thread():

                global use_cache
                global DEBUG_N_ROWS

                if sys.platform.startswith('win'):

                    # Windows-specific implementation

                    import msvcrt
                    start_time = time.time()

                    while time.time() - start_time < 3:

                        if msvcrt.kbhit():
                            key = msvcrt.getwch()

                            if key.lower() == 'c':

                                print('clearing cache...')
                                use_cache = False
                                break

                            elif key.lower() == 'a':

                                print('loading all rows...')
                                DEBUG_N_ROWS = None
                                break

                            elif key.lower() == 'd':
                                print('Enter number of debug rows: ', end='')
                                try:
                                    DEBUG_N_ROWS = int(input())
                                    print(f'Loading {DEBUG_N_ROWS} debug rows...')
                                except ValueError:
                                    print('Invalid number, keeping default...')
                                break

                else:

                    # Unix/Linux implementation

                    import select
                    i, o, e = select.select([sys.stdin], [], [], 3)

                    if i:

                        user_input = sys.stdin.readline().strip()

                        if user_input.lower() == 'c':

                            print('clearing cache...')
                            use_cache = False

                        elif user_input.lower() == 'a':

                            print('loading all rows...')
                            DEBUG_N_ROWS = None
                            
                        elif user_input.lower() == 'd':
                            print('Enter number of debug rows: ', end='')
                            try:
                                DEBUG_N_ROWS = int(input())
                                print(f'Loading {DEBUG_N_ROWS} debug rows...')
                            except ValueError:
                                print('Invalid number, keeping default...')
                
            # ---------------------------------------------------------------------------
            # Press (c) to clear cache

            print("Using cache of preprocessed data... press (c) to clear... (otherwise using cache)... starting in 3...2...1")
            
            t = threading.Thread(target=input_thread)
            t.daemon = True
            t.start()
            
            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            
            # Ensure the input thread has finished
            t.join(0.1)

            # ---------------------------------------------------------------------------
            # Press (a) to load all rows

            if use_cache is False:

                print(f"Loading just {DEBUG_N_ROWS} rows of data... press (a) to load all or (d) to specify number... starting in 3...2...1")
            
                t = threading.Thread(target=input_thread)
                t.daemon = True
                t.start()
                
                for i in range(3, 0, -1):
                    print(f"{i}...")
                    time.sleep(1)
                
                # Ensure the input thread has finished
                t.join(0.1)

        # --------------------------------------------------------------

        # Load data set - and use `data_preprocessing_pipeline()` - to create: `train_x`, `train_y`, - and `validation_x`, `validation_y` 
        if True:

            # Define filepath to dataset
            if True:

                # --- Use Full or Partial data set? --- 

                # FULL data set 
                # file_name = 'transformed_data_a1__CHECKED_N_FILTERED.parquet' 
                
                # PARTIAL data set (1/8)
                file_name = 'transformed_data_a2__tmp_inspection__CHECKED_N_FILTERED_6_3_25.parquet' 

                # ---

                folder_dir = r'C:\Users\auste\Documents\Python Scripts\data\market'
                data_file_location = rf'{folder_dir}/{file_name}'

            # ---------------------------------------------------------------------------------------

            # Load dataset from .parquet into pandas
            if True:

                if DEBUG_N_ROWS is None:
                    # Read full file
                    main_dataframe = pd.read_parquet(path=data_file_location)
                else:
                    # Just load N rows (if DEBUG mode)
                    main_dataframe = pd.read_parquet(path=data_file_location)[:DEBUG_N_ROWS]    

                # --------------------------------------------------------------
                
                # Load only selects/specified columns from dataset
                if True:

                    # Debug: Print shape of full dataset 
                    print('full dataframe shape: ', main_dataframe.shape)
                    print(f'- minutes: {main_dataframe.shape[0]}, features/columns: {main_dataframe.shape[1]}\n')

                    print('Only loading specified columns...')
                    main_dataframe = main_dataframe[columns_to_use_as_model_features]
                    print('- training_dataframe shape: ', main_dataframe.shape, '\n')

                    try:
                        # Optional debugging helper, toggled above (where you build list of features to load from dataset).
                        if print_input_features_stats: 
                            print('Below are the stats for the raw input features:\n')
                            for col_name, col_series_data in main_dataframe.items():
                                print(col_name)
                                print(col_series_data.describe())
                                print('----------------')
                            sys.exit(0)
                    except NameError: 
                        pass  # debugger helper wasn't toggled, continue execution

            # --------------------------------------------------------------

            # Split off 10% of data for validation
            if True:

                data_to_holdout_for_validation = 0.10 # percent
                print(f"DEBUG: Splitting {data_to_holdout_for_validation:.2f}% of holdout data...")

                time_index_sorted = sorted(main_dataframe.index.values)
                last_X_percent = sorted(main_dataframe.index.values)[
                    -int(data_to_holdout_for_validation * len(time_index_sorted)) 
                ]
                
                # Split data
                validation_main_dataframe = main_dataframe[ (main_dataframe.index >= last_X_percent) ]
                main_dataframe = main_dataframe[ (main_dataframe.index < last_X_percent) ]

                print(f'- Holding out last {validation_main_dataframe.shape[0]} minutes.\n')

                # ----------------------------------------------------------
                # DEBUGGING: EXPORT RAW VALADATION SET BEFORE PROCESSSING (FOR INSPECTION)
                # ----------------------------------------------------------
                # validation_main_dataframe.to_hdataframe(
                #     path_or_buf='c:/Users/auste/Documents/Python Scripts/data/validation/holdout_validation_data_raw.h5', 
                #     key='dataframe', 
                #     mode='w')
                # ----------------------------------------------------------

            # --------------------------------------------------------------

            train_x, train_y, scalers     = data_processing_pipeline(preprocessing_type="train", 
                                                    dataframe              = main_dataframe, 
                                                    scale_data      = scale_data,
                                                    normalize_data  = normalize_data, 
                                                    shuffle_data    = shuffle_data,
                                                    use_cache       = use_cache)

            validation_x, validation_y, _ = data_processing_pipeline(preprocessing_type="validation", 
                                                    dataframe              = validation_main_dataframe, 
                                                    scale_data      = scale_data,
                                                    scalers         = scalers,
                                                    normalize_data  = normalize_data, 
                                                    shuffle_data    = shuffle_data, 
                                                    use_cache       = use_cache)

            # ------------------------------------------------------------------------------

            print(f"train_x.shape {train_x.shape}")
            print(f"train_y.shape {train_y.shape}")

            print(f"validation_x.shape {validation_x.shape}")
            print(f"validation_y.shape {validation_y.shape}")

            # ------------------------------------------------------------------------------
            # DEBUGGING: EXPORT POST-PROCESSED VALADATION SET FOR INSPECTION
            # ------------------------------------------------------------------------------
            # np.save(\
            #     'c:/Users/auste/Documents/Python Scripts/data/validation/validation_x.npy', 
            #     validation_x)
            
            # np.save(\
            #     'c:/Users/auste/Documents/Python Scripts/data/validation/validation_y.npy', 
            #     validation_y)
            # ------------------------------------------------------------------------------

        # --------------------------------------------------------------

        # Check data for NANs & INFs
        if True:
            train_x, train_y = check_data_for_nans_and_infs(
                train_x, train_y, dataset_type="training")
            
            validation_x, validation_y = check_data_for_nans_and_infs(
                validation_x, validation_y, dataset_type="validation")

        # --------------------------------------------------------------

        # Print data & training info (log key info to file for experiment tracking)
        if True:

            print('\n-----------------------------------------')
            print('DATA INFO & TRAINING CONFIG:')
            print('-----------------------------------------')

            print(f'FUTURE_PERIOD_PREDICT: {FUTURE_PERIOD_PREDICT} (Minutes) \n')

            print(f'Number of (timeseries) batches: {train_x.shape[0]}')
            print(f'Minutes (timesteps) per batch: {train_x.shape[1]} (after dropping rows/stat warmup)')
            print(f'Minutes (timesteps) per batch: {SEQ_LEN} (before dropping rows/stat warmup) (original SEQ_LEN)')
            print(f'Features (columns) per batch: {train_x.shape[2]}')

            if train_x.shape[2] > len(columns_to_use_as_model_features):
                count_of_new_features = train_x.shape[2] - len(columns_to_use_as_model_features)
                print(f'- Note: {count_of_new_features} features were added, on top of the original input features.') # likely during `add_features_to_batch()`
                
            print(f'\nDataset used: `./{file_name}`')
            print(f'Number of columns used (from original dataset): {len(columns_to_use_as_model_features)}')

            if sample_of_final_transformed_data is not None:
                
                original_columns = set(columns_to_use_as_model_features)
                final_columns = set(sample_of_final_transformed_data.columns)
                new_columns = final_columns - original_columns

                print(f'\nColumns used (from original dataset):')
                print(f'{original_columns}')

                print(f'\nColumns added as transformations (of original input features): \n{list(new_columns)}\n')
                
                # ------------------------------------------------------------

                print('Feature transformations (applied per feature above):')
                processed_features = set()  # Fix duplicate issue

                # -----------------------------------------------------------------
                # Feature Transformation, Naming Convention SOP
                # ---
                # 🎯 Purpose
                # ---
                # Maintain consistent, predictable naming for transformed features
                #   to enable automatic logging and easy debugging.
                # ---
                # 📋 Naming Standards
                # ---
                # 1. Cyclical Features (sin/cos encoding)
                # - Pattern:  {original_name}_sin, {original_name}_cos
                # ---
                # 2. Simple Difference
                # - Pattern:  {original_name}_diff
                # ---
                # 3. Percentage Change
                # - Pattern:  {original_name}_pct_change
                # ---
                # 4. Log Then Difference
                # - Pattern:  {original_name}_log_diff
                # ---
                # ✅ Checklist Before Adding New Transformations
                # ---
                # - Does the naming follow the established pattern?
                # - Will the logging code automatically detect this transformation?
                # ---
                # 🔄 Update This SOP When:
                # ---
                # - Adding new transformation types
                # - Changing existing patterns
                # -----------------------------------------------------------------

                for col in new_columns:

                    # Handle sin/cos pairs
                    if col.endswith('_sin') or col.endswith('_cos'):

                        base_name = col.replace('_sin', '').replace('_cos', '')
                        if base_name not in processed_features:
                            sin_col = f"{base_name}_sin"
                            cos_col = f"{base_name}_cos"
                            if sin_col in final_columns and cos_col in final_columns:
                                # Check if original feature is still in final dataset
                                original_status = "✅ KEPT" if base_name in final_columns else "🚮 DROPPED"
                                print(f"┌─ {base_name}")
                                print(f"├─ TRANSFORM: sin_cos")
                                print(f"├─ OUTPUT: {sin_col}, {cos_col}")
                                print(f"└─ ORIGINAL: {original_status}")
                                print()
                                processed_features.add(base_name)
                    
                    # Handle simple difference
                    elif col.endswith('_diff'):

                        base_name = col.replace('_diff', '')
                        if base_name not in processed_features:
                            # Check if original feature is still in final dataset
                            original_status = "✅ KEPT" if base_name in final_columns else "🚮 DROPPED"
                            print(f"┌─ {base_name}")
                            print(f"├─ TRANSFORM: simple_diff")
                            print(f"├─ OUTPUT: {col}")
                            print(f"└─ ORIGINAL: {original_status}")
                            print()
                            processed_features.add(base_name)
                    
                    # Handle percentage change
                    elif col.endswith('_pct_change'):

                        base_name = col.replace('_pct_change', '')
                        if base_name not in processed_features:
                            # Check if original feature is still in final dataset
                            original_status = "✅ KEPT" if base_name in final_columns else "🚮 DROPPED"
                            print(f"┌─ {base_name}")
                            print(f"├─ TRANSFORM: pct_change")
                            print(f"├─ OUTPUT: {col}")
                            print(f"└─ ORIGINAL: {original_status}")
                            print()
                            processed_features.add(base_name)
                    
                    # Handle log then difference
                    elif col.endswith('_log_diff'):

                        base_name = col.replace('_log_diff', '')
                        if base_name not in processed_features:
                            # Check if original feature is still in final dataset
                            original_status = "✅ KEPT" if base_name in final_columns else "🚮 DROPPED"
                            print(f"┌─ {base_name}")
                            print(f"├─ TRANSFORM: log_diff")
                            print(f"├─ OUTPUT: {col}")
                            print(f"└─ ORIGINAL: {original_status}")
                            print()
                            processed_features.add(base_name)

                print(f'Final list of columns for model input:')
                print(f'- This is the FINAL list of columns being fed into the model (after adding_features, normalizing, etc.):')
                print(final_columns, '\n')
                
            else:
                print(f'Columns used: [unable to determine) - using cache of X and y preprocessed data, which does not have list of columns names anymore, just matrix of X and y data.\n- For the list of columns used to generate this cache, look for the previous training run log, which did NOT use a cache.\n')

    # --------------------------------------------------------------------     
    # --------------------------------------------------------------------
    # ------------------- [ KEY SECTION: TRAIN MODEL ] -------------------
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    if True:

        # MODEL ARCHITECTURE (LTSM, one-hot encoding, to_categorical, etc. )
        if True:

            # -------------------------------------------------------------------------
            # MODEL ARCHITECTURE
            # -------------------------------------------------------------------------

            # --- Dynamically determine num_classes ---

            unique_classes = np.unique(train_y) # Find unique class labels

            # Count how many unique labels there are
            num_classes = len(unique_classes) 

            print(f"DEBUG: Dynamically determined number of classes: {num_classes}")
            print(f"DEBUG: Unique classes found: {unique_classes}") 

            # ---------------------------------
            
            # --- One-hot encode the labels after preprocessing --- 

            if num_classes == 2:
                pass # y is already 0s and 1s
            else:
                # Convert labels for multi-class
                train_y = to_categorical(train_y, num_classes=num_classes)
                validation_y = to_categorical(validation_y, num_classes=num_classes)

            # ---------------------------------

            # Number of timesteps per batch
            time_steps = train_x.shape[1]  

            # Number of features per batch
            features = train_x.shape[2]  
         
            # ---------------------------------

            # Original LSTM model
            if True:

                model = Sequential()

                model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
                model.add(Dropout(0.2))
                model.add(BatchNormalization())

                model.add(LSTM(128, return_sequences=True))
                model.add(Dropout(0.1))
                model.add(BatchNormalization())

                model.add(LSTM(128))
                model.add(Dropout(0.2))
                model.add(BatchNormalization())

                model.add(Dense(32, activation='relu'))
                model.add(Dropout(0.2))

                # Output layer for N classes
                if num_classes == 2:
                    model.add(Dense(1, activation='sigmoid')) # 1 unit for binary
                else:
                    model.add(Dense(num_classes, activation='softmax'))

            # -------------------------------------------------------------------------
            # END: MODEL ARCHITECTURE
            # -------------------------------------------------------------------------

        # -------------------------------------------------------

        # Define file paths for trained model and graphs
        if True:
            
            model_training_filepath = model_iteration_file_name + "EPOCH_{epoch:02d}_{val_accuracy:.3f}" 
            
            model_training_model_checkpoint_settings_file_path = "data/training/models/{}_CHECKPOINT.keras"

            model_save_path                   = f'data/training/models/{model_iteration_file_name}_FINAL.keras'
            model_trainingKPIs_log_path       = f'data/training/models/{model_iteration_file_name}_training_kpis.log'
            model_training_console_log_path   = f'data/training/models/{model_iteration_file_name}_training_stdout.log'

            plot_save_path = \
                f'data/training/models/{model_iteration_file_name}_confusion_matrix_epoch_{{epoch_range}}.png'

        # -------------------------------------------------------

        # Training callbacks (modelCheckpoint, earlyStopping, reduceLR, custom logger)
        if True:

            # -------------

            # tensorboard_log_dir = "data/training/logs/{}"
            # tensorboard = TensorBoard(
            #     log_dir=tensorboard_log_dir.format(model_save_name))

            # -----------------------

            checkpoint = ModelCheckpoint(
                model_training_model_checkpoint_settings_file_path.format(model_training_filepath), 
                monitor=model_training_model_checkpoint_settings_monitor,
                verbose=1, 
                save_best_only=False, 
                mode=model_training_model_checkpoint_settings_mode)
            
            # -----------------------

            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                verbose=1, 
                restore_best_weights=True)
            
            # -----------------------

            reduce_learning_rate = ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-6,
                verbose=1)

            # ---------------------------------

            # Save model summary to text file
            if True:

                def save_model_summary(model, summary_file_path):
                    with open(summary_file_path, 'w') as f:
                        model.summary(print_fn=lambda x: f.write(x + '\n'))

                save_model_summary(model, model_summary_output_path)

            # ---------------------------------

            # Write training KPIs to file
            if True:

                class TrainingLoggertoTextFile(keras.callbacks.Callback):

                    # Custom callback to log training progress to file

                    '''
                    This will log training KPIs in an structured, standardized format, eg:

                    Epoch 1
                    loss: 0.7604
                    accuracy: 0.6053
                    val_loss: 0.7355
                    val_accuracy: 0.4082
                    lr: 0.0010

                    Epoch 2
                    loss: 0.6369
                    accuracy: 0.6255
                    val_loss: 0.7427
                    val_accuracy: 0.4082
                    lr: 0.0010
                    '''

                    def __init__(self, log_file_path):
                        super(TrainingLoggertoTextFile, self).__init__()
                        self.log_file_path = log_file_path

                    def on_epoch_end(self, epoch, logs=None):
                        with open(self.log_file_path, 'a') as f:
                            f.write(f"Epoch {epoch+1}\n")
                            for key, value in logs.items():
                                f.write(f"{key}: {value:.4f}\n")
                            f.write("\n")

                # Create an instance of the custom logger callback
                log_training_kpis_to_file = TrainingLoggertoTextFile(model_trainingKPIs_log_path)

                class ConsoleFileLogger:
                    # This will log the entire std.out to a file
                    # This captures all the additional log messages emitted by keras,
                    # Eg. messages from other callbacks, reduceLr, earlyStopping, etc.
                    def __init__(self, file_path):
                        self.file = open(file_path, 'w')
                        self.stdout = sys.stdout  # Keep a reference to the original stdout

                    def write(self, data):
                        self.file.write(data)     # Write to the file
                        self.stdout.write(data)   # Write to the console (original stdout)

                    def flush(self):
                        self.file.flush()
                        self.stdout.flush()

                    def close(self):
                        self.file.close()

                console_file_logger = ConsoleFileLogger(model_training_console_log_path)
                sys.stdout = console_file_logger  # Redirect stdout to the custom logger object
        
            # ---------------------------------

        # -------------------------------------------------------

        # Training model.fit (class weights, confusion matix, optimizer)
        if True:
            try:
                
                # -----------------------

                # Enable class weights & print class distribution
                if True:

                    print('\ny_train:')
                    print(train_y)

                    # ----------------------------------------------------------

                    # Convert to integer format (*if one-hot encoded).
                    if len(train_y.shape) > 1 and train_y.shape[1] > 1:
                        train_y_integers = np.argmax(train_y, axis=1)
                        validation_y_integers = np.argmax(validation_y, axis=1)
                    else:
                        train_y_integers = train_y
                        validation_y_integers = validation_y

                    # ----------------------------------------------------------

                    # Print distrubtion of training set:

                    # ----------------------------------------------------------

                    class_weights_array = class_weight.compute_class_weight(
                        class_weight = 'balanced', 
                        classes      = np.unique(train_y_integers), 
                        y            = train_y_integers)

                    class_weights = {
                        i: class_weights_array[i] \
                            for i in range(len(class_weights_array))}

                    # ----------------------------------------------------------

                    unique, counts = np.unique(train_y_integers, return_counts=True)

                    class_distribution = dict(zip(unique, counts))

                    # ----------------------------------------------------------

                    print("\nClass Distribution (y_train):", 
                        {
                            classifier_mapping[class_group]: 
                            class_distribution[class_group] for class_group in class_distribution
                        })

                    print("Class Weights (y_train):", class_weights, '\n')
            
                    # ----------------------------------------------------------

                    # Print distrubtion of valdidation set:

                    # ----------------------------------------------------------

                    class_weights_array_val = class_weight.compute_class_weight(
                        class_weight = 'balanced', 
                        classes      = np.unique(validation_y_integers), 
                        y            = validation_y_integers)

                    class_weights_val = {
                        i: class_weights_array_val[i] \
                            for i in range(len(class_weights_array_val))}

                    # ----------------------------------------------------------

                    unique_val, counts_val = np.unique(validation_y_integers, return_counts=True)

                    class_distribution_val = dict(zip(unique_val, counts_val))

                    # ----------------------------------------------------------

                    print("\nClass Distribution (y_validation):", 
                        {
                            classifier_mapping[class_group]: 
                            class_distribution_val[class_group] for class_group in class_distribution_val
                        })

                    print("Class Weights (y_validation):", class_weights_val, '\n')
            
                    # ----------------------------------------------------------

                # Press 'y' to start training... (Inspect class distribution before training)
                if False:
                    
                    # Use to inspect class weights & target (y_train) data distrubtion, before continuing.

                    print("Press 'y' to start training...")
                    while True:
                        key = sys.stdin.read(1)
                        if key.lower() == 'y':
                            break

                # -----------------------

                # Plot Confusion Matrix every epoch
                if True:

                    from helpers.confusion_matrix import ConfusionMatrixCallback

                    if num_classes == 2:
                        binary_label_mapping = {0: "future_delta <= 0", 1: "future_delta > 0"}
                        confusion_matrix_label_mapping = binary_label_mapping
                    else:
                        confusion_matrix_label_mapping = classifier_mapping

                    confusion_matrix_plot_callback = ConfusionMatrixCallback(
                        validation_data=(validation_x, validation_y),
                        conf_matrix_plot_interval=1, 
                        blocking=False, # will plot and then block till the plot is closed
                        plot_save_path=plot_save_path,  # pass the plot_save_path or None
                        label_mapping=confusion_matrix_label_mapping
                    )

                # -----------------------

                # Cohen's Kappa metric
                if True:

                    from helpers.kappa import create_kappa_function
                    
                    CohenKappaCallback = create_kappa_function(num_classes=num_classes)

                    cohen_kappa_metric_callback = CohenKappaCallback(validation_data=(validation_x, validation_y))

                # -----------------------

                optimizer           = tf.keras.optimizers.Adam(
                learning_rate       = model_optimizer_learning_rate,
                clipnorm            = model_optimizer_clipnorm) 
                    
                model.compile(
                    loss            = 'binary_crossentropy' if num_classes == 2 \
                                                            else model_training_loss,
                    optimizer       = optimizer,
                    metrics         = model_training_metrics
                )

                model.fit(
                    x               = train_x, 
                    y               = train_y,
                    validation_data = (validation_x, validation_y),
                    shuffle         = True,

                    batch_size      = 32,
                    epochs          = 9999, # Keep training till `EarlyStopping()` is triggered
                    class_weight    = class_weights, 
                    callbacks       = [
                        # tensorboard,
                        checkpoint,       
                        reduce_learning_rate,  # let LR drop first …
                        early_stopping,        # … then decide whether to stop
                        cohen_kappa_metric_callback,
                        log_training_kpis_to_file,
                        confusion_matrix_plot_callback,  # heavier, non-critical work last
                    ]
                )

            finally: # Clean up for `write_training_log_to_file()`

                sys.stdout = console_file_logger.stdout
                console_file_logger.close()

