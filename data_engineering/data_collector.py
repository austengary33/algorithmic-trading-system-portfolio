# 1. HOW TO DEBUG THIS SCRIPT 
# ---
# Set DEBUG = True
# ---
# Stop the forex feed, to free up order book connection (max 3):
# ---
    # > sudo systemctl stop monitor_orderbook.service
# ---
# Use BTC Crypto feed:
# ---
    # `/home/austengary/my-jupyter-env/bin/python /home/austengary/Desktop/Dev/System_Code/Testing/IB_API/main_datafeed_orderbook.py BTC Crypto``
# ---
# Monitor the log file for BTC:
# ---
    # /localDataStoreDisk/logs/orderbook_python/BTC_orderbook.log
# ---
# 2. HOW TO RESTART FEED (WHEN FINISHED WITH DEBUGING):
# ---
# A. Disable DEBUG-mode in py script (Set `DEBUG = False`)
# ---
# B. Restart orderbook service: 
# ---
    # `sudo systemctl start monitor_orderbook.service`‸
# ---
# C. Check in GUI that data feeds reconnected.
# - Link: https://remotedesktop.google.com/access/session/be06e795-fa40-4439-9be8-9e38c7809150
# ---


# ------------------------------------

DEBUG = False

# ------------------------------------

from datetime import datetime, timezone

import threading
import sys
import logging
import logging.handlers
import pandas as pd
import joblib
import ib_insync

from libHelpers import timeHelpers
from libHelpers import fileHelpers
from libHelpers import getConfig
from libHelpers import ibAPIHelpers

# ------------------------------------

buffer_list_of_orderbook_ticks = []

pending_tasks = []

ticker = None

# Used for tracking script run time while api connected
last_successful_api_connect_time = None 

# -------------------------------------

# ------------------------------------
# DEBUGGING HELPER: 
# - Test how a unhandled exception is handled by the linux service.
# - Uncomment the following lines to test:
# ------------------------------------
# import os
# import time

# print('test')
# os.abort()

# time.sleep(5)
# print("k" + 1) 
# ------------------------------------

class CustomIBLogFilter(logging.FileHandler):
    def emit(self, record):
        # Change the log level of expected IB API errors
        # Eg: 317 since we've handled it in custom_error_handler()
        if "Error 317" in record.msg and "Market depth data has been RESET" in record.msg:
            record.levelno = logging.INFO
            record.levelname = logging.getLevelName(record.levelno)
        super().emit(record)
    
def custom_error_handler_IB_API(reqId, errorCode, errorString, contract):
    # Catch and handle IB API errors
    # ---
    # Example: 2023-12-14 22:09:05,383 [ERROR] wrapper.wrapper.py:1113 error: Error 317, reqId 4: Market depth data has been RESET. 
    # Please empty deep book contents before applying any new entries., contract: Forex('EURUSD', conId=12087792, exchange='IDEALPRO', localSymbol='EUR.USD', tradingClass='EUR.USD')
    
    if errorCode == 317 or errorCode == 1101:
        # 317: "Market depth data has been RESET" 
        # 1101: "Connectivity between IB and TWS has been restored - data lost.""
        script_logger.info(f"Handling Error {errorCode}: Adding pending_task to resubscribe_depth_data next flush for contract {contract}")
        pending_tasks.append('resubscribe_depth_data()') # will run during next flush_buffer

    # elif errorCode == 1100:
    #     # Expected maintenance window for North American IB servers
    #     # Eg: 2023-12-18 05:15:31,534 [ERROR] wrapper.wrapper.py:1113 error: Error 1100, reqId -1: Connectivity between IB and Trader Workstation has been lost.
    #     script_logger.info(f"Handling Error 1100: Expected Interactive Brokers Maintenance Window for North American servers. Disconnected from API. Supressing error.")
        
    # elif errorCode == 1102:
    #     # Eg: 2023-12-18 05:15:57,926 [ERROR] wrapper.wrapper.py:1113 error: Error 1102, reqId -1: Connectivity between IB and Trader Workstation has been restored - data maintained. All data farms are connected: cashfarm; usfarm; ushmds; secdefnj.
    #     script_logger.info(f"Handling Error 1102: Expected Interactive Brokers Maintenance Window for North American servers. Reconnected to IB API successfully. Supressing error.")
    
def setup_logging(instrument):

    # if debugging, set script_logger_log_level level to DEBUG
    if DEBUG:
        script_logger_log_level = logging.DEBUG # WARNING, ERROR, CRITICAL
    else:
        script_logger_log_level = logging.INFO

    ib_api_logger_level = logging.ERROR

    log_file_path = "/localDataStoreDisk/logs/orderbook_python"
    log_filename_format = f"{log_file_path}/{instrument}_orderbook.log"

    # include file name and function name in log events
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s.%(filename)s:%(lineno)d %(funcName)s: %(message)s') 
    # include only time, log level and log message
    # formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_filename_format,
        when='midnight',
        interval=1,
        backupCount=30
    )
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y-%m-%d.log" # ensure rotated files keep the '.log' filename format

    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    
    # Logger for this script
    script_logger = logging.getLogger(__name__)
    script_logger.setLevel(script_logger_log_level)
    script_logger.addHandler(file_handler)
    # script_logger.addHandler(stream_handler)

    # Logger for ib_insync
    ib_logger = logging.getLogger('ib_insync')
    ib_logger.setLevel(ib_api_logger_level)
    # ib_logger.addHandler(file_handler)
    # ib_logger.addHandler(stream_handler)
    custom_ib_log_filter = CustomIBLogFilter(log_filename_format)
    custom_ib_log_filter.setFormatter(formatter)
    ib_logger.addHandler(custom_ib_log_filter)

    script_logger.debug(f"Logging started for {instrument}")
    return script_logger

# ----------------------------------------------------
# ----------------------------------------------------
# ------------- MAIN SET UP --------------------------
# ----------------------------------------------------
# ----------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage: python monitor_instrument_orderbook.py <instrument> <contract_type (optional)>")
        # contract_type: options are: <Forex|Crypto>. eg. BTC
        # ---
        # eg. /home/austengary/my-jupyter-env/bin/python /home/austengary/Desktop/Dev/System_Code/Testing/IB_API/main_datafeed_orderbook.py BTC Crypto
        sys.exit(1)

    instrument_to_monitor = sys.argv[1] # 'EURUSD', 'USDJPY', 'GBPUSD'

    try: 
        # Optional argument
        type_of_contract = sys.argv[2]
    except:
        # If optional argument is not given, default to forex
        type_of_contract = 'Forex'

    script_logger = setup_logging(instrument=instrument_to_monitor)

# ----------------------------------------------------
# ----------------------------------------------------
# ------------- MAIN SET UP: END ---------------------
# ----------------------------------------------------
# ----------------------------------------------------

def on_orderbook_update(tick_update):

    global buffer_list_of_orderbook_ticks

    ###########
    
    now = datetime.now()

    _update_dict = {'timestamp': now}

    if DEBUG:
        source_timestamp = tick_update.time # This 'time' is updated by ib_insync with broker time

        # Check if it's a datetime object and not None
        # script_logger.debug(f"tick_update.time: {source_timestamp}, type: {type(source_timestamp)}")
        # Compare it with datetime.now() to see the typical delta
        # current_py_time = datetime.now()
        current_py_time = datetime.now(timezone.utc)
        print(f"datetime.now(): {current_py_time}, diff (py-ib): {(current_py_time - source_timestamp).total_seconds() if source_timestamp else 'N/A'}")
        # script_logger.debug(f"datetime.now(): {current_py_time}, diff (py-ib): {(current_py_time - source_timestamp).total_seconds() if source_timestamp else 'N/A'}")
        pass # Minimal logging in production callback
    
    if DEBUG:
        print(f'Got orderbook update for {instrument_to_monitor} at {now}')
        #script_logger.debug(f"Got orderbook update for {instrument_to_monitor} at {now}")

    ###########

    _dataframe = pd.DataFrame(
        index=range(num_orderbook_levels),
        columns='bidSize bidPrice askPrice askSize'.split())

    # Bids, eg. levels 1 through 10
    bids = tick_update.domBids

    for i in range(num_orderbook_levels):
        _dataframe.iloc[i, 1] = bids[i].price if i < len(bids) else 0
        _dataframe.iloc[i, 0] = bids[i].size if i < len(bids) else 0

        _update_dict["bid_price_"+ str(i+1)] = bids[i].price if i < len(bids) else 0
        _update_dict["bid_size_"+ str(i+1)] = bids[i].size if i < len(bids) else 0

    # Asks
    asks = tick_update.domAsks

    for i in range(num_orderbook_levels):
        _dataframe.iloc[i, 2] = asks[i].price if i < len(asks) else 0
        _dataframe.iloc[i, 3] = asks[i].size if i < len(asks) else 0

        _update_dict["ask_price_"+ str(i+1)] = asks[i].price if i < len(asks) else 0
        _update_dict["ask_size_"+ str(i+1)] = asks[i].size if i < len(asks) else 0

    ###########

    # DEBUGGING: USE TO PRINT ORDERBOOK DF TO CONSOLE
    # ---
    # Eg:
    #     bidSize  bidPrice  askPrice   askSize
    # 0  1.289315  94601.25   94602.5       1.5
    # 1     0.015   94601.0  94635.25       4.0
    # 2       0.5   94591.0   94636.0     0.015
    # 3       1.5   94578.5   94645.5       0.5
    # 4       4.0   94577.0   94651.0       1.5
    # 5       0.5   93505.0   95332.5      0.02
    # 6      0.04   92973.5  95656.75  0.001068
    # 7  0.005373  92724.25   95934.5  0.052316
    # 8  0.146083   92674.5   96336.0  0.001066
    # 9  0.005425   91842.5   97707.0  0.149984
    # ---
    # os.system("cls" if os.name == "nt" else "clear")
    # print(now)
    # print(_dataframe)

    # if _dataframe['bidSize'].sum() > _dataframe['askSize'].sum():
    #     market_actor_in_control = 'Bids control - by x {}, y% {}'.format(
    #         _dataframe['bidSize'].sum() - _dataframe['askSize'].sum(), 

    #         format(_dataframe['askSize'].sum()/_dataframe['bidSize'].sum()))
        
    # elif _dataframe['askSize'].sum() > _dataframe['bidSize'].sum():
    #     market_actor_in_control = 'Sellers control - by x {}, y% {}'.format(
    #         _dataframe['askSize'].sum() - _dataframe['bidSize'].sum(), 

    #         format(_dataframe['bidSize'].sum() / _dataframe['askSize'].sum()))
    # else: 
    #     market_actor_in_control = 'BALANCED'
    # print(market_actor_in_control)

    # Add bids and ask ticks to buffer
    buffer_list_of_orderbook_ticks.append(_update_dict)

def resubscribe_depth_data():
    # Function to be used when data connection is reset
    global ticker
    
    try:
        script_logger.info('Attempting to resubscribe_depth_data().')

        # Clean up existing subscription if any
        if ticker is not None:

            # Remove existing handler first
            ticker.updateEvent -= on_orderbook_update

            # Then cancel the market depth subscription
            ib.cancelMktDepth(contract, isSmartDepth=True)

            # Clear the reference
            ticker = None  

            # Sleep to give IB Java API a chance to clear any remaining handlers
            ib.sleep(0.2)

        # Create new subscription
        ticker = ib.reqMktDepth(contract, 
                               numRows=num_orderbook_levels, 
                               isSmartDepth=True)
        
        # Add new handler
        ticker.updateEvent += on_orderbook_update

        script_logger.info('Ran resubscribe_depth_data() successfully.')

        return True
    
    except Exception as e:
        script_logger.exception(f"Error processing resubscribe_depth_data(): {e}.")

        # If the error message indicates the connection is lost, 
        # re-raise the exception so the main loop reconnect logic can handle it
        if isinstance(e, ConnectionError) or "Not connected" in str(e) or "Socket disconnect" in str(e):
            raise

        return False

def ib_connect():

    global num_orderbook_levels, \
        last_successful_api_connect_time, \
            pending_tasks, \
                ticker

    #############

    # Read IB API config file
    if True:

        config = getConfig.read_config_file()
        defaults = config.defaults()

        host = config.get(defaults['env'], 'host')
        port = config.getint(defaults['env'], 'port')

    #############

    # Connect to IB API

    num_ib_api_connect_attempts = 0

    while True:

        try:

            unused_client_id = ibAPIHelpers.find_unused_ib_client_id(script_logger)
            if unused_client_id is None:
                raise Exception("Got invalid clientID from ibAPIHelpers.find_unused_ib_client_id.")
            
            script_logger.info(
                f"Attempting to connect to IB API: {host}, {port}, "
                f"clientId={unused_client_id}, contract={instrument_to_monitor}.")

            ib.connect(host, port, clientId=unused_client_id)
            ib.errorEvent += custom_error_handler_IB_API

            # Track script runtime while connected to IB API
            last_successful_api_connect_time = datetime.now() 

            script_logger.info(
                f"Successfully connected to IB API at: {last_successful_api_connect_time}. "
                f"Connection details: {host}, {port}, clientId={unused_client_id}.")

            # Clear tasks from previous api sesssion (eg. resubscribe_depth_data)
            pending_tasks = [] 

            # End while loop to connect
            break 
        
        except Exception as e:
            num_ib_api_connect_attempts += 1
            backoff_time = 0.05 * num_ib_api_connect_attempts
            
            script_logger.error(
                f"Failure connecting to IB API: {e}. "
                f"Retrying in {backoff_time} seconds. "
                f"Attempt: {num_ib_api_connect_attempts}")

            ib.sleep(backoff_time)

    ###################

    # Setup contact for instrument
    if True:

        # Docs on how to use IB contracts API: 
        if True:
            # Guide to set up IB contracts for various instruments (eg. stock, options, bond, crypto, etc.): 
            # > https://github.com/erdewit/ib_insync/blob/master/ib_insync/contract.py
            # ---
            pass

        if type_of_contract == 'Forex':
            contract = ib_insync.Forex(instrument_to_monitor)

        elif type_of_contract == 'Crypto':
            # Using zerohash exchange
            contract = ib_insync.Crypto(instrument_to_monitor, 'ZEROHASH', 'USD')

        else:
            raise ValueError('Invalid type_of_contract. Options are: <Forex|Crypto>')
    
        ib.qualifyContracts(contract)

    ###################

    # Setup orderbook connection
    if True:
        num_orderbook_levels = 10
        
        script_logger.info(
            f"Starting orderbook monitoring for {instrument_to_monitor}, with {num_orderbook_levels} levels of market depth.")
        
        if DEBUG:
            print(f"Starting orderbook monitoring for {instrument_to_monitor}, {num_orderbook_levels} levels of market depth")

        # Setup orderbook callback
        ticker = ib.reqMktDepth(contract, 
                                numRows=num_orderbook_levels, 
                                isSmartDepth=True)
        ticker.updateEvent += on_orderbook_update

    return contract

    ################### [FUNCTION END: ib_connect()]

def ib_disconnect(contract):
    script_logger.info('Disconnecting from IB API...')

    try: 
        ib.cancelMktDepth(contract, isSmartDepth=True)
        ib.disconnect()
        script_logger.info('Successfully disconnected from IB API...')

    except Exception as e:
        script_logger.exception(f"Error processing ib_disconnect(): {e}.")

def dump_buffer_to_file(data_array, incomplete_sample_type=None):

    # ------------------------------------------------------
    # Be careful NOT to introduce any signficiant processing TIME into this function (flush_buffer)
    # ------
    # Because if it takes LONGER than expected to flush the buffer,
    # Ticks from the next minute, could bleed over into the previous minute.
    # ------
    # To avoid this issue, you could create a COPY() of the buffer (data_array) BEFORE passing it to the `flush_buffer` function. 
    # -------
    # This way, any new ticks added during the flush  will NOT be included in the data being saved, as they will be added to the original buffer, not the COPY being flushed. 
    # -------
    # For more information and a modified verison of the code, with this tweak: https://workflowy.com/#/84a40dc2e0a0
    # ------------------------------------------------------

    script_logger.debug('Flushing buffer...')

    if incomplete_sample_type is not None:
        file_label = incomplete_sample_type
        
    current_datetime, current_date = timeHelpers.get_current_date_and_time()

    # Local store for data files
    _local_data_directory = '/localDataStoreDisk/orderbook_data/'
    _local_data_directory = _local_data_directory + current_date + '/'

    fileHelpers.create_folder_if_not_exists(_local_data_directory)

    file_label = None

    base_name = f"{instrument_to_monitor}_orderbook_ticks_{current_datetime}"
    if file_label is not None:
        base_name += f"_{file_label}"

    output_file_name = fileHelpers.generate_unique_filename(
        base_name=base_name,
        extension="joblib",
        directory=_local_data_directory)
    
    output_file_name_with_path = _local_data_directory + output_file_name

    script_logger.debug(
        f"Saving buffer contents to local file: {output_file_name_with_path}\n"
        f"Total orderbook update ticks written: {len(data_array)}")
    
    if DEBUG:
        print(
            f"Saving buffer contents to local file: {output_file_name_with_path}\n"
            f"Total orderbook update ticks written: {len(data_array)}")
   
    # Use joblib to write to file
    joblib.dump(data_array, output_file_name_with_path) 

    # Clear buffer
    data_array = [] 

    # If we have pending tasks, execute them
    if len(pending_tasks) > 0:

        # For each pending task
        for task in pending_tasks.copy():
            # If orderbook data was reset
            if task == "resubscribe_depth_data()":
                if resubscribe_depth_data():
                    # Remove task only *AFTER* it has sucessfully ran
                    pending_tasks.remove(task)

    return data_array

# ---------------

# MAIN WHILE LOOP
try:

    # Set up connection to IB API
    ib = ib_insync.IB() 
    contract = ib_connect()

    # Loop params
    iterations = 1
    last_write_time = None

    while True:    
        try:
            now = datetime.now() # update current time

            if DEBUG:
                # Print current time
                print(f'now: {now}') 

            # IB.sleep() until the START of the NEXT minute.
            time_to_next_minute = timeHelpers.time_until_next_minute()
            ib_insync.IB.sleep(time_to_next_minute)

            if DEBUG:
                print(f'sleeping till {time_to_next_minute}.')

            # --------------------

            # If we have ticks to flush, write them to the file.
            # - We use `>1` and NOT `0` - as sometimes the API passes a single tick/char during closed periods, which would result in an empty data file being created.

            if len(buffer_list_of_orderbook_ticks) > 1: 
                buffer_copy = buffer_list_of_orderbook_ticks.copy()

                # Clear current buffer
                buffer_list_of_orderbook_ticks = [] 

                threading.Thread(target=dump_buffer_to_file, args=(buffer_copy,)).start()

                if DEBUG:
                    print('flushing data')

            else: 
                # If we have no ticks to flush

                if DEBUG:
                    print('skipping flush due to insufficient data.')
                    
                script_logger.info('skipping flush due to insufficent data.')

                if instrument_to_monitor == 'USDJPY':
                    # What this does, is that for every minute we don't have any ticks in the buffer, for USDJPY, we reset the orderbook connection.
                    # - I'm testing this change, since USDJPY seems to be having issues reconnecting to the datafeed (periodically/intermittently) after the daily 5pm ET forex market pause
                    # - For more info, see: (`TEXT/DONE/11.18.24 - BIG PROBLEM DATA FEED.txt`).
                    resubscribe_depth_data()

            # -------------

            iterations = iterations + 1
            
            #script_logger.debug('{} iterations'.format(iterations))
            #os.system('cls' if os.name == 'nt' else 'clear') # Reset terminal stream ouput

        except Exception as e:
            # Include full traceback in error
            script_logger.exception(f"Error processing orderbook datafeed: {e}.")

            #########
            # ERROR METRICS
            # ----- 
            script_logger.info(
                f"Number of orderbook ticks in memory at time of failure. "
                f"# of ticks flushed: {len(buffer_list_of_orderbook_ticks)}")
            # -----
            script_logger.info(
                f"Duration the script was running for before it failed: "
                f"{timeHelpers.td_format(datetime.now() - last_successful_api_connect_time)}")
            ########
            # ----- 
            # ERROR METRICS: END
            ########

            # Reset script running time
            disconnect_time = now
            last_successful_api_connect_time = None 
            
            # If we have ticks to flush
            if len(buffer_list_of_orderbook_ticks) > 1: 
                script_logger.info(
                    f"Initiating emergency flush of the ticks in the buffer due to a failure. "
                    f"# of ticks flushed: {len(buffer_list_of_orderbook_ticks)}.")
                
                buffer_copy = buffer_list_of_orderbook_ticks.copy()

                # Clear current buffer
                buffer_list_of_orderbook_ticks = [] 

                threading.Thread(target=dump_buffer_to_file, args=(buffer_copy,)).start()

            # Reconnect to IB API
            while True:
                try: 
                    script_logger.info(f"Attempting to reconnect to IB API via ib_connect_loop.")

                    ib_disconnect(contract)
                    contract = ib_connect()

                    reconnect_time_taken = timeHelpers.td_format(datetime.now() - disconnect_time)
                    script_logger.info(f"Reconnected successfully! "
                                        f"Took {reconnect_time_taken} to reconnect. ")

                    break # reconnected, break this inner while loop, to continue with main while loop?

                except Exception as e:
                    script_logger.error(f"Failure in ib_reconnect_loop!")    
                              
except Exception as e:
    if DEBUG:
        print(str(e))

    script_logger.error(e)

    ib_disconnect(contract)