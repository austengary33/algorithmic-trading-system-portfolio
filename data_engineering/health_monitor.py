from datetime import datetime, timedelta, timezone
from google.cloud import logging as gcp_logging
from joblib import load

from libHelpers import getConfig
from libHelpers import mktHours
from libHelpers import pubSub

import os
import sys
import time
import re
import glob
import logging
import pytz
import json

################### DEVELOPMENT NOTES, START ################### 

# To make changes to this file: 

# > 1. Inspect log entries in the located at: /localDataStoreDisk/logs/orderbook_python/datafeed_monitor.log

# > 2. Test changes in debug mode FIRST, to ensure no syntax errors were introduced (which is COMMON), by running script like this: 'python main_datafeed_monitor.py DEBUG'

# > 3. Once you are confident the changes are ready to deploy, restart the linux service, by using 'sudo systemctl restart datafeed_monitor.service'

################### DEVELOPMENT NOTES, END ################### 

# ------- <><><><><><><><><><><> SEPERATOR <><><><><><><><><><><> ------- 

################### TESTS, START ################### 

# Test: See how a python exit is handled by the linux service
# print('test')
# os.abort()

# ------------------------

# Test: See how a unhandled exception is handled by the linux service
# time.sleep(5)
# print("k" + 1) 

# ------------------------

# Test: Check if python print statements are being logged to systemd service log
# print('test')
# time.sleep(5)
# os.abort()

# ------------------------

################### TESTS, END ################### 

config = getConfig.read_config_file()
default_config = config.defaults()

DATA_DIR_ORDERBOOK = config.get(default_config['env'], 'data_dir')
INSTRUMENTS_FILE = config.get(default_config['env'], 'base_path') + "/config/instrumentsToMonitor.txt"

def setup_logging():
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # IMPORTANT NOTICE ABOUT THIS LOGGING IN THIS SCRIPT (datafeed_monitor.log):

    # Some log statements will show up *LATE* in in the datafeed_monitor.log.
    # Due to sleeping/waiting till the *END* of the minute before continue processing.

    # FOR EXAMPLE

    # This logger entry was being trigger in the code with this logic:
    # `if now.hour == 22 and now.minute == 15:`

    # But the log entry appears like this: 
    # 2024-02-29 22:16:00,059 [INFO] Skipping last minute of forex trading session.

    # Take notice that it happens at 22:16, not 22:15
    # This is because this code is blocked the by .sleep() function, ~line 200
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------

    log_level = logging.INFO # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Store alongside logs from main_datafeed_orderbook.py
    log_file_path = "/localDataStoreDisk/logs/orderbook_python"
    log_filename_format = f"{log_file_path}/datafeed_monitor.log"

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_filename_format,
        when='midnight',
        interval=1,
        backupCount=7 # will rotate daily anyways, so this value never will get higher than 2 (cron job)
    )
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y-%m-%d.log" # ensure rotated files keep the '.log' filename format

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.addHandler(file_handler)

    # Configure the mkt_hours logger
    mkt_hours_logger = logging.getLogger('mkt_hours_logger')
    mkt_hours_logger.setLevel(logging.INFO)
    mkt_hours_logger.addHandler(file_handler)

    logger.info(f"Logging started.")

    return logger

# Configure python logging
logger = setup_logging()

# Configure GCP logging
gcp_client = gcp_logging.Client()
gcp_cloud_logger_health_check = gcp_client.logger("echelon_datafeed_orderbook_health_check")
gcp_cloud_logger_volume_monitor = gcp_client.logger("echelon_datafeed_orderbook_volume_monitor")
gcp_cloud_logger_market_status = gcp_client.logger("echelon_market_status")
gcp_cloud_logger_generic = gcp_client.logger("echelon_generic")

def read_instruments(file_path):
    with open(file_path, "r") as f:
        instruments = [line.strip() for line in f.readlines()]
    return instruments

def check_if_data_files_exist(instruments, minute_dt):

    # initialize an empty list to store missing files
    missing_files = []

    successfully_created_files = []

    # iterate through each instrument
    for instrument in instruments:

        # Construct the timestamp part of the pattern
        timestamp = minute_dt.strftime('%Y-%m-%dT%H:%M')

        # Create a pattern to match the file names
        pattern = f"{instrument}_orderbook_ticks_{timestamp}:\\d{{2}}\\.\\d{{6}}\\.joblib"

        # Construct the date part of the path
        date_path = minute_dt.strftime('%Y-%m-%d')

        # Get the list of all files in the directory, for the given date
        current_files = glob.glob(os.path.join(
                DATA_DIR_ORDERBOOK, date_path, f"{instrument}_orderbook_ticks_*.joblib"))

        # Filter the list of files to *ONLY* include those that match the pattern
        matching_files = [file for file in current_files if re.search(pattern, file)]
        
        # Log the file path 
        for file in matching_files:
            successfully_created_files.append(file)

        # If no matching files are found, add the missing file to the list
        if not matching_files:
            missing_files.append(f"{instrument}_orderbook_ticks_{minute_dt.strftime('%Y-%m-%dT%H:%M')}")

    # Return the lists of file
    return missing_files, successfully_created_files

def broadcast_event_via_pubSub_with_error_handling(MODE, pubsub_topic_name, event_payload):
    try:
        response = pubSub.broadcast_event(
            pubsub_topic_name=pubsub_topic_name,
            event_payload=event_payload
        )
        
        # Log PubSub confirmation to local server log.
        logger.info(f"PubSub confirmation: {response}")

    except Exception as e: # PubSub broadcast error
        # Log PubSub error to local server log.
        logger.error(
            f"PubSub broadcast error:\n"
            f"Error occured in file: main_datafeed_monitor.py\n"
            f"Full error: {str(e)}"
        )
        # Log PubSub error to GCP.
        if MODE == 'PROD':
            gcp_cloud_logger_health_check.log_text(
                severity="ERROR",
                text=(
                    f"PubSub broadcast error:\n"
                    f"Error occured in file: main_datafeed_monitor.py\n"
                    f"Full error: {str(e)}"
                )
            )
        # end exception
    return True

def main(MODE):
    
    instruments_to_monitor = read_instruments(INSTRUMENTS_FILE)

    while True:

        now = datetime.utcnow().replace(tzinfo=pytz.UTC) # Eg. 5:15 PM

        current_minute = now.replace(second=0, microsecond=0)

        # Calculate the next minute 
        next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1) # Eg. 5:15 + 1 = 5:16 PM

        #################################################################################
        # WE SLEEP HERE - IMPORTANT TO BE AWARE OF, AS WILL CAUSE DELAY FOR CODE BELOW
        #################################################################################

        # Wait until the *END* of the current minute, before checking for files
        # Eg. script runs at x time, then sleeps till y time

        # Calculate duration to wait before checking
        next_time_to_check = next_minute.replace(second=0, microsecond=1000) 
        # Wait 1000 microseconds (1 milliseconds) after the minute goes past, eg. 5:15.1000pm
        
        seconds_till_next_check = (next_time_to_check - now).total_seconds()

        logger.info(f"Waiting {seconds_till_next_check} seconds until next check at {next_time_to_check}")

        #################################################################################
        # WE SLEEP HERE - IMPORTANT TO BE AWARE OF, AS WILL CAUSE DELAY FOR CODE BELOW
        #################################################################################

        time.sleep(seconds_till_next_check)

        # --------------------------------------------------------------------
        # --------------------------------------------------------------------
        # -------- REMINDER: ALL CODE BELOW - WILL BE DELAYED - BY CODE ABOVE
        # --------------------------------------------------------------------
        # --------------------------------------------------------------------

        if mktHours.is_trading_minute(current_minute) is False:

            logger.info(f"Market isn't open, at {current_minute}, skipping check for data files.")

            continue # SKIP EXECUTION OF ALL CODE BELOW

        else: # MARKET IS OPEN

            # The market closes at 5 PM ET, and re-opens at 5:15 PM ET.

            # The *FIRST* data file will be created at: 22:16:00,
            # Which will contain all the ticks, which occured from 22:15.01 - to - 22:15.59.
            
            # Eg:
            # '2024-03-04 22:15:00.060438', 
            # '2024-03-04 22:15:00.070710',
            # '2024-03-04 22:15:00.073446', 
            # '2024-03-04 22:15:00.139025',
            # '2024-03-04 22:15:00.142368', 
            # '2024-03-04 22:15:00.161965',

            # You can confirm this yourself,
            # By running the notebook: `analysis_orderbook_ticks_single.ipynb`
            # With a file for the first minute of trading data, like this: 
            # '/localDataStoreDisk/orderbook_data/2024-03-04/GBPUSD_orderbook_ticks_2024-03-04T22:16:00.010191.joblib'

            # So there will be *NO* data file created at: 22:15:00,
            # Since that would contain ticks from 22:14, which is when the market is *CLOSED*.

            # So we wait until open + 1min to check for data. 
            # Since we just monitor data/ticks the first minute, and do NOT write it till the beginning of the second minute, eg. the first write occurs at open+1min (e.g 22:16)

            # ----------------------------------------------------------

            # Check if is the FIRST minute of the forex session (eg. 5:15 PM ET).
            # If yes, skip check for data files.

            # Forex

            if mktHours.is_dst(now) is True: 
                # DST ACTIVE, UTC + 4
                # -----
                # When Daylight Saving Time (DST) is active, Eastern Time (ET) is 4 hours behind Coordinated Universal Time (UTC). So, 5 PM ET in UTC 24-hour time would be 21:00 (9 PM) UTC.
                hour_to_skip = 21
            else:
                # NOT DST, UTC + 5
                # -----
                # When Daylight Saving Time (DST) is not active, Eastern Standard Time (EST) is 5 hours behind Coordinated Universal Time (UTC). Therefore, 5 PM ET in UTC 24-hour time would be 22:00 (10 PM) UTC.
                hour_to_skip = 22

            if now.hour == hour_to_skip and now.minute == 15:

                # This is a temporary fix, to stop getting errors/alerts at this time. 
                # But this WILL have to be changed during DST, to 21:15, vs 22.15 (DST/non-DST)

                # This is a ugly fix too (hardcoding like this).
                # This *WILL* break the datafeed with other instruments (non-forex), that have different market hours (different open/closing times).

                logger.info('Skipping data check for first minute of forex trading session.')

                # Note: This log statement shows up *LATE* in the `datafeed_monitor.log`, like this:
                # `2024-02-29 22:16:00,059 [INFO] Skipping last minute of forex trading session.``

                # It shows up a minute late. 
                # Even though we are using `if now.hour == 22 and now.minute == 15:`

                # Because by the time logic execution resumes,
                # The `now` variable is out of sync with the current time (real now),
                # Due to the .sleep() usage, above.

                continue # SKIP EXECUTION OF ALL CODE BELOW

            # -----
            # If you add more instruments to monitor in the future beyond forex,
            # It would be wise to just have a check if it's the first minute of trading hours,
            # For that instrument, and then just skip the check.
            # -----

        # ----------

        DEBUG = True
        
        if DEBUG:
            logger.info('Starting check for files...')

        time_elapsed = 0 

        # Check for data files every 1000 microseconds (1 millisecond)
        CHECK_FILES_SLEEP_INTERVAL = 0.001  

         # Wait 1 second before triggering an error (if data files not yet created)
        ERROR_WAIT_TIME = 1 

        # Keep checking for files, until the max wait time 
        while time_elapsed < ERROR_WAIT_TIME:

            missing_instrument_files, success_instrument_files = check_if_data_files_exist(
                instruments_to_monitor, 
                current_minute
            )

            # If NO files are missing, print a "success" message, and exit the loop
            if not missing_instrument_files:
                
                log_success_message_for_new_minute_data = True
                if log_success_message_for_new_minute_data:
                    logger.info(f"Success: All files present for {current_minute}")
                    # Log success msg to GCP log
                    if MODE == 'PROD':
                        gcp_cloud_logger_health_check.log_text(
                            f"Success: All files present for: {current_minute}", severity="INFO")
                    
                pubsub_broadcast_new_minute_data = True
                if pubsub_broadcast_new_minute_data: 

                    # Broadcast we got new data files for minute.
                    broadcast_event_via_pubSub_with_error_handling(
                        MODE=MODE,
                        pubsub_topic_name="data-files-created",
                        event_payload=dict(
                            name='Successfully created data files for minute.',
                            minute=current_minute.isoformat(),
                            files=success_instrument_files
                        )
                    )

                data_integrity_checks = True
                if data_integrity_checks:
                    if len(success_instrument_files) > len(instruments_to_monitor):
                        # For more info about this error, see this workflowy bug report:,
                        # https://workflowy.com/#/b12cbb791aa2
                        gcp_cloud_logger_health_check.log_text(
                            severity="ERROR",
                            text=(
                                f"Got MORE data files, then the number of instruments we are monitoring. Occured during: {current_minute}. Error occured in main_datafeed_monitor.py. See bug report link in comments."))
                        
                generate_volume_summary_statistics = True
                if generate_volume_summary_statistics:
                    # Measure the ticks per minute, for volume monitoring.
                    tick_volume_summary = {}

                    for file_name in success_instrument_files:

                        instrument_name = file_name.split('_orderbook_ticks')[0].split('/')[-1]
                        instrument_file_contents = load(file_name)

                        num_of_ticks_for_instrument = len(instrument_file_contents)

                        tick_volume_summary[instrument_name] = num_of_ticks_for_instrument

                        logger.info(f"Data file path: {file_name}")

                    # Log the tick summary, for the minute, to the datafeed_monitor.log file 
                    logger.info(tick_volume_summary)

                    # Log the tick summary for the minute minute to GCP.
                    if MODE == 'PROD':
                        gcp_cloud_logger_volume_monitor.log_struct(
                            severity="INFO",
                            info=tick_volume_summary)

                break # Done - stop while loop for checking for files, for this minute.

            # If files NOT found yet, WAIT and try again.
            time.sleep(CHECK_FILES_SLEEP_INTERVAL)
            
            time_elapsed = (datetime.now(timezone.utc) - next_time_to_check).total_seconds()

            if DEBUG:
                logger.info(dict(
                    time_elapsed=time_elapsed,
                    message="sleeping... (files NOT found yet, WAIT and try again.)"
                ))

        # After waiting for max wait time, if there  ARE* missing files, trigger an error.
        if missing_instrument_files:

            # Log the error to the datafeed_monitor.log file.
            logger.error((
                f"Error: Missing data files for minute: {current_minute}."
                f" List of missing files: { ', '.join(missing_instrument_files) }"))
            
            # Log the error to GCP
            if MODE == 'PROD':
                gcp_cloud_logger_health_check.log_text(
                    severity="ERROR",
                    text=(
                        f"Error: Missing data files for minute: {current_minute}."
                        f" List of missing files: { ', '.join(missing_instrument_files) }"))
                
            # Broadcast the error via PubSub.
            broadcast_event_via_pubSub_with_error_handling(
                MODE=MODE,
                pubsub_topic_name="data-files-created",
                event_payload=dict(
                    name='Failed to create data files for minute.',
                    minute=current_minute.isoformat(),
                    files=None
                )
            )
            # Also broadcast this same error, to the order management data pipeline handler
            broadcast_event_via_pubSub_with_error_handling(
                MODE=MODE,
                pubsub_topic_name="order-management",
                event_payload=dict(
                    name='Failed to create data files for minute.',
                    minute=current_minute.isoformat(),
                    files=None
                )
            )

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python main_datafeed_orderbook_monitor.py <mode (DEBUG or PROD)>")
        sys.exit(1)

    else:

        IS_DEBUG_MODE = sys.argv[1]
        
        if IS_DEBUG_MODE == 'DEBUG':
            # DEBUG will disable GCP logging.
            print('Running script in debug mode. Logging information will NOT be sent to GCP log.')

            # Run test code in debug mode below:
            # slack.post_to_slack(msg_txt='test', gcp_logger=gcp_cloud_logger_generic)
        
        elif IS_DEBUG_MODE == 'PROD':
            print('Running script in production mode. Logging information will be sent to GCP log.')
            
        else:
            raise Exception('Wrong mode! Must either be DEBUG or PROD')
        
        # Run main script
        main(IS_DEBUG_MODE)
