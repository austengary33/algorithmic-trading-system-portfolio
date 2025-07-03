#!/bin/bash

# Activate the virtual environment
source /home/austengary/my-jupyter-env/bin/activate

echo "Starting datafeed monitor" 

cd /home/austengary/Desktop/Dev/System_Code/Testing/IB_API

# Pipe the output from the python script (ie. print statements) back to the service so it can be inspected in the service/journal logs
PYTHONUNBUFFERED=1 python main_datafeed_monitor.py PROD 2>&1