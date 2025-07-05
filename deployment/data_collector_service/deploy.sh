#!/bin/bash

SERVICE_NAME="orderbook.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"

# Step 1: Check if the service exists
if systemctl list-units --full --all | grep -Fq "$SERVICE_NAME"; then
    # Step 2: Update the service configuration file and reload the service
    echo "Service already exists. Updating and reloading it."
    sudo cp orderbook.service $SERVICE_PATH
    sudo systemctl daemon-reload
    sudo systemctl restart $SERVICE_NAME
else
    # Step 3: Create and enable the service
    echo "Service does not exist. Creating and enabling it."
    sudo cp orderbook.service $SERVICE_PATH
    sudo systemctl enable $SERVICE_NAME

    # Step 4: Start the service (if it was not already running)
    sudo systemctl start $SERVICE_NAME
fi

# Step 5: Display the service status
sudo systemctl status $SERVICE_NAME