# Deployment & Service Management

This directory contains the configuration files and scripts used to deploy and manage the Python applications as robust, production-grade services on a Linux server, ensuring the entire application is automated, resilient, and maintainable.

## Service Configuration

Each subdirectory (`data_collector_service`, `data_monitor_service`, etc.) contains two key files:

### `deploy.sh`

**Purpose:** An idempotent Bash script responsible for deploying or updating the service. It copies the latest service configuration, reloads the `systemd` daemon, and restarts the service to apply changes. This automates the deployment process and ensures consistency.

### `service.service`

**Purpose:** A `systemd` unit file that defines how the Python application runs as a background service.

**Key Skills Demonstrated:**
*   **Resilience:** The `Restart=on-failure` directive ensures the service automatically recovers from crashes, providing high availability.
*   **Production Service Management:** Defines the user, working directory, and execution command, following Linux best practices for running applications.
*   **Centralized Logging:** The `StandardOutput=journal` and `StandardError=journal` directives pipe all application output (including `print` statements and errors) directly to `journald`, allowing for robust, centralized logging and monitoring.
