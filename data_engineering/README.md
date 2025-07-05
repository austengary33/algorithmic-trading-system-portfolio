# Data Engineering Pipeline

This directory contains the core components of the real-time data ingestion and monitoring pipeline. These services are designed to run 24/7 on a Linux VM, capturing high-frequency financial data with a strong emphasis on reliability and data integrity.

## Modules

### `data_collector.py`

**Purpose:** This is a multi-process Python service responsible for streaming Level 2 order book data from the Interactive Brokers API.

**Key Skills Demonstrated:**
*   **API Integration:** Connects to a real-time, high-throughput financial data API (`ib_insync`).
*   **Resilience & Error Handling:** Implements custom error handlers and robust, exponential-backoff reconnection logic to handle API disconnects and market data resets gracefully.
*   **Concurrency:** Uses threading to perform non-blocking I/O operations, writing data to disk in one-minute batches without interrupting the real-time data stream.
*   **Data Persistence:** Buffers incoming ticks and serializes them to disk using `joblib` for efficient storage and retrieval.

### `health_monitor.py`

**Purpose:** This is a companion service that acts as a watchdog for the data pipeline. It runs every minute to validate the output of the `data_collector.py` service.

**Key Skills Demonstrated:**
*   **System Monitoring & Validation:** Proactively checks for data file existence, completeness across all instruments, and timeliness to ensure data quality.
*   **Cloud-Native Integration:** Leverages Google Cloud Platform (GCP) services for production-grade monitoring. It publishes health status, volume metrics, and error alerts to dedicated **GCP Pub/Sub** topics and logs structured data to **GCP Cloud Logging**.
*   **Event-Driven Architecture:** Acts as the primary publisher in the system's event-driven architecture, triggering downstream processes (like ML inference) upon successful data validation.
