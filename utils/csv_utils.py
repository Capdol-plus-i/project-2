#!/usr/bin/env python3
"""
CSV utilities module for managing robot snapshot data storage and retrieval.
"""

import os
import csv
import threading
import logging
from config import CSV_FILENAME, CSV_HEADERS

logger = logging.getLogger(__name__)

# CSV thread lock
CSV_LOCK = threading.Lock()

def init_csv_file():
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADERS)
        logger.info(f"Created new CSV file: {CSV_FILENAME}")
    else:
        logger.info(f"Using existing CSV file: {CSV_FILENAME}")

def save_snapshot_to_csv(data):
    """Save snapshot data to CSV file"""
    try:
        with CSV_LOCK:
            # Prepare row data
            row_data = data
            
            # Write to CSV
            with open(CSV_FILENAME, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)
            
            # Count total snapshots
            with open(CSV_FILENAME, 'r') as csvfile:
                total_snapshots = sum(1 for line in csvfile) - 1  # Subtract header
            
            logger.info(f"Snapshot saved to CSV. Total snapshots: {total_snapshots}")
            return True, total_snapshots
            
    except Exception as e:
        logger.error(f"Error saving snapshot to CSV: {e}")
        return False, 0

def get_csv_stats():
    """Get CSV file statistics"""
    try:
        if not os.path.exists(CSV_FILENAME):
            return {"exists": False, "total_snapshots": 0, "file_size": 0}
            
        with open(CSV_FILENAME, 'r') as csvfile:
            total_lines = sum(1 for line in csvfile)
            total_snapshots = max(0, total_lines - 1)  # Subtract header
            
        file_size = os.path.getsize(CSV_FILENAME)
        
        return {
            "exists": True, 
            "total_snapshots": total_snapshots, 
            "file_size": file_size,
            "filename": CSV_FILENAME
        }
    except Exception as e:
        logger.error(f"Error getting CSV stats: {e}")
        return {"exists": False, "total_snapshots": 0, "file_size": 0}