#!/usr/bin/env python3
"""
Script to help download and prepare data for analysis
Since direct download from Google Drive folder is not working,
this script will help you manually download and prepare the data
"""

import os
import pandas as pd
import numpy as np

def check_data_files():
    """Check if data files exist in the workspace"""
    data_dir = '/workspace/data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    print("\nPlease manually download the 5 datasets from the Google Drive link:")
    print("https://drive.google.com/drive/folders/1hgJPs06K4Q6wyFnYpBYvTVN8hYkJLW3e")
    print("\nSave them in the /workspace/data directory with descriptive names like:")
    print("- registrations.csv")
    print("- deposits.csv") 
    print("- bets.csv")
    print("- activity.csv")
    print("- player_info.csv")
    print("\nOr use their original names from Google Drive")
    
    # Check for existing CSV files
    existing_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if existing_files:
        print(f"\nFound existing CSV files in {data_dir}:")
        for f in existing_files:
            print(f"  - {f}")
    else:
        print(f"\nNo CSV files found in {data_dir}")
    
    return existing_files

if __name__ == "__main__":
    check_data_files()