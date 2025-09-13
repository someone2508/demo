#!/usr/bin/env python3
"""
Script to load and analyze your actual data from Google Drive
Instructions:
1. Download the 5 CSV files from Google Drive link
2. Place them in /workspace/data/ directory
3. Update the file names below to match your actual files
4. Run this script to perform the analysis on real data
"""

import pandas as pd
import numpy as np
import os
from player_analytics_assignment import PlayerAnalytics

class RealDataAnalytics(PlayerAnalytics):
    def __init__(self):
        super().__init__()
        self.data_dir = '/workspace/data'
        
    def load_real_data(self):
        """Load actual data from CSV files"""
        print("Loading data from CSV files...")
        
        # Update these file names to match your actual files
        files_to_load = {
            'registrations': 'registrations.csv',  # Update with actual filename
            'deposits': 'deposits.csv',            # Update with actual filename
            'bets': 'bets.csv',                   # Update with actual filename
            'activity': 'activity.csv',           # Update with actual filename
            'player_info': 'player_info.csv'      # Update with actual filename
        }
        
        loaded_data = {}
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            print(f"Creating directory: {self.data_dir}")
            os.makedirs(self.data_dir)
            print("\n⚠️  Please download the CSV files from Google Drive and place them in /workspace/data/")
            print("Then update the filenames in this script and run again.")
            return False
        
        # Try to load each file
        for data_type, filename in files_to_load.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    loaded_data[data_type] = df
                    print(f"✅ Loaded {data_type}: {len(df)} rows")
                    
                    # Display first few columns to understand structure
                    print(f"   Columns: {list(df.columns)[:5]}...")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
            else:
                print(f"⚠️  File not found: {filepath}")
        
        # Map loaded data to class attributes based on content
        self.map_data_to_attributes(loaded_data)
        
        return len(loaded_data) > 0
    
    def map_data_to_attributes(self, loaded_data):
        """Map loaded data to class attributes based on column analysis"""
        print("\nMapping data to analysis attributes...")
        
        # Try to identify which dataset is which based on columns
        for name, df in loaded_data.items():
            columns_lower = [col.lower() for col in df.columns]
            
            # Identify registrations data
            if any('regist' in col for col in columns_lower):
                self.registrations_df = df
                print(f"  → {name} identified as registrations data")
                
                # Standardize column names
                self.standardize_registration_columns()
            
            # Identify deposits data
            elif any('deposit' in col for col in columns_lower):
                self.deposits_df = df
                print(f"  → {name} identified as deposits data")
                self.standardize_deposit_columns()
            
            # Identify bets data
            elif any('bet' in col or 'wager' in col for col in columns_lower):
                self.bets_df = df
                print(f"  → {name} identified as bets data")
                self.standardize_bet_columns()
            
            # Identify activity data
            elif any('active' in col or 'activity' in col for col in columns_lower):
                self.activity_df = df
                print(f"  → {name} identified as activity data")
                self.standardize_activity_columns()
    
    def standardize_registration_columns(self):
        """Standardize registration dataframe columns"""
        if self.registrations_df is None:
            return
        
        # Common column mappings (update based on your actual columns)
        column_mappings = {
            'user_id': 'player_id',
            'userid': 'player_id',
            'customer_id': 'player_id',
            'reg_date': 'registration_date',
            'signup_date': 'registration_date',
            'created_at': 'registration_date',
            'channel': 'acquisition_channel',
            'source': 'acquisition_channel',
            'utm_source': 'acquisition_channel'
        }
        
        # Apply mappings
        for old_col, new_col in column_mappings.items():
            if old_col in self.registrations_df.columns:
                self.registrations_df.rename(columns={old_col: new_col}, inplace=True)
        
        # Convert date columns
        date_columns = ['registration_date']
        for col in date_columns:
            if col in self.registrations_df.columns:
                self.registrations_df[col] = pd.to_datetime(self.registrations_df[col])
    
    def standardize_deposit_columns(self):
        """Standardize deposit dataframe columns"""
        if self.deposits_df is None:
            return
        
        column_mappings = {
            'user_id': 'player_id',
            'userid': 'player_id',
            'customer_id': 'player_id',
            'deposit_date': 'first_deposit_date',
            'transaction_date': 'first_deposit_date',
            'amount': 'first_deposit_amount',
            'deposit_amount': 'first_deposit_amount'
        }
        
        for old_col, new_col in column_mappings.items():
            if old_col in self.deposits_df.columns:
                self.deposits_df.rename(columns={old_col: new_col}, inplace=True)
        
        # Convert date and numeric columns
        if 'first_deposit_date' in self.deposits_df.columns:
            self.deposits_df['first_deposit_date'] = pd.to_datetime(self.deposits_df['first_deposit_date'])
        
        if 'first_deposit_amount' in self.deposits_df.columns:
            self.deposits_df['first_deposit_amount'] = pd.to_numeric(self.deposits_df['first_deposit_amount'], errors='coerce')
    
    def standardize_bet_columns(self):
        """Standardize bet dataframe columns"""
        if self.bets_df is None:
            return
        
        column_mappings = {
            'user_id': 'player_id',
            'userid': 'player_id',
            'customer_id': 'player_id',
            'bet_date': 'first_bet_date',
            'wager_date': 'first_bet_date',
            'placed_at': 'first_bet_date'
        }
        
        for old_col, new_col in column_mappings.items():
            if old_col in self.bets_df.columns:
                self.bets_df.rename(columns={old_col: new_col}, inplace=True)
        
        if 'first_bet_date' in self.bets_df.columns:
            self.bets_df['first_bet_date'] = pd.to_datetime(self.bets_df['first_bet_date'])
    
    def standardize_activity_columns(self):
        """Standardize activity dataframe columns"""
        if self.activity_df is None:
            return
        
        column_mappings = {
            'user_id': 'player_id',
            'userid': 'player_id',
            'customer_id': 'player_id',
            'active_days': 'days_active_30',
            'days_active': 'days_active_30',
            'activity_days': 'days_active_30'
        }
        
        for old_col, new_col in column_mappings.items():
            if old_col in self.activity_df.columns:
                self.activity_df.rename(columns={old_col: new_col}, inplace=True)
        
        if 'days_active_30' in self.activity_df.columns:
            self.activity_df['days_active_30'] = pd.to_numeric(self.activity_df['days_active_30'], errors='coerce')
    
    def validate_data(self):
        """Validate that we have the minimum required data"""
        print("\nValidating data...")
        
        issues = []
        
        # Check for required dataframes
        if self.registrations_df is None or len(self.registrations_df) == 0:
            issues.append("❌ No registration data found")
        else:
            print(f"✅ Registration data: {len(self.registrations_df)} players")
        
        if self.deposits_df is None or len(self.deposits_df) == 0:
            issues.append("⚠️  No deposit data found (will affect deposit analysis)")
        else:
            print(f"✅ Deposit data: {len(self.deposits_df)} deposits")
        
        if self.bets_df is None or len(self.bets_df) == 0:
            issues.append("⚠️  No bet data found (will affect bet analysis)")
        else:
            print(f"✅ Bet data: {len(self.bets_df)} first bets")
        
        if self.activity_df is None or len(self.activity_df) == 0:
            issues.append("⚠️  No activity data found (will affect retention analysis)")
        else:
            print(f"✅ Activity data: {len(self.activity_df)} active players")
        
        # Check for required columns
        if self.registrations_df is not None:
            required_cols = ['player_id']
            missing_cols = [col for col in required_cols if col not in self.registrations_df.columns]
            if missing_cols:
                issues.append(f"❌ Missing required columns in registrations: {missing_cols}")
        
        if issues:
            print("\n⚠️  Data validation issues:")
            for issue in issues:
                print(f"  {issue}")
            print("\nThe analysis will proceed with available data, but some sections may be incomplete.")
        else:
            print("\n✅ All data validated successfully!")
        
        return len(issues) == 0
    
    def run_analysis_on_real_data(self):
        """Run the complete analysis on real data"""
        print("\n" + "="*80)
        print("PLAYER ANALYTICS - REAL DATA ANALYSIS")
        print("="*80)
        
        # Load data
        if not self.load_real_data():
            print("\n⚠️  No data files found. Please follow these steps:")
            print("1. Download the 5 CSV files from the Google Drive link")
            print("2. Save them in /workspace/data/ directory")
            print("3. Update the file names in this script")
            print("4. Run this script again")
            return
        
        # Validate data
        self.validate_data()
        
        # Run analyses with available data
        try:
            if self.registrations_df is not None:
                funnel_data, channel_df = self.funnel_conversion_analysis()
            
            if self.activity_df is not None:
                activity_enriched, deposit_bet_merge = self.retention_engagement_analysis()
            
            if self.deposits_df is not None:
                deposit_sorted, bucket_analysis, cluster_analysis = self.player_segmentation_analysis()
            
            # Create visualizations
            if self.registrations_df is not None and self.deposits_df is not None:
                self.create_visualizations(funnel_data, channel_df, activity_enriched, deposit_sorted, bucket_analysis)
            
            # Generate report
            self.generate_report()
            
            print("\n" + "="*80)
            print("REAL DATA ANALYSIS COMPLETE!")
            print("="*80)
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            print("Please check your data format and try again.")
            import traceback
            traceback.print_exc()

# Main execution
if __name__ == "__main__":
    # Instructions
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                  REAL DATA ANALYSIS SETUP                        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  1. Download your 5 datasets from Google Drive                   ║
    ║  2. Save them in /workspace/data/ directory                      ║
    ║  3. Update the filenames in this script (lines 23-29)           ║
    ║  4. Run: python3 load_real_data.py                              ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    analytics = RealDataAnalytics()
    analytics.run_analysis_on_real_data()