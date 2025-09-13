import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Configure visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Load datasets from CSV files"""
    # Update these paths to match your actual file names from Google Drive
    data_files = {
        'registrations': 'data/registrations.csv',
        'deposits': 'data/deposits.csv', 
        'bets': 'data/bets.csv',
        'activity': 'data/activity.csv',
        'players': 'data/players.csv'
    }
    
    loaded_data = {}
    
    # Try to load each file
    for name, filepath in data_files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                loaded_data[name] = df
                print(f"Loaded {name}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")
    
    # If no real data found, generate sample data for demonstration
    if len(loaded_data) == 0:
        print("\nGenerating sample data for demonstration...")
        return generate_sample_data()
    
    # Process and return the loaded data
    return process_loaded_data(loaded_data)

def process_loaded_data(data_dict):
    """Process loaded data and standardize column names"""
    registrations_df = None
    deposits_df = None
    bets_df = None
    activity_df = None
    
    for name, df in data_dict.items():
        columns_lower = [col.lower() for col in df.columns]
        
        # Identify and process registration data
        if any('regist' in col for col in columns_lower) or name == 'registrations':
            registrations_df = df
            # Standardize column names
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'user' in col_lower or 'player' in col_lower or 'customer' in col_lower:
                    rename_map[col] = 'player_id'
                elif 'date' in col_lower and 'reg' in col_lower:
                    rename_map[col] = 'registration_date'
                elif 'channel' in col_lower or 'source' in col_lower:
                    rename_map[col] = 'acquisition_channel'
            registrations_df.rename(columns=rename_map, inplace=True)
            
        # Identify and process deposit data
        elif any('deposit' in col for col in columns_lower) or name == 'deposits':
            deposits_df = df
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'user' in col_lower or 'player' in col_lower or 'customer' in col_lower:
                    rename_map[col] = 'player_id'
                elif 'date' in col_lower:
                    rename_map[col] = 'first_deposit_date'
                elif 'amount' in col_lower:
                    rename_map[col] = 'first_deposit_amount'
            deposits_df.rename(columns=rename_map, inplace=True)
            
        # Identify and process bet data
        elif any('bet' in col or 'wager' in col for col in columns_lower) or name == 'bets':
            bets_df = df
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'user' in col_lower or 'player' in col_lower or 'customer' in col_lower:
                    rename_map[col] = 'player_id'
                elif 'date' in col_lower:
                    rename_map[col] = 'first_bet_date'
            bets_df.rename(columns=rename_map, inplace=True)
            
        # Identify and process activity data
        elif any('active' in col or 'activity' in col for col in columns_lower) or name == 'activity':
            activity_df = df
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'user' in col_lower or 'player' in col_lower or 'customer' in col_lower:
                    rename_map[col] = 'player_id'
                elif 'days' in col_lower or 'active' in col_lower:
                    rename_map[col] = 'days_active_30'
            activity_df.rename(columns=rename_map, inplace=True)
    
    # Convert date columns to datetime
    if registrations_df is not None and 'registration_date' in registrations_df.columns:
        registrations_df['registration_date'] = pd.to_datetime(registrations_df['registration_date'])
    if deposits_df is not None and 'first_deposit_date' in deposits_df.columns:
        deposits_df['first_deposit_date'] = pd.to_datetime(deposits_df['first_deposit_date'])
    if bets_df is not None and 'first_bet_date' in bets_df.columns:
        bets_df['first_bet_date'] = pd.to_datetime(bets_df['first_bet_date'])
    
    return registrations_df, deposits_df, bets_df, activity_df

def generate_sample_data():
    """Generate sample data if no real data is available"""
    np.random.seed(42)
    
    # Generate sample player data
    n_players = 10000
    start_date = pd.Timestamp('2024-01-01')
    reg_dates = pd.date_range(start_date, periods=90, freq='D')
    
    # Registration data
    registrations = []
    for i in range(n_players):
        player_id = f"P{i+1:05d}"
        reg_date = np.random.choice(reg_dates)
        channel = np.random.choice(
            ['Organic', 'Paid Search', 'Social Media', 'Affiliate', 'Direct'],
            p=[0.25, 0.30, 0.20, 0.15, 0.10]
        )
        registrations.append({
            'player_id': player_id,
            'registration_date': reg_date,
            'acquisition_channel': channel
        })
    
    registrations_df = pd.DataFrame(registrations)
    
    # Generate deposits
    deposits = []
    for _, player in registrations_df.iterrows():
        # Channel-based deposit probability
        if player['acquisition_channel'] in ['Paid Search', 'Affiliate']:
            deposit_prob = 0.45
        else:
            deposit_prob = 0.30
            
        if np.random.random() < deposit_prob:
            days_to_deposit = min(np.random.exponential(3), 30)
            deposit_date = player['registration_date'] + timedelta(days=days_to_deposit)
            
            # Deposit amount distribution
            if np.random.random() < 0.70:
                amount = np.random.lognormal(3, 0.5)
            elif np.random.random() < 0.95:
                amount = np.random.lognormal(4.5, 0.5)
            else:
                amount = np.random.lognormal(6, 0.8)
            
            deposits.append({
                'player_id': player['player_id'],
                'first_deposit_date': deposit_date,
                'first_deposit_amount': round(amount, 2),
                'acquisition_channel': player['acquisition_channel'],
                'registration_date': player['registration_date']
            })
    
    deposits_df = pd.DataFrame(deposits)
    
    # Generate bets
    bets = []
    deposited_players = set(deposits_df['player_id'])
    
    for _, player in registrations_df.iterrows():
        if player['player_id'] in deposited_players:
            bet_prob = 0.90
        else:
            bet_prob = 0.15
            
        if np.random.random() < bet_prob:
            if player['player_id'] in deposited_players:
                deposit_info = deposits_df[deposits_df['player_id'] == player['player_id']].iloc[0]
                bet_date = deposit_info['first_deposit_date'] + timedelta(hours=np.random.exponential(12))
            else:
                days_to_bet = np.random.exponential(2)
                bet_date = player['registration_date'] + timedelta(days=days_to_bet)
            
            bets.append({
                'player_id': player['player_id'],
                'first_bet_date': bet_date,
                'acquisition_channel': player['acquisition_channel'],
                'registration_date': player['registration_date']
            })
    
    bets_df = pd.DataFrame(bets)
    
    # Generate activity data
    activity = []
    for _, player in registrations_df.iterrows():
        active_prob = 0.35 if player['player_id'] in deposited_players else 0.20
        
        if np.random.random() < active_prob:
            if player['player_id'] in deposited_players:
                days_active = max(1, np.random.binomial(30, 0.4))
            else:
                days_active = max(1, np.random.binomial(30, 0.15))
            
            activity.append({
                'player_id': player['player_id'],
                'days_active_30': days_active,
                'acquisition_channel': player['acquisition_channel'],
                'registration_date': player['registration_date']
            })
    
    activity_df = pd.DataFrame(activity)
    
    return registrations_df, deposits_df, bets_df, activity_df

def funnel_analysis(registrations_df, deposits_df, bets_df, activity_df):
    """Task 1: Funnel & Conversion Analysis"""
    print("\n" + "="*60)
    print("FUNNEL & CONVERSION ANALYSIS")
    print("="*60)
    
    # Calculate funnel stages
    total_registrations = len(registrations_df) if registrations_df is not None else 0
    total_first_deposits = len(deposits_df) if deposits_df is not None else 0
    total_first_bets = len(bets_df) if bets_df is not None else 0
    total_active_30 = len(activity_df) if activity_df is not None else 0
    
    # Conversion rates
    reg_to_deposit = (total_first_deposits / total_registrations * 100) if total_registrations > 0 else 0
    reg_to_bet = (total_first_bets / total_registrations * 100) if total_registrations > 0 else 0
    reg_to_active = (total_active_30 / total_registrations * 100) if total_registrations > 0 else 0
    
    print(f"\nFunnel Stages:")
    print(f"Registrations: {total_registrations:,}")
    print(f"First Deposits: {total_first_deposits:,} ({reg_to_deposit:.1f}% conversion)")
    print(f"First Bets: {total_first_bets:,} ({reg_to_bet:.1f}% conversion)")
    print(f"Active in 30 days: {total_active_30:,} ({reg_to_active:.1f}% conversion)")
    
    # Identify largest drop-off
    drop_offs = {
        'Registration → Deposit': 100 - reg_to_deposit,
        'Registration → Bet': 100 - reg_to_bet,
        'Registration → Active': 100 - reg_to_active
    }
    
    if drop_offs:
        max_dropoff = max(drop_offs.items(), key=lambda x: x[1])
        print(f"\nLargest drop-off: {max_dropoff[0]} ({max_dropoff[1]:.1f}% loss)")
    
    # Channel analysis if data available
    if registrations_df is not None and 'acquisition_channel' in registrations_df.columns:
        print("\nConversion by Acquisition Channel:")
        for channel in registrations_df['acquisition_channel'].unique():
            channel_regs = len(registrations_df[registrations_df['acquisition_channel'] == channel])
            if deposits_df is not None and 'acquisition_channel' in deposits_df.columns:
                channel_deps = len(deposits_df[deposits_df['acquisition_channel'] == channel])
                channel_conv = (channel_deps / channel_regs * 100) if channel_regs > 0 else 0
                print(f"  {channel}: {channel_conv:.1f}% deposit conversion")
    
    return {
        'registrations': total_registrations,
        'deposits': total_first_deposits,
        'bets': total_first_bets,
        'active_30': total_active_30,
        'conversion_rates': {
            'deposit': reg_to_deposit,
            'bet': reg_to_bet,
            'active': reg_to_active
        }
    }

def retention_analysis(registrations_df, deposits_df, bets_df, activity_df):
    """Task 2: Retention & Engagement Analysis"""
    print("\n" + "="*60)
    print("RETENTION & ENGAGEMENT ANALYSIS")
    print("="*60)
    
    if activity_df is None or len(activity_df) == 0:
        print("No activity data available for retention analysis")
        return None, None
    
    # Merge activity with deposits if available
    if deposits_df is not None and 'player_id' in deposits_df.columns:
        activity_enriched = activity_df.merge(
            deposits_df[['player_id', 'first_deposit_amount']], 
            on='player_id', 
            how='left'
        )
    else:
        activity_enriched = activity_df.copy()
        activity_enriched['first_deposit_amount'] = 0
    
    # Create cohorts
    if 'days_active_30' in activity_enriched.columns:
        activity_enriched['cohort'] = pd.cut(
            activity_enriched['days_active_30'],
            bins=[0, 2, 5, 10, 30],
            labels=['1-2 days', '3-5 days', '6-10 days', '11-30 days'],
            include_lowest=True
        )
        
        print("\nActivity Cohorts (Days Active in First 30 Days):")
        for cohort in ['1-2 days', '3-5 days', '6-10 days', '11-30 days']:
            cohort_data = activity_enriched[activity_enriched['cohort'] == cohort]
            count = len(cohort_data)
            pct = (count / len(activity_enriched) * 100) if len(activity_enriched) > 0 else 0
            
            # Calculate deposit contribution
            cohort_deposits = cohort_data['first_deposit_amount'].sum()
            total_deposits = activity_enriched['first_deposit_amount'].sum()
            deposit_pct = (cohort_deposits / total_deposits * 100) if total_deposits > 0 else 0
            
            print(f"  {cohort}: {count} players ({pct:.1f}%), ${cohort_deposits:,.0f} deposits ({deposit_pct:.1f}%)")
    
    # Time gap analysis if both deposits and bets available
    deposit_bet_merge = None
    if deposits_df is not None and bets_df is not None:
        if 'player_id' in deposits_df.columns and 'player_id' in bets_df.columns:
            deposit_bet_merge = deposits_df.merge(
                bets_df[['player_id', 'first_bet_date']], 
                on='player_id', 
                how='inner'
            )
            
            if 'first_deposit_date' in deposit_bet_merge.columns and 'first_bet_date' in deposit_bet_merge.columns:
                deposit_bet_merge['gap_days'] = (
                    deposit_bet_merge['first_bet_date'] - deposit_bet_merge['first_deposit_date']
                ).dt.total_seconds() / 86400
                
                print(f"\nTime Gap (First Deposit → First Bet):")
                print(f"  Mean: {deposit_bet_merge['gap_days'].mean():.1f} days")
                print(f"  Median: {deposit_bet_merge['gap_days'].median():.1f} days")
                print(f"  75th percentile: {deposit_bet_merge['gap_days'].quantile(0.75):.1f} days")
                print(f"  Maximum: {deposit_bet_merge['gap_days'].max():.1f} days")
                
                # Distribution shape analysis
                if deposit_bet_merge['gap_days'].median() < deposit_bet_merge['gap_days'].mean():
                    print("\nDistribution: Right-skewed (most players bet quickly, some take longer)")
                else:
                    print("\nDistribution: Left-skewed or symmetric")
    
    return activity_enriched, deposit_bet_merge

def segmentation_analysis(deposits_df):
    """Task 3: Player Segmentation Analysis"""
    print("\n" + "="*60)
    print("PLAYER SEGMENTATION ANALYSIS")
    print("="*60)
    
    if deposits_df is None or len(deposits_df) == 0:
        print("No deposit data available for segmentation analysis")
        return None
    
    # Ensure we have the required column
    if 'first_deposit_amount' not in deposits_df.columns:
        print("Missing 'first_deposit_amount' column in deposits data")
        return None
    
    # Sort by deposit amount
    deposits_sorted = deposits_df.sort_values('first_deposit_amount', ascending=False)
    
    # Top 10% analysis
    top_10_pct_count = int(len(deposits_sorted) * 0.1)
    if top_10_pct_count > 0:
        top_10_pct = deposits_sorted.head(top_10_pct_count)
        
        total_deposits = deposits_sorted['first_deposit_amount'].sum()
        top_10_pct_amount = top_10_pct['first_deposit_amount'].sum()
        top_10_pct_share = (top_10_pct_amount / total_deposits * 100) if total_deposits > 0 else 0
        
        print(f"\nTop 10% of Depositors:")
        print(f"  Count: {top_10_pct_count} players")
        print(f"  Total deposits: ${top_10_pct_amount:,.2f}")
        print(f"  Share of total: {top_10_pct_share:.1f}%")
        
        # Concentration analysis
        print("\nDeposit Concentration:")
        if top_10_pct_share > 50:
            print(f"  High concentration: Top 10% control {top_10_pct_share:.1f}% of deposits")
        elif top_10_pct_share > 35:
            print(f"  Moderate concentration: Top 10% control {top_10_pct_share:.1f}% of deposits")
        else:
            print(f"  Low concentration: Top 10% control {top_10_pct_share:.1f}% of deposits")
    
    # Create deposit buckets
    buckets = [0, 10, 25, 50, 100, 250, 500, 1000, float('inf')]
    labels = ['$0-10', '$10-25', '$25-50', '$50-100', '$100-250', '$250-500', '$500-1000', '$1000+']
    
    deposits_sorted['bucket'] = pd.cut(
        deposits_sorted['first_deposit_amount'],
        bins=buckets,
        labels=labels,
        include_lowest=True
    )
    
    print("\nDeposit Amount Buckets:")
    for bucket in labels:
        bucket_data = deposits_sorted[deposits_sorted['bucket'] == bucket]
        if len(bucket_data) > 0:
            count = len(bucket_data)
            pct = (count / len(deposits_sorted) * 100)
            total = bucket_data['first_deposit_amount'].sum()
            print(f"  {bucket}: {count} players ({pct:.1f}%), ${total:,.0f} total")
    
    # Clustering insights
    print("\nClustering Observations:")
    if len(deposits_sorted) > 0:
        modal_bucket = deposits_sorted['bucket'].mode()[0]
        print(f"  Most common deposit range: {modal_bucket}")
        
        # Check for natural clusters
        if len(deposits_sorted[deposits_sorted['first_deposit_amount'] < 20]) / len(deposits_sorted) > 0.3:
            print("  Large micro-deposit segment detected (<$20)")
        if len(deposits_sorted[deposits_sorted['first_deposit_amount'] > 250]) > 0:
            print(f"  High-value segment: {len(deposits_sorted[deposits_sorted['first_deposit_amount'] > 250])} players depositing >$250")
        
        # Profitability indicator
        print(f"\nFirst deposit amounts indicate:")
        avg_deposit = deposits_sorted['first_deposit_amount'].mean()
        median_deposit = deposits_sorted['first_deposit_amount'].median()
        if avg_deposit > median_deposit * 1.5:
            print(f"  Whale-driven economy (mean ${avg_deposit:.2f} >> median ${median_deposit:.2f})")
        else:
            print(f"  Balanced distribution (mean ${avg_deposit:.2f}, median ${median_deposit:.2f})")
    
    return deposits_sorted

def create_visualizations(funnel_metrics, activity_df, deposits_df):
    """Create visualizations for the analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Funnel Chart
    stages = ['Registration', 'First Deposit', 'First Bet', 'Active 30d']
    values = [
        funnel_metrics['registrations'],
        funnel_metrics['deposits'],
        funnel_metrics['bets'],
        funnel_metrics['active_30']
    ]
    
    ax1 = axes[0, 0]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax1.bar(stages, values, color=colors)
    ax1.set_ylabel('Number of Players')
    ax1.set_title('Player Funnel Analysis')
    
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val > 0:
            height = bar.get_height()
            pct = (val / values[0] * 100) if values[0] > 0 else 0
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:,}\n({pct:.1f}%)',
                    ha='center', va='bottom')
    
    # 2. Channel Performance
    ax2 = axes[0, 1]
    if deposits_df is not None and 'acquisition_channel' in deposits_df.columns:
        channel_data = deposits_df.groupby('acquisition_channel').size()
        channel_data.plot(kind='bar', ax=ax2, color='skyblue')
        ax2.set_title('Deposits by Acquisition Channel')
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Number of Deposits')
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No channel data available', ha='center', va='center')
        ax2.set_title('Deposits by Acquisition Channel')
    
    # 3. Activity Distribution
    ax3 = axes[1, 0]
    if activity_df is not None and 'days_active_30' in activity_df.columns:
        activity_df['days_active_30'].hist(bins=30, ax=ax3, color='green', alpha=0.7)
        ax3.set_xlabel('Days Active (First 30 Days)')
        ax3.set_ylabel('Number of Players')
        ax3.set_title('Player Activity Distribution')
    else:
        ax3.text(0.5, 0.5, 'No activity data available', ha='center', va='center')
        ax3.set_title('Player Activity Distribution')
    
    # 4. Deposit Distribution
    ax4 = axes[1, 1]
    if deposits_df is not None and 'first_deposit_amount' in deposits_df.columns:
        deposits_df['first_deposit_amount'].hist(bins=50, ax=ax4, color='orange', alpha=0.7)
        ax4.set_xlabel('First Deposit Amount ($)')
        ax4.set_ylabel('Number of Players')
        ax4.set_title('First Deposit Distribution')
        ax4.set_xlim(0, min(500, deposits_df['first_deposit_amount'].max()))
    else:
        ax4.text(0.5, 0.5, 'No deposit data available', ha='center', va='center')
        ax4.set_title('First Deposit Distribution')
    
    plt.tight_layout()
    plt.savefig('player_analysis_charts.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved as 'player_analysis_charts.png'")

def main():
    """Main analysis execution"""
    print("PLAYER ANALYTICS ANALYSIS")
    print("="*60)
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        print("\nCreated 'data' directory.")
        print("Please place your CSV files from Google Drive in the 'data' folder and run again.")
    
    # Load data
    print("\nLoading data...")
    registrations_df, deposits_df, bets_df, activity_df = load_data()
    
    if registrations_df is not None:
        print(f"Data loaded: {len(registrations_df)} registrations")
    else:
        print("Warning: No registration data available")
    
    # Task 1: Funnel Analysis
    funnel_metrics = funnel_analysis(registrations_df, deposits_df, bets_df, activity_df)
    
    # Task 2: Retention Analysis
    activity_enriched, deposit_bet_merge = retention_analysis(
        registrations_df, deposits_df, bets_df, activity_df
    )
    
    # Task 3: Segmentation Analysis
    deposits_sorted = segmentation_analysis(deposits_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(funnel_metrics, activity_df, deposits_df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()