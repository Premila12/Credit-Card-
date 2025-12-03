import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from risk_engine import calculate_risk_scores

# Load sample data
df = pd.read_csv(r'c:\HDFC_Credit_Card\data\sample_data.csv')

# Run risk engine
print("Running Risk Engine...")
df_scored = calculate_risk_scores(df)

if df_scored is not None:
    print("Risk Scoring Complete.")
    print(df_scored[['customer_id', 'risk_score', 'risk_tier']].head())
    
    # Check if scores are varying
    print("\nScore Distribution:")
    print(df_scored['risk_score'].describe())
    
    if df_scored['risk_score'].sum() == 0:
        print("\nWARNING: All scores are 0. Model might not be working.")
    else:
        print("\nSUCCESS: Risk scores generated.")
else:
    print("Error: Risk engine returned None.")
