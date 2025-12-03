from data_loader import load_data
from risk_engine import calculate_risk_scores
import pandas as pd

# Load data
print("Loading data...")
df = load_data(r"c:\HDFC_Credit_Card\data\sample_data.csv")

if df is not None:
    print("Data loaded successfully.")
    print("Columns:", df.columns.tolist())
    
    # Calculate risk
    print("Calculating risk...")
    scored_df = calculate_risk_scores(df)
    
    if scored_df is not None:
        print("Risk calculation successful.")
        print(scored_df[['customer_id', 'risk_score', 'risk_tier']].head())
    else:
        print("Risk calculation failed.")
else:
    print("Data loading failed.")
