import pandas as pd
import os

# Path to the Excel file
excel_path = r"c:\HDFC_Credit_Card\data\Credit Card Delinquency Watch.xlsx"
csv_path = r"c:\HDFC_Credit_Card\data\sample_data.csv"

try:
    # Read the Excel file - Sheet 'Sample'
    df = pd.read_excel(excel_path, sheet_name='Sample')
    
    # Standardize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Successfully converted {excel_path} to {csv_path}")
    print("Columns found:", df.columns.tolist())
    print(df.head())
    
except Exception as e:
    print(f"Error converting file: {e}")
