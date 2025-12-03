import pandas as pd
import streamlit as st

def load_data(filepath):
    """
    Loads credit card data from a CSV file.
    
    Expected Columns (from User Excel):
    - customer_id
    - credit_limit
    - utilisation_%
    - avg_payment_ratio
    - min_due_paid_frequency
    - merchant_mix_index
    - cash_withdrawal_%
    - recent_spend_change_%
    - dpd_bucket_next_month
    """
    try:
        df = pd.read_csv(filepath)
        
        # Standardize column names (lowercase, strip spaces)
        df.columns = [c.strip().lower().replace(' ', '_').replace('%', 'pct') for c in df.columns]
        
        # Map to internal names if needed, or just use standardized names
        # Standardized names will be:
        # customer_id, credit_limit, utilisation_pct, avg_payment_ratio, 
        # min_due_paid_frequency, merchant_mix_index, cash_withdrawal_pct, 
        # recent_spend_change_pct, dpd_bucket_next_month
        
        required_columns = ['customer_id', 'utilisation_pct', 'avg_payment_ratio']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
            
        # Ensure data types
        df['utilisation_pct'] = pd.to_numeric(df['utilisation_pct'], errors='coerce').fillna(0)
        df['avg_payment_ratio'] = pd.to_numeric(df['avg_payment_ratio'], errors='coerce').fillna(100)
        df['cash_withdrawal_pct'] = pd.to_numeric(df['cash_withdrawal_pct'], errors='coerce').fillna(0)
        df['recent_spend_change_pct'] = pd.to_numeric(df['recent_spend_change_pct'], errors='coerce').fillna(0)
        
        return df
    except FileNotFoundError:
        st.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    except FileNotFoundError:
        st.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
