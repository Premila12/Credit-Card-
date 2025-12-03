import pandas as pd
import numpy as np
import joblib
import os

# Paths
MODEL_PATH = r"c:\HDFC_Credit_Card\models\risk_model.pkl"

def calculate_risk_scores(df):
    """
    Calculates risk scores using the trained Random Forest model.
    """
    if df is None or df.empty:
        return None
        
    try:
        # Load Model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
            
        clf = joblib.load(MODEL_PATH)
        
        # Features (Must match training exactly)
        features = ['utilisation_pct', 'avg_payment_ratio', 'min_due_paid_frequency', 
                    'merchant_mix_index', 'cash_withdrawal_pct', 'recent_spend_change_pct']
        
        # Prepare X (Handle missing values same as training)
        X = df[features].fillna(0)
        
        # Predict Probability of Delinquency (Class 1)
        # predict_proba returns [prob_0, prob_1]
        probs = clf.predict_proba(X)[:, 1]
        
        # Scale to 0-100 Risk Score
        df['risk_score'] = (probs * 100).round(1)
        
        # --- Risk Tiers ---
        def assign_tier(score):
            if score >= 60:
                return 'Intervene'  # High Risk -> Intervene (Red)
            elif score >= 30:
                return 'Engage'     # Emerging Risk -> Engage (Yellow)
            else:
                return 'Monitor'    # Low Risk -> Monitor (Blue)
                
        df['risk_tier'] = df['risk_score'].apply(assign_tier)
        
        return df
        
    except Exception as e:
        print(f"Error in risk scoring: {e}")
        # Fallback to 0 if model fails (or handle differently)
        df['risk_score'] = 0
        df['risk_tier'] = 'Error'
        return df
