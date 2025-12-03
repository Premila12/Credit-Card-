import pandas as pd
import numpy as np
import joblib
import os
import json

# Paths - Updated to use active model directory
MODEL_PATH = r"c:\HDFC_Credit_Card\models\active\model.pkl"
METADATA_PATH = r"c:\HDFC_Credit_Card\models\active\metadata.json"

def predict_from_active_model(df):
    """
    Predict risk scores using the currently active model
    
    Args:
        df: DataFrame with customer data
        
    Returns:
        DataFrame with risk_score and risk_tier columns
    """
    return calculate_risk_scores(df)

def get_active_model_info():
    """
    Get information about the currently active model
    
    Returns:
        dict: Model metadata including version, metrics, deployment date
    """
    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            return {
                'version': metadata.get('version', 'unknown'),
                'date': metadata.get('deployment_date', 'N/A'),
                'accuracy': metadata.get('metrics', {}).get('accuracy', 0),
                'auc': metadata.get('metrics', {}).get('auc', 0)
            }
        else:
            return {
                'version': '1.0',
                'date': 'N/A',
                'accuracy': 0,
                'auc': 0
            }
    except Exception as e:
        print(f"Error loading model info: {e}")
        return {'version': 'unknown', 'date': 'N/A', 'accuracy': 0, 'auc': 0}

def calculate_risk_scores(df):
    """
    Calculates risk scores using the trained Random Forest model.
    """
    if df is None or df.empty:
        return None
        
    try:
        # Load Model from active directory
        if not os.path.exists(MODEL_PATH):
            # Fallback to old model path if active model doesn't exist yet
            fallback_path = r"c:\HDFC_Credit_Card\models\risk_model.pkl"
            if os.path.exists(fallback_path):
                clf = joblib.load(fallback_path)
            else:
                raise FileNotFoundError(f"Model not found. Please train the model first.")
        else:
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
