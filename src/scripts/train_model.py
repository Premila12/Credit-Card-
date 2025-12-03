import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Paths
DATA_PATH = r"c:\HDFC_Credit_Card\data\sample_data.csv"
MODEL_DIR = r"c:\HDFC_Credit_Card\models"
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")

def train():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Target: dpd_bucket_next_month > 0 (Delinquent)
    # 0 = No Delay, 1+ = Delay
    df['target'] = df['dpd_bucket_next_month'].apply(lambda x: 1 if x > 0 else 0)
    
    # Features
    features = ['utilisation_pct', 'avg_payment_ratio', 'min_due_paid_frequency', 
                'merchant_mix_index', 'cash_withdrawal_pct', 'recent_spend_change_pct']
    
    X = df[features]
    y = df['target']
    
    # Handle missing values
    X = X.fillna(0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    print("Training Random Forest Model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
