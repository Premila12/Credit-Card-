"""
Initialize Continuous Learning System
Run this once to set up the system with the current model
"""

import os
import shutil
import joblib
import json
from datetime import datetime

def initialize_system():
    """Initialize the continuous learning system"""
    
    print("="*60)
    print("Initializing Continuous Learning System")
    print("="*60)
    
    # Check if current model exists
    current_model_path = "models/risk_model.pkl"
    active_model_path = "models/active/model.pkl"
    
    if not os.path.exists(current_model_path):
        print("\n❌ Error: No trained model found at models/risk_model.pkl")
        print("Please run 'python src/train_model.py' first")
        return False
    
    # Copy current model to active directory
    print("\n1. Copying current model to active directory...")
    shutil.copy2(current_model_path, active_model_path)
    print("   ✅ Model copied")
    
    # Create initial metadata
    print("\n2. Creating initial metadata...")
    metadata = {
        "version": "1.0",
        "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "data_size": 100,  # Initial sample size
        "metrics": {
            "accuracy": 0.87,  # Placeholder - update with actual metrics
            "precision": 0.85,
            "recall": 0.82,
            "f1": 0.83,
            "auc": 0.89
        },
        "feature_importance": {},
        "deployed": True,
        "deployment_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = "models/active/metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print("   ✅ Metadata created")
    
    # Copy to versions directory
    print("\n3. Creating initial version...")
    version_model_path = "models/versions/model_v1_0.pkl"
    version_metadata_path = "models/metadata/model_v1_0.json"
    
    shutil.copy2(active_model_path, version_model_path)
    shutil.copy2(metadata_path, version_metadata_path)
    print("   ✅ Version 1.0 created")
    
    # Initialize master dataset from sample data
    print("\n4. Initializing master dataset...")
    sample_data_path = "data/sample_data.csv"
    master_data_path = "data/training/master_dataset.csv"
    
    if os.path.exists(sample_data_path):
        shutil.copy2(sample_data_path, master_data_path)
        print("   ✅ Master dataset initialized")
    else:
        print("   ⚠️  Warning: sample_data.csv not found")
    
    # Create deployment log
    print("\n5. Creating deployment log...")
    deployment_log = {
        "deployments": [
            {
                "version": "1.0",
                "action": "deployed",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "metrics": metadata["metrics"]
            }
        ]
    }
    
    with open("logs/deployments.json", 'w') as f:
        json.dump(deployment_log, f, indent=2)
    print("   ✅ Deployment log created")
    
    print("\n" + "="*60)
    print("✅ Continuous Learning System Initialized Successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Upload new data via the dashboard sidebar")
    print("2. Run manual retraining: python src/retrain_manual.py")
    print("3. Start scheduler: python src/ml_pipeline/scheduler.py")
    print("\n")
    
    return True

if __name__ == "__main__":
    initialize_system()
