"""
Model Trainer Module
Handles model training, evaluation, and versioning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib
import json
import os
from datetime import datetime
import logging

logging.basicConfig(
    filename='logs/retraining.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModelTrainer:
    def __init__(self):
        self.feature_columns = [
            'utilisation_pct', 'avg_payment_ratio', 'min_due_paid_frequency',
            'merchant_mix_index', 'cash_withdrawal_pct', 'recent_spend_change_pct'
        ]
        self.target_column = 'dpd_bucket_next_month'
        self.versions_dir = 'models/versions'
        self.metadata_dir = 'models/metadata'
        
    def train_model(self, df, test_size=0.2, random_state=42):
        """
        Train Random Forest model with new data
        
        Args:
            df: Training dataframe
            test_size: Test split ratio
            random_state: Random seed
            
        Returns:
            tuple: (model, metrics, version)
        """
        try:
            logging.info(f"Starting model training with {len(df)} samples")
            
            # Prepare features and target
            X = df[self.feature_columns]
            y = (df[self.target_column] > 0).astype(int)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            logging.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
            logging.info(f"Class distribution - 0: {sum(y_train==0)}, 1: {sum(y_train==1)}")
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            logging.info("Model training completed")
            
            # Calculate metrics
            metrics = self.calculate_metrics(model, X_test, y_test, X_train, y_train)
            
            # Get version number
            version = self._get_next_version()
            
            # Save model
            model_path = self.save_model_version(model, version, metrics, len(df))
            
            logging.info(f"Model saved: {model_path}")
            logging.info(f"Accuracy: {metrics['test_accuracy']:.4f}, AUC: {metrics['test_auc']:.4f}")
            
            return model, metrics, version
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise
    
    def calculate_metrics(self, model, X_test, y_test, X_train=None, y_train=None):
        """
        Calculate comprehensive model metrics
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            X_train: Training features (optional)
            y_train: Training labels (optional)
            
        Returns:
            dict: Metrics dictionary
        """
        # Test predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_pred, zero_division=0),
            'test_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Training metrics if provided
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        
        # Feature importance
        feature_importance = dict(zip(
            self.feature_columns,
            model.feature_importances_.tolist()
        ))
        metrics['feature_importance'] = feature_importance
        
        return metrics
    
    def save_model_version(self, model, version, metrics, data_size):
        """
        Save model with version number and metadata
        
        Args:
            model: Trained model
            version: Version string (e.g., "1.2")
            metrics: Metrics dictionary
            data_size: Number of training samples
            
        Returns:
            str: Path to saved model
        """
        try:
            # Save model
            model_filename = f"model_v{version.replace('.', '_')}.pkl"
            model_path = os.path.join(self.versions_dir, model_filename)
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata = {
                'version': version,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_size': data_size,
                'metrics': {
                    'accuracy': float(metrics['test_accuracy']),
                    'precision': float(metrics['test_precision']),
                    'recall': float(metrics['test_recall']),
                    'f1': float(metrics['test_f1']),
                    'auc': float(metrics['test_auc'])
                },
                'feature_importance': metrics['feature_importance'],
                'confusion_matrix': metrics['confusion_matrix'],
                'deployed': False,
                'deployment_date': None
            }
            
            metadata_filename = f"model_v{version.replace('.', '_')}.json"
            metadata_path = os.path.join(self.metadata_dir, metadata_filename)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logging.info(f"Metadata saved: {metadata_path}")
            
            return model_path
            
        except Exception as e:
            logging.error(f"Error saving model version: {str(e)}")
            raise
    
    def _get_next_version(self):
        """
        Get next version number
        
        Returns:
            str: Version string (e.g., "1.2")
        """
        try:
            # Get existing versions
            versions = []
            for filename in os.listdir(self.versions_dir):
                if filename.startswith('model_v') and filename.endswith('.pkl'):
                    # Extract version from filename
                    version_str = filename.replace('model_v', '').replace('.pkl', '').replace('_', '.')
                    try:
                        major, minor = map(int, version_str.split('.'))
                        versions.append((major, minor))
                    except:
                        continue
            
            if not versions:
                return "1.0"
            
            # Get latest version
            latest = max(versions)
            major, minor = latest
            
            # Increment minor version
            return f"{major}.{minor + 1}"
            
        except Exception as e:
            logging.error(f"Error getting next version: {str(e)}")
            return "1.0"
    
    def get_model_info(self, version):
        """
        Get metadata for a specific model version
        
        Args:
            version: Version string
            
        Returns:
            dict: Model metadata
        """
        try:
            metadata_filename = f"model_v{version.replace('.', '_')}.json"
            metadata_path = os.path.join(self.metadata_dir, metadata_filename)
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            else:
                return None
        except Exception as e:
            logging.error(f"Error getting model info: {str(e)}")
            return None
