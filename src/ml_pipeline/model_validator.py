"""
Model Validator Module
Validates new models against current production model
"""

import joblib
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import logging

logging.basicConfig(
    filename='logs/retraining.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModelValidator:
    def __init__(self):
        self.active_model_path = 'models/active/model.pkl'
        self.active_metadata_path = 'models/active/metadata.json'
        self.metadata_dir = 'models/metadata'
        
        # Validation thresholds
        self.min_accuracy = 0.70
        self.min_auc = 0.65
        self.max_accuracy_drop = 0.05  # Max 5% drop allowed
        self.max_drift_threshold = 0.15  # Max 15% drift
        
    def validate_model(self, new_model_version, X_test, y_test):
        """
        Validate new model against current production model
        
        Args:
            new_model_version: Version string of new model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            tuple: (should_deploy: bool, validation_report: dict)
        """
        try:
            logging.info(f"Validating model version {new_model_version}")
            
            # Load new model and metadata
            new_model, new_metadata = self._load_model_version(new_model_version)
            
            if new_model is None:
                return False, {'error': 'New model not found'}
            
            # Load current active model
            current_model, current_metadata = self._load_active_model()
            
            # Validation report
            report = {
                'new_version': new_model_version,
                'current_version': current_metadata.get('version', 'unknown') if current_metadata else 'none',
                'checks': {},
                'recommendation': None,
                'reason': []
            }
            
            # Check 1: Minimum performance thresholds
            new_accuracy = new_metadata['metrics']['accuracy']
            new_auc = new_metadata['metrics']['auc']
            
            report['checks']['min_accuracy'] = new_accuracy >= self.min_accuracy
            report['checks']['min_auc'] = new_auc >= self.min_auc
            
            if not report['checks']['min_accuracy']:
                report['reason'].append(f"Accuracy {new_accuracy:.4f} below minimum {self.min_accuracy}")
            
            if not report['checks']['min_auc']:
                report['reason'].append(f"AUC {new_auc:.4f} below minimum {self.min_auc}")
            
            # Check 2: Compare with current model
            if current_model is not None and current_metadata is not None:
                current_accuracy = current_metadata['metrics']['accuracy']
                current_auc = current_metadata['metrics']['auc']
                
                accuracy_diff = new_accuracy - current_accuracy
                auc_diff = new_auc - current_auc
                
                report['checks']['accuracy_improvement'] = accuracy_diff >= -self.max_accuracy_drop
                report['checks']['auc_improvement'] = auc_diff >= 0
                
                report['metrics_comparison'] = {
                    'accuracy': {
                        'current': current_accuracy,
                        'new': new_accuracy,
                        'diff': accuracy_diff
                    },
                    'auc': {
                        'current': current_auc,
                        'new': new_auc,
                        'diff': auc_diff
                    }
                }
                
                if not report['checks']['accuracy_improvement']:
                    report['reason'].append(f"Accuracy dropped by {abs(accuracy_diff):.4f} (max allowed: {self.max_accuracy_drop})")
                
                # Check 3: Drift detection
                drift_score = self.check_drift(new_model, current_model, X_test)
                report['checks']['drift_acceptable'] = drift_score < self.max_drift_threshold
                report['drift_score'] = drift_score
                
                if not report['checks']['drift_acceptable']:
                    report['reason'].append(f"Prediction drift {drift_score:.4f} exceeds threshold {self.max_drift_threshold}")
            
            else:
                # No current model, deploy if meets minimum thresholds
                report['checks']['accuracy_improvement'] = True
                report['checks']['auc_improvement'] = True
                report['checks']['drift_acceptable'] = True
                report['reason'].append("No current model, deploying if minimum thresholds met")
            
            # Final decision
            all_checks_passed = all(report['checks'].values())
            
            if all_checks_passed:
                report['recommendation'] = 'DEPLOY'
                report['reason'].append("All validation checks passed")
                logging.info(f"Validation PASSED for model {new_model_version}")
            else:
                report['recommendation'] = 'REJECT'
                logging.warning(f"Validation FAILED for model {new_model_version}: {report['reason']}")
            
            return all_checks_passed, report
            
        except Exception as e:
            logging.error(f"Error validating model: {str(e)}")
            return False, {'error': str(e)}
    
    def check_drift(self, new_model, current_model, X_test):
        """
        Check prediction drift between models
        
        Args:
            new_model: New trained model
            current_model: Current production model
            X_test: Test features
            
        Returns:
            float: Drift score (0-1, lower is better)
        """
        try:
            # Get predictions from both models
            new_pred = new_model.predict_proba(X_test)[:, 1]
            current_pred = current_model.predict_proba(X_test)[:, 1]
            
            # Calculate mean absolute difference
            drift = np.mean(np.abs(new_pred - current_pred))
            
            logging.info(f"Prediction drift: {drift:.4f}")
            return drift
            
        except Exception as e:
            logging.error(f"Error checking drift: {str(e)}")
            return 1.0  # Return high drift on error
    
    def _load_model_version(self, version):
        """Load specific model version and metadata"""
        try:
            model_filename = f"model_v{version.replace('.', '_')}.pkl"
            model_path = os.path.join('models/versions', model_filename)
            
            metadata_filename = f"model_v{version.replace('.', '_')}.json"
            metadata_path = os.path.join(self.metadata_dir, metadata_filename)
            
            if os.path.exists(model_path) and os.path.exists(metadata_path):
                model = joblib.load(model_path)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return model, metadata
            else:
                return None, None
        except Exception as e:
            logging.error(f"Error loading model version {version}: {str(e)}")
            return None, None
    
    def _load_active_model(self):
        """Load current active model and metadata"""
        try:
            if os.path.exists(self.active_model_path) and os.path.exists(self.active_metadata_path):
                model = joblib.load(self.active_model_path)
                with open(self.active_metadata_path, 'r') as f:
                    metadata = json.load(f)
                return model, metadata
            else:
                return None, None
        except Exception as e:
            logging.error(f"Error loading active model: {str(e)}")
            return None, None
    
    def should_deploy(self, validation_report):
        """
        Determine if model should be deployed based on validation report
        
        Args:
            validation_report: Validation report dict
            
        Returns:
            bool: True if should deploy
        """
        return validation_report.get('recommendation') == 'DEPLOY'
