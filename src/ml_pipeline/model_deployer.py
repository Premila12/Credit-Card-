"""
Model Deployer Module
Handles safe deployment and rollback of models
"""

import joblib
import json
import shutil
import os
from datetime import datetime
import logging

logging.basicConfig(
    filename='logs/retraining.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModelDeployer:
    def __init__(self):
        self.active_dir = 'models/active'
        self.versions_dir = 'models/versions'
        self.metadata_dir = 'models/metadata'
        self.active_model_path = os.path.join(self.active_dir, 'model.pkl')
        self.active_metadata_path = os.path.join(self.active_dir, 'metadata.json')
        self.deployment_log_path = 'logs/deployments.json'
        
    def deploy_model(self, version):
        """
        Deploy a specific model version to production
        
        Args:
            version: Version string to deploy
            
        Returns:
            bool: Success status
        """
        try:
            logging.info(f"Deploying model version {version}")
            
            # Load model and metadata
            model_filename = f"model_v{version.replace('.', '_')}.pkl"
            model_path = os.path.join(self.versions_dir, model_filename)
            
            metadata_filename = f"model_v{version.replace('.', '_')}.json"
            metadata_path = os.path.join(self.metadata_dir, metadata_filename)
            
            if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                logging.error(f"Model version {version} not found")
                return False
            
            # Backup current model if exists
            if os.path.exists(self.active_model_path):
                self._backup_current_model()
            
            # Copy new model to active directory
            shutil.copy2(model_path, self.active_model_path)
            
            # Update metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['deployed'] = True
            metadata['deployment_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save updated metadata to versions dir
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Copy metadata to active dir
            with open(self.active_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Log deployment
            self.log_deployment(version, 'deployed', metadata)
            
            logging.info(f"Successfully deployed model version {version}")
            return True
            
        except Exception as e:
            logging.error(f"Error deploying model: {str(e)}")
            return False
    
    def rollback_model(self, target_version=None):
        """
        Rollback to a previous model version
        
        Args:
            target_version: Specific version to rollback to (None = previous)
            
        Returns:
            bool: Success status
        """
        try:
            if target_version is None:
                # Get previous deployed version from log
                target_version = self._get_previous_version()
            
            if target_version is None:
                logging.error("No previous version found for rollback")
                return False
            
            logging.info(f"Rolling back to version {target_version}")
            
            # Deploy the target version
            success = self.deploy_model(target_version)
            
            if success:
                self.log_deployment(target_version, 'rollback', None)
                logging.info(f"Successfully rolled back to version {target_version}")
            
            return success
            
        except Exception as e:
            logging.error(f"Error rolling back model: {str(e)}")
            return False
    
    def _backup_current_model(self):
        """Backup current active model"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_model_path = os.path.join(self.active_dir, f'model_backup_{timestamp}.pkl')
            backup_metadata_path = os.path.join(self.active_dir, f'metadata_backup_{timestamp}.json')
            
            shutil.copy2(self.active_model_path, backup_model_path)
            if os.path.exists(self.active_metadata_path):
                shutil.copy2(self.active_metadata_path, backup_metadata_path)
            
            logging.info(f"Backed up current model to {backup_model_path}")
            
        except Exception as e:
            logging.error(f"Error backing up model: {str(e)}")
    
    def log_deployment(self, version, action, metadata):
        """
        Log deployment event
        
        Args:
            version: Model version
            action: 'deployed' or 'rollback'
            metadata: Model metadata
        """
        try:
            # Load existing log
            if os.path.exists(self.deployment_log_path):
                with open(self.deployment_log_path, 'r') as f:
                    log = json.load(f)
            else:
                log = {'deployments': []}
            
            # Add new entry
            entry = {
                'version': version,
                'action': action,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': metadata.get('metrics') if metadata else None
            }
            
            log['deployments'].append(entry)
            
            # Save log
            with open(self.deployment_log_path, 'w') as f:
                json.dump(log, f, indent=2)
            
            logging.info(f"Logged deployment: {action} version {version}")
            
        except Exception as e:
            logging.error(f"Error logging deployment: {str(e)}")
    
    def _get_previous_version(self):
        """Get the previous deployed version from log"""
        try:
            if not os.path.exists(self.deployment_log_path):
                return None
            
            with open(self.deployment_log_path, 'r') as f:
                log = json.load(f)
            
            deployments = log.get('deployments', [])
            
            # Get last two deployments
            if len(deployments) >= 2:
                return deployments[-2]['version']
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting previous version: {str(e)}")
            return None
    
    def get_active_model_info(self):
        """
        Get information about currently active model
        
        Returns:
            dict: Model metadata
        """
        try:
            if os.path.exists(self.active_metadata_path):
                with open(self.active_metadata_path, 'r') as f:
                    return json.load(f)
            else:
                return {
                    'version': 'unknown',
                    'deployment_date': 'N/A',
                    'metrics': {}
                }
        except Exception as e:
            logging.error(f"Error getting active model info: {str(e)}")
            return {}
    
    def get_deployment_history(self, limit=10):
        """
        Get recent deployment history
        
        Args:
            limit: Number of recent deployments to return
            
        Returns:
            list: Deployment history
        """
        try:
            if os.path.exists(self.deployment_log_path):
                with open(self.deployment_log_path, 'r') as f:
                    log = json.load(f)
                
                deployments = log.get('deployments', [])
                return deployments[-limit:]
            else:
                return []
        except Exception as e:
            logging.error(f"Error getting deployment history: {str(e)}")
            return []
