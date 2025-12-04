"""
Scheduler Module
Automates the retraining pipeline on a schedule
"""

import schedule
import time
from datetime import datetime
import logging
from .data_manager import DataManager
from .model_trainer import ModelTrainer
from .model_validator import ModelValidator
from .model_deployer import ModelDeployer
from sklearn.model_selection import train_test_split

logging.basicConfig(
    filename='logs/retraining.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RetrainingScheduler:
    def __init__(self):
        self.data_manager = DataManager()
        self.model_trainer = ModelTrainer()
        self.model_validator = ModelValidator()
        self.model_deployer = ModelDeployer()
        
    def run_pipeline(self):
        """
        Execute the complete retraining pipeline
        
        Returns:
            dict: Pipeline execution report
        """
        try:
            logging.info("="*50)
            logging.info("Starting retraining pipeline")
            logging.info("="*50)
            
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'steps': {},
                'success': False
            }
            
            # Step 1: Check for new data
            new_files = self.data_manager.get_new_files()
            report['steps']['new_files_found'] = len(new_files)
            
            if not new_files:
                logging.info("No new files found, skipping retraining")
                report['steps']['status'] = 'skipped'
                return report
            
            logging.info(f"Found {len(new_files)} new files")
            
            # Step 2: Merge and clean data
            logging.info("Merging and cleaning data...")
            df = self.data_manager.merge_and_clean_data()
            report['steps']['data_merged'] = True
            report['steps']['total_rows'] = len(df)
            
            # Step 3: Train new model
            logging.info("Training new model...")
            model, metrics, version = self.model_trainer.train_model(df)
            report['steps']['model_trained'] = True
            report['steps']['new_version'] = version
            report['steps']['metrics'] = metrics
            
            # Step 4: Prepare test data for validation
            X = df[self.model_trainer.feature_columns]
            y = (df[self.model_trainer.target_column] > 0).astype(int)
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Step 5: Validate new model
            logging.info("Validating new model...")
            should_deploy, validation_report = self.model_validator.validate_model(
                version, X_test, y_test
            )
            report['steps']['validation'] = validation_report
            
            # Step 6: Deploy if validation passed
            if should_deploy:
                logging.info("Validation passed, deploying model...")
                deployment_success = self.model_deployer.deploy_model(version)
                report['steps']['deployed'] = deployment_success
                
                if deployment_success:
                    logging.info(f"Successfully deployed model version {version}")
                    report['success'] = True
                else:
                    logging.error("Deployment failed")
            else:
                logging.warning("Validation failed, model not deployed")
                report['steps']['deployed'] = False
                report['steps']['rejection_reason'] = validation_report.get('reason', [])
            
            logging.info("="*50)
            logging.info("Pipeline execution completed")
            logging.info("="*50)
            
            return report
            
        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}")
            report['error'] = str(e)
            return report
    
    def schedule_daily(self, time_str="02:00"):
        """
        Schedule daily retraining at specified time
        
        Args:
            time_str: Time in HH:MM format (24-hour)
        """
        logging.info(f"Scheduling daily retraining at {time_str}")
        schedule.every().day.at(time_str).do(self.run_pipeline)
        
    def schedule_weekly(self, day="monday", time_str="02:00"):
        """
        Schedule weekly retraining
        
        Args:
            day: Day of week
            time_str: Time in HH:MM format
        """
        logging.info(f"Scheduling weekly retraining on {day} at {time_str}")
        getattr(schedule.every(), day.lower()).at(time_str).do(self.run_pipeline)
    
    def run_scheduler(self):
        """
        Run the scheduler loop
        """
        logging.info("Scheduler started")
        print("Retraining scheduler is running...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logging.info("Scheduler stopped by user")
            print("\nScheduler stopped")

def main():
    """Main entry point for scheduler"""
    scheduler = RetrainingScheduler()
    
    # Schedule daily retraining at 2 AM
    scheduler.schedule_daily("02:00")
    
    # Run scheduler
    scheduler.run_scheduler()

if __name__ == "__main__":
    main()
