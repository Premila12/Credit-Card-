"""
Manual Retraining Script
Run this to manually trigger the retraining pipeline
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.scheduler import RetrainingScheduler

def main():
    print("="*60)
    print("HDFC Credit Risk Model - Manual Retraining")
    print("="*60)
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION REPORT")
    print("="*60)
    
    print(f"\nTimestamp: {report['timestamp']}")
    print(f"Success: {report.get('success', False)}")
    
    if 'steps' in report:
        steps = report['steps']
        print(f"\nNew files found: {steps.get('new_files_found', 0)}")
        
        if steps.get('data_merged'):
            print(f"Total training rows: {steps.get('total_rows', 0)}")
        
        if steps.get('model_trained'):
            print(f"New model version: {steps.get('new_version', 'N/A')}")
            metrics = steps.get('metrics', {})
            print(f"Accuracy: {metrics.get('test_accuracy', 0):.4f}")
            print(f"AUC: {metrics.get('test_auc', 0):.4f}")
        
        if 'validation' in steps:
            validation = steps['validation']
            print(f"\nValidation: {validation.get('recommendation', 'N/A')}")
            if validation.get('reason'):
                print("Reasons:")
                for reason in validation['reason']:
                    print(f"  - {reason}")
        
        if steps.get('deployed'):
            print("\n✅ Model deployed successfully!")
        elif steps.get('deployed') == False:
            print("\n❌ Model deployment rejected")
    
    if 'error' in report:
        print(f"\n❌ Error: {report['error']}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
