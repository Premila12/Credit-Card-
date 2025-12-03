"""
Model Rollback Script
Rollback to a previous model version
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml_pipeline.model_deployer import ModelDeployer

def main():
    parser = argparse.ArgumentParser(description='Rollback to a previous model version')
    parser.add_argument('--version', type=str, help='Specific version to rollback to (e.g., 1.2)')
    parser.add_argument('--list', action='store_true', help='List deployment history')
    
    args = parser.parse_args()
    
    deployer = ModelDeployer()
    
    if args.list:
        print("="*60)
        print("DEPLOYMENT HISTORY")
        print("="*60)
        
        history = deployer.get_deployment_history(limit=10)
        
        if not history:
            print("\nNo deployment history found")
        else:
            for entry in reversed(history):
                print(f"\nVersion: {entry['version']}")
                print(f"Action: {entry['action']}")
                print(f"Timestamp: {entry['timestamp']}")
                if entry.get('metrics'):
                    print(f"Accuracy: {entry['metrics'].get('accuracy', 'N/A')}")
                print("-"*60)
    
    else:
        print("="*60)
        print("MODEL ROLLBACK")
        print("="*60)
        
        if args.version:
            print(f"\nRolling back to version {args.version}...")
            success = deployer.rollback_model(target_version=args.version)
        else:
            print("\nRolling back to previous version...")
            success = deployer.rollback_model()
        
        if success:
            print("\n✅ Rollback successful!")
            
            # Show active model info
            info = deployer.get_active_model_info()
            print(f"\nActive model: {info.get('version', 'unknown')}")
            print(f"Deployed: {info.get('deployment_date', 'N/A')}")
        else:
            print("\n❌ Rollback failed!")

if __name__ == "__main__":
    main()
