"""
Data Manager Module
Handles data ingestion, storage, merging, and cleaning for continuous learning.
"""

import pandas as pd
import os
from datetime import datetime
import shutil
import logging

# Setup logging
logging.basicConfig(
    filename='logs/retraining.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataManager:
    def __init__(self):
        self.new_data_dir = 'data/new'
        self.training_data_dir = 'data/training'
        self.archive_dir = 'data/archive'
        self.master_dataset_path = os.path.join(self.training_data_dir, 'master_dataset.csv')
        
    def store_new_data(self, uploaded_file, filename=None):
        """
        Store uploaded CSV file with timestamp
        
        Args:
            uploaded_file: File object from Streamlit uploader
            filename: Optional custom filename
            
        Returns:
            str: Path to saved file
        """
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create filename
            if filename is None:
                filename = f"upload_{timestamp}.csv"
            else:
                name, ext = os.path.splitext(filename)
                filename = f"{name}_{timestamp}{ext}"
            
            # Save file
            filepath = os.path.join(self.new_data_dir, filename)
            
            # Read and save
            if hasattr(uploaded_file, 'read'):
                df = pd.read_csv(uploaded_file)
                df.to_csv(filepath, index=False)
            else:
                # If it's already a dataframe
                uploaded_file.to_csv(filepath, index=False)
            
            logging.info(f"New data stored: {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Error storing new data: {str(e)}")
            raise
    
    def get_new_files(self):
        """
        Get list of new files waiting to be processed
        
        Returns:
            list: List of file paths
        """
        try:
            files = [
                os.path.join(self.new_data_dir, f) 
                for f in os.listdir(self.new_data_dir) 
                if f.endswith('.csv')
            ]
            return sorted(files)
        except Exception as e:
            logging.error(f"Error getting new files: {str(e)}")
            return []
    
    def merge_and_clean_data(self):
        """
        Merge new data with master dataset, clean and deduplicate
        
        Returns:
            pd.DataFrame: Updated master dataset
        """
        try:
            # Get new files
            new_files = self.get_new_files()
            
            if not new_files:
                logging.info("No new files to process")
                return self.get_training_data()
            
            logging.info(f"Processing {len(new_files)} new files")
            
            # Load existing master dataset
            if os.path.exists(self.master_dataset_path):
                master_df = pd.read_csv(self.master_dataset_path)
                logging.info(f"Loaded master dataset: {len(master_df)} rows")
            else:
                master_df = pd.DataFrame()
                logging.info("No existing master dataset, creating new one")
            
            # Merge new files
            new_data_frames = []
            for filepath in new_files:
                df = pd.read_csv(filepath)
                new_data_frames.append(df)
                logging.info(f"Loaded {filepath}: {len(df)} rows")
            
            # Combine all new data
            if new_data_frames:
                new_df = pd.concat(new_data_frames, ignore_index=True)
                
                # Combine with master
                if not master_df.empty:
                    combined_df = pd.concat([master_df, new_df], ignore_index=True)
                else:
                    combined_df = new_df
                
                # Clean and deduplicate
                combined_df = self._clean_data(combined_df)
                
                # Save updated master dataset
                combined_df.to_csv(self.master_dataset_path, index=False)
                logging.info(f"Updated master dataset: {len(combined_df)} rows")
                
                # Archive processed files
                self._archive_processed_files(new_files)
                
                return combined_df
            
            return master_df
            
        except Exception as e:
            logging.error(f"Error merging data: {str(e)}")
            raise
    
    def _clean_data(self, df):
        """
        Clean and deduplicate data
        
        Args:
            df: DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        # Remove duplicates based on customer_id
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['customer_id'], keep='last')
        removed = initial_rows - len(df)
        
        if removed > 0:
            logging.info(f"Removed {removed} duplicate rows")
        
        # Remove rows with missing critical values
        critical_columns = ['customer_id', 'credit_limit', 'utilisation_pct']
        df = df.dropna(subset=critical_columns)
        
        # Sort by customer_id
        df = df.sort_values('customer_id').reset_index(drop=True)
        
        return df
    
    def _archive_processed_files(self, files):
        """
        Move processed files to archive
        
        Args:
            files: List of file paths to archive
        """
        try:
            for filepath in files:
                filename = os.path.basename(filepath)
                archive_path = os.path.join(self.archive_dir, filename)
                shutil.move(filepath, archive_path)
                logging.info(f"Archived: {filename}")
        except Exception as e:
            logging.error(f"Error archiving files: {str(e)}")
    
    def get_training_data(self):
        """
        Load master training dataset
        
        Returns:
            pd.DataFrame: Training data
        """
        try:
            if os.path.exists(self.master_dataset_path):
                df = pd.read_csv(self.master_dataset_path)
                logging.info(f"Loaded training data: {len(df)} rows")
                return df
            else:
                # Fallback to sample data
                sample_path = 'data/sample_data.csv'
                if os.path.exists(sample_path):
                    df = pd.read_csv(sample_path)
                    # Save as master
                    df.to_csv(self.master_dataset_path, index=False)
                    logging.info(f"Initialized master dataset from sample: {len(df)} rows")
                    return df
                else:
                    logging.error("No training data available")
                    return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading training data: {str(e)}")
            raise
    
    def get_data_stats(self):
        """
        Get statistics about current data
        
        Returns:
            dict: Data statistics
        """
        stats = {
            'new_files': len(self.get_new_files()),
            'archived_files': len([f for f in os.listdir(self.archive_dir) if f.endswith('.csv')]),
            'training_rows': 0,
            'last_update': None
        }
        
        if os.path.exists(self.master_dataset_path):
            df = pd.read_csv(self.master_dataset_path)
            stats['training_rows'] = len(df)
            stats['last_update'] = datetime.fromtimestamp(
                os.path.getmtime(self.master_dataset_path)
            ).strftime('%Y-%m-%d %H:%M:%S')
        
        return stats
