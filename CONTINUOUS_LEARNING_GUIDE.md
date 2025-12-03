# Continuous Learning System - User Guide

## 🚀 Quick Start

### 1. Initialize the System (One-Time Setup)
```bash
python src/initialize_cl_system.py
```

This will:
- Copy your current model to the active directory
- Create initial metadata and versioning
- Set up the master training dataset
- Initialize deployment logs

### 2. Use the Dashboard
```bash
streamlit run src/app.py
```

- Upload new customer data via the sidebar
- Data is automatically saved for future model training
- Dashboard always uses the latest deployed model

### 3. Manual Retraining (Optional)
```bash
python src/retrain_manual.py
```

Manually trigger the retraining pipeline when you have new data.

### 4. Start Automated Scheduler (Recommended)
```bash
python src/ml_pipeline/scheduler.py
```

Runs daily at 2:00 AM to check for new data and retrain if needed.

---

## 📋 System Workflow

```
User Uploads CSV → Saved to data/new/
                ↓
        (Daily at 2 AM)
                ↓
    Merge with Master Dataset
                ↓
        Train New Model
                ↓
    Validate (Accuracy, Drift)
                ↓
        Deploy if Approved
                ↓
    Dashboard Auto-Updates
```

---

## 🛠️ Commands Reference

### View Deployment History
```bash
python src/rollback.py --list
```

### Rollback to Previous Version
```bash
python src/rollback.py
```

### Rollback to Specific Version
```bash
python src/rollback.py --version 1.2
```

### Check System Status
```python
from ml_pipeline.data_manager import DataManager
dm = DataManager()
print(dm.get_data_stats())
```

---

## 📁 Directory Structure

```
data/
├── new/              # Uploaded files (staging)
├── training/         # Master dataset
└── archive/          # Processed files

models/
├── active/           # Current production model
│   ├── model.pkl
│   └── metadata.json
├── versions/         # Historical models
│   ├── model_v1_0.pkl
│   ├── model_v1_1.pkl
│   └── ...
└── metadata/         # Model performance logs
    ├── model_v1_0.json
    └── ...

logs/
├── retraining.log    # Training logs
└── deployments.json  # Deployment history
```

---

## ⚙️ Configuration

### Change Retraining Schedule
Edit `src/ml_pipeline/scheduler.py`:
```python
# Daily at 3 AM
scheduler.schedule_daily("03:00")

# Weekly on Monday at 2 AM
scheduler.schedule_weekly("monday", "02:00")
```

### Adjust Validation Thresholds
Edit `src/ml_pipeline/model_validator.py`:
```python
self.min_accuracy = 0.70        # Minimum accuracy required
self.max_accuracy_drop = 0.05   # Max 5% drop allowed
self.max_drift_threshold = 0.15 # Max 15% drift
```

---

## 🔍 Monitoring

### Check Logs
```bash
# View retraining logs
cat logs/retraining.log

# View deployment history
cat logs/deployments.json
```

### Model Metrics
All model versions include:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC
- Feature Importance
- Confusion Matrix

---

## 🚨 Troubleshooting

### Model Not Updating
1. Check if new files exist in `data/new/`
2. Review `logs/retraining.log` for errors
3. Verify scheduler is running

### Validation Failing
- Check validation thresholds in `model_validator.py`
- Review metrics comparison in logs
- Consider if new data quality is poor

### Rollback Needed
```bash
python src/rollback.py --list  # See history
python src/rollback.py         # Rollback to previous
```

---

## 📊 Dashboard Integration

The dashboard automatically:
- ✅ Uses the latest deployed model
- ✅ Saves uploaded data for training
- ✅ Displays current model version
- ✅ Shows deployment date

No manual intervention needed!

---

## 🔐 Safety Features

1. **Validation Before Deployment**
   - Accuracy must meet minimum threshold
   - Cannot drop more than 5% from current model
   - Drift detection prevents bad deployments

2. **Version Control**
   - All models are versioned and saved
   - Full metadata for each version
   - Easy rollback to any previous version

3. **Logging**
   - All operations logged
   - Deployment history tracked
   - Audit trail maintained

---

## 💡 Best Practices

1. **Monitor Regularly**: Check logs weekly
2. **Validate Data**: Ensure uploaded data quality
3. **Test Rollback**: Practice rollback procedure
4. **Keep Versions**: Don't delete old model versions
5. **Review Metrics**: Compare model performance over time

---

## 📞 Support

For issues or questions:
1. Check `logs/retraining.log`
2. Review this guide
3. Contact ML team
