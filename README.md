# HDFC Credit Card Risk Early Warning System

## 🎯 Overview
ML-powered risk monitoring dashboard with automated continuous learning capabilities for early detection of credit card delinquency.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize System (One-Time)
```bash
python src/scripts/initialize_cl_system.py
```

### 3. Run Dashboard
```bash
streamlit run src/app.py
```

Access at: **http://localhost:8501**

---

## 📁 Project Structure

```
HDFC_Credit_Card/
├── data/
│   ├── sample_data.csv       # Sample customer data
│   ├── new/                  # Uploaded files (staging)
│   ├── training/             # Master training dataset
│   └── archive/              # Processed files
├── models/
│   ├── active/               # Current production model
│   ├── versions/             # Historical model versions
│   ├── metadata/             # Model performance logs
│   └── risk_model.pkl        # Initial trained model
├── notebooks/
│   └── risk_model_development.ipynb  # EDA & model development
├── src/
│   ├── app.py                # Streamlit dashboard
│   ├── ml_pipeline/          # Continuous learning modules
│   │   ├── data_manager.py
│   │   ├── model_trainer.py
│   │   ├── model_validator.py
│   │   ├── model_deployer.py
│   │   └── scheduler.py
│   ├── utils/                # Helper utilities
│   │   ├── data_loader.py
│   │   ├── risk_engine.py
│   │   └── convert_data.py
│   ├── scripts/              # Execution scripts
│   │   ├── train_model.py
│   │   ├── retrain_manual.py
│   │   ├── rollback.py
│   │   └── initialize_cl_system.py
│   └── assets/               # Static assets
│       └── style.css
├── assets/
│   └── logo.png              # HDFC Bank logo
├── logs/
│   ├── retraining.log        # Training logs
│   └── deployments.json      # Deployment history
└── .streamlit/
    └── config.toml           # Streamlit theme config
```

---

## 🎓 Features

### Dashboard
- **Portfolio Overview**: Risk distribution, metrics, charts
- **Customer Investigation**: Detailed customer profiles with risk radar
- **High Risk Alerts**: Actionable list of customers needing intervention
- **Drilldown**: Advanced filtering and analysis
- **Smart Search**: Type customer ID to see risk status
- **Real-Time Upload**: Upload CSV files for instant analysis

### Continuous Learning System
- **Automated Retraining**: Daily checks at 2 AM
- **Model Validation**: Performance comparison and drift detection
- **Safe Deployment**: Version control with rollback capability
- **Full Logging**: Audit trail for all operations

---

## 🛠️ Usage

### Upload New Data
1. Open dashboard sidebar
2. Upload CSV file
3. Data automatically saved for future training

### Manual Retraining
```bash
python src/scripts/retrain_manual.py
```

### Start Automated Scheduler
```bash
python src/ml_pipeline/scheduler.py
```

### Rollback Model
```bash
# View history
python src/scripts/rollback.py --list

# Rollback to previous
python src/scripts/rollback.py

# Rollback to specific version
python src/scripts/rollback.py --version 1.2
```

---

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: Utilization, payment ratio, cash withdrawal, spending trends
- **Output**: Risk scores (0-100) and tiers (Intervene/Engage/Monitor)
- **Class Imbalance**: Handled with balanced class weights

---

## 📖 Documentation

- **User Guide**: `CONTINUOUS_LEARNING_GUIDE.md`
- **Walkthrough**: See artifacts directory

---

## 🔐 Requirements

- Python 3.8+
- streamlit
- pandas
- plotly
- scikit-learn
- joblib
- schedule

---

## 📞 Support

For detailed instructions, see `CONTINUOUS_LEARNING_GUIDE.md`

---

**Status**: ✅ Production Ready
