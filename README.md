# Early Risk Signal System

A data-driven framework to identify early behavioral signals of credit card delinquency.

## Project Structure

- **`src/`**: Source code for the application.
    - `app.py`: Main Streamlit dashboard (Frontend & Backend).
    - `risk_engine.py`: Core logic for risk scoring.
    - `data_loader.py`: Data ingestion and validation.
- **`data/`**: Input datasets.
- **`notebooks/`**: Jupyter notebooks for analysis and model development.

## Architecture

This project uses **Streamlit**, a modern framework that combines Frontend (UI) and Backend (Python Logic) into a single, lightweight application. 

- **Why Streamlit?** For data science and risk analytics, Streamlit allows for rapid prototyping and interactive visualizations without the overhead of managing separate React/Angular frontends and Flask/Django backends.
- **Scalability**: If the application needs to scale to thousands of concurrent users, we can split the backend into a FastAPI service and keep Streamlit (or move to React) for the frontend.

## Environment Setup

It is recommended to use a virtual environment to manage dependencies.

1. **Run the Setup Script** (Windows):
   Double-click `setup_env.bat` or run:
   ```cmd
   setup_env.bat
   ```

2. **Manual Setup**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

## How to Run

1. **Activate Environment** (if not already active):
   ```bash
   .\venv\Scripts\activate
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run src/app.py
   ```

3. **Explore the Model**:
   Open `notebooks/risk_model_development.ipynb` in Jupyter.
