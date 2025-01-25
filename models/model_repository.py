# models/model_repository.py
import os
import pickle
import streamlit as st

@st.cache_resource
def load_model_and_scaler(model_path: str, scaler_path: str):
    """Loads and caches the machine learning model and scaler."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

class ModelRepository:
    def __init__(self, model_path: str = None, scaler_path: str = None):
        # Determine the directory where this script resides
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # If no paths are provided, default to the 'models' directory
        if model_path is None:
            model_path = os.path.join(current_dir, 'rf_model.pkl')
        if scaler_path is None:
            scaler_path = os.path.join(current_dir, 'scaler.pkl')

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model, self.scaler = load_model_and_scaler(self.model_path, self.scaler_path)
