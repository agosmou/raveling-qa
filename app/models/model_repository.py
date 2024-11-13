# models/model_repository.py
import os
import pickle
import streamlit as st

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
        self.model, self.scaler = self.load_model_and_scaler()

    @st.cache_resource
    def load_model_and_scaler(_self):
        # Check if the model file exists
        if not os.path.exists(_self.model_path):
            raise FileNotFoundError(f"Model file not found at {_self.model_path}")
        if not os.path.exists(_self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {_self.scaler_path}")
        
        with open(_self.model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(_self.scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
