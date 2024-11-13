# services/prediction_service.py
import pandas as pd

class PredictionService:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, features: list) -> list:
        X_new = pd.DataFrame(features)
        X_new_scaled = self.scaler.transform(X_new)
        predictions = self.model.predict(X_new_scaled)
        return predictions.tolist()
