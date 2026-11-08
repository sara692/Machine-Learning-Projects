from house_price.config import settings as default_settings
from house_price.config import Settings
import joblib
import pandas as pd

class PredictModel:
    def __init__(self, settings: Settings = None):
        self.settings = settings or default_settings
        self.model = None
        self.scaler = None
        self.posted_by_encoder = None
        self.kmeans = None

    def load_artifacts(self):
        """
        Load the trained model, scaler, and encoder from the specified paths.
        """
        try:
            self.model = joblib.load(self.settings.model_path)
            self.scaler = joblib.load(self.settings.scaler_path)
            self.posted_by_encoder = joblib.load(self.settings.posted_by_encoder_path)
            self.city_encoder = joblib.load(self.settings.city_encoder_path)
            self.kmeans = joblib.load(self.settings.kmeans_path)  # Load the KMeans model
        except FileNotFoundError as e:
            print(f"Artifact not found: {e}")

    def is_loaded(self) -> bool:
        """
        Check if the model, scaler, and encoder are loaded.
        """
        return self.model is not None and self.scaler is not None and self.posted_by_encoder is not None and self.city_encoder is not None and self.kmeans is not None
    
    def predict(self, input_data: dict):
        """
        Make predictions using the loaded model.
        """
        if not self.is_loaded():
            raise ValueError("Model, scaler, and encoder must be loaded before making predictions.")
        df = pd.DataFrame([input_data])

    # --------------------------------------------------
    # 2. Encode categorical features
    # --------------------------------------------------
        try:
            df["POSTED_BY"] = self.posted_by_encoder.transform(
                df["POSTED_BY"]
            )

        except ValueError as e:
            raise ValueError(
                f"Unknown categorical value in input: {e}"
            )
        labels = self.kmeans.predict(df[["LATITUDE", "LONGITUDE"]])
        for i in range(self.kmeans.n_clusters):
            df[f"loc_tier_tier{i + 1}"] = int(labels[0] == i)

        expected = list(self.scaler.feature_names_in_)
        if "BHK_NO." in expected and "BHK_NO" in df.columns:
            df = df.rename(columns={"BHK_NO": "BHK_NO."})
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features for the model: {missing}")
        
        # Preprocess input data
        input_data_scaled = self.scaler.transform(df[expected])

        # Make predictions
        predictions = self.model.predict(input_data_scaled)
        return predictions[0]

