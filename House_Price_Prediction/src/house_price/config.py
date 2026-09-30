from dataclasses import dataclass
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]

@dataclass
class Settings:
    """
    Configuration settings for the house price prediction application.
    """
    model_path: str = ROOT_DIR/"Artifacts/house_price_model.pkl"
    scaler_path: str = ROOT_DIR/"Artifacts/scaler.pkl"
    city_encoder_path: str = ROOT_DIR/"Artifacts/city_encoder.pkl"
    posted_by_encoder_path: str = ROOT_DIR/"Artifacts/posted_by_encoder.pkl"
    kmeans_path: str = ROOT_DIR/"Artifacts/kmeans.pkl"
    data_path: str = ROOT_DIR/"Data/train.csv"
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 3
    mlflow_experiment_name: str = "house_price_prediction"
    registered_model_name: str = "house_price_regressor"
    mlflow_tracking_uri: str = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
    # Service
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

settings = Settings()
    
    