import house_price.models.schemas as schemas
import house_price.models.predict as predict
from house_price.config import Settings
from house_price.config import settings as default_settings

class ModelController:
    def __init__(self, settings: Settings = None):
        self.settings = settings or default_settings
        self.predict_model = predict.PredictModel(self.settings)

    def ensure_model_loaded(self):
        """
        Ensure that the model, scaler, and encoder are loaded.
        """
        if not self.predict_model.is_loaded():
            self.predict_model.load_artifacts()

    def predict_price(self, input_data: schemas.InputDataSchema) -> schemas.PredictionResult:
        """
        Predict the house price based on the input data.
        """
        self.ensure_model_loaded()
        if not self.predict_model.is_loaded():
            raise ValueError("Model artifacts are not loaded. Please load them before making predictions.")
        
        # Convert Pydantic model to dictionary
        input_dict = input_data.model_dump()
        
        # Make prediction
        predicted_price = self.predict_model.predict(input_dict)
        
        # Return the result as a Pydantic model
        return schemas.PredictionResult(predicted_price=predicted_price)