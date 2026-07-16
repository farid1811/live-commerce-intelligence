import os
import joblib
import datetime
from src.utils.config import MODEL_DIR

class ModelRepository:
    def __init__(self, registry_dir=None):
        self.registry_dir = registry_dir or MODEL_DIR
        os.makedirs(self.registry_dir, exist_ok=True)

    def save_model(self, model_name: str, model_data: dict) -> str:
        """Saves a model dict (containing weights, bias, scalers, and metrics) as a joblib file."""
        # Sanitize filename
        filename = f"{model_name.replace(' ', '_').lower()}.pkl"
        filepath = os.path.join(self.registry_dir, filename)
        
        # Add timestamp and identifier
        model_data["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_data["filename"] = filename
        
        joblib.dump(model_data, filepath)
        return filepath

    def load_model(self, filename: str) -> dict:
        """Loads a model from registry directory."""
        filepath = os.path.join(self.registry_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        return joblib.load(filepath)

    def list_models(self) -> list[dict]:
        """Lists all registered models in the registry with their metadata."""
        models = []
        for file in os.listdir(self.registry_dir):
            if file.endswith(".pkl"):
                try:
                    data = joblib.load(os.path.join(self.registry_dir, file))
                    model_meta = {
                        "filename": file,
                        "name": data.get("name", file),
                        "model_type": data.get("model_type", "Linear"),
                        "data_type": data.get("data_type", "All Data"),
                        "metrics": data.get("metrics", {}),
                        "hyperparameters": data.get("hyperparameters", {}),
                        "timestamp": data.get("timestamp", "N/A"),
                        "is_active": data.get("is_active", False)
                    }
                    models.append(model_meta)
                except Exception:
                    pass
        # Sort by timestamp descending
        models.sort(key=lambda x: x["timestamp"], reverse=True)
        return models

    def set_active_model(self, filename: str) -> None:
        """Sets a specific model as the active one, unsetting others."""
        for file in os.listdir(self.registry_dir):
            if file.endswith(".pkl"):
                try:
                    filepath = os.path.join(self.registry_dir, file)
                    data = joblib.load(filepath)
                    if file == filename:
                        data["is_active"] = True
                    else:
                        data["is_active"] = False
                    joblib.dump(data, filepath)
                except Exception:
                    pass

    def get_active_model(self) -> dict | None:
        """Gets the active model from the registry."""
        models = self.list_models()
        # Find active model
        for m in models:
            if m["is_active"]:
                return self.load_model(m["filename"])
        
        # If no active model, return the newest one or None
        if models:
            # Set the first one as active
            self.set_active_model(models[0]["filename"])
            return self.load_model(models[0]["filename"])
        
        return None
