import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from src.repositories.model_repository import ModelRepository

class TrainingService:
    def __init__(self, model_repo=None):
        self.model_repo = model_repo or ModelRepository()

    @staticmethod
    def train_sgd_constrained(X_scaled, y_scaled, learning_rate=0.01, max_epochs=1000, tolerance=0.001, callback=None):
        """
        Trains a model using SGD with constraints (theta >= 0, bias >= 0).
        Yields progress if callback is provided, or returns final weights.
        """
        m, n = X_scaled.shape
        theta = np.zeros(n)
        bias = 0.1
        rmse_prev = float('inf')
        losses = []

        for epoch in range(1, max_epochs + 1):
            # Shuffle data for SGD
            shuffled_indices = np.random.permutation(m)
            X_shuff = X_scaled[shuffled_indices]
            y_shuff = y_scaled[shuffled_indices]

            for i in range(m):
                y_pred = np.dot(X_shuff[i], theta) + bias
                error = y_pred - y_shuff[i]

                grad_theta = error * X_shuff[i]
                grad_bias = error

                theta = theta - learning_rate * grad_theta
                bias = bias - learning_rate * grad_bias

                # Apply constraints
                theta = np.maximum(theta, 0)
                bias = max(bias, 0)

            # Evaluate epoch
            y_pred_all = np.dot(X_scaled, theta) + bias
            loss = np.mean((y_pred_all - y_scaled) ** 2)
            losses.append(loss)

            rmse = np.sqrt(mean_squared_error(y_scaled, y_pred_all))
            delta = abs(rmse_prev - rmse)

            if callback:
                callback(epoch, theta, bias, loss, rmse, delta)

            if delta < tolerance:
                break
            
            rmse_prev = rmse

        return theta, bias, losses

    def train_model(self, df: pd.DataFrame, model_type: str, data_type: str, 
                    learning_rate=0.01, max_epochs=1000, tolerance=0.001, 
                    callback=None) -> dict:
        """
        Full pipeline: pre-process, transform, scale, train SGD, calculate metrics, save.
        """
        # 1. Preprocessing based on data_type
        # Data Service should already filter or aggregate, but let's make sure we do it here:
        from src.services.data_service import DataService
        df_processed = DataService.get_preprocessed_dataset(df, data_type)
        
        X_raw = df_processed[['Durasi_Jam', 'Penonton Aktif']].values
        y_raw = df_processed['Produk Terjual'].values

        # 2. Model transformation
        if model_type == "Logarithmic":
            # X_trans = log(X + 1)
            X_trans = np.log(X_raw + 1)
            poly_transformer = None
        elif model_type == "Polynomial":
            # Polynomial features of degree 2
            poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
            X_trans = poly_transformer.fit_transform(X_raw)
        else: # Linear
            X_trans = X_raw
            poly_transformer = None

        # 3. Standardization
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X_trans)
        y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

        # 4. Training SGD
        epoch_history = []
        def epoch_callback(epoch, current_theta, current_bias, loss, rmse, delta):
            epoch_history.append({
                "epoch": epoch,
                "loss": loss,
                "rmse": rmse,
                "delta": delta
            })
            if callback:
                callback(epoch, current_theta, current_bias, loss, rmse, delta)

        theta, bias, losses = self.train_sgd_constrained(
            X_scaled, y_scaled, 
            learning_rate=learning_rate, 
            max_epochs=max_epochs, 
            tolerance=tolerance, 
            callback=epoch_callback
        )

        # 5. Reverse scaling for unscaled equation print
        # theta_thesis (thesis formula style: theta / scale_X)
        theta_thesis = theta / scaler_X.scale_
        bias_thesis = bias - np.dot(scaler_X.mean_ / scaler_X.scale_, theta)
        # Apply inverse transform on the bias intercept
        bias_thesis = scaler_y.inverse_transform([[bias_thesis]])[0][0]

        # theta_correct (mathematically correct formula style: theta_scaled * std_y / std_x)
        scale_y = scaler_y.scale_[0]
        mean_y = scaler_y.mean_[0]
        theta_correct = (theta / scaler_X.scale_) * scale_y
        bias_correct = (bias - np.dot(scaler_X.mean_ / scaler_X.scale_, theta)) * scale_y + mean_y

        # 6. Evaluation metrics on original data (using correct scaling prediction)
        y_pred_scaled = np.dot(X_scaled, theta) + bias
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        mae = mean_absolute_error(y_raw, y_pred)
        rmse_val = np.sqrt(mean_squared_error(y_raw, y_pred))
        
        # R2
        ss_res = np.sum((y_raw - y_pred) ** 2)
        ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # MAPE (avoiding division by zero)
        mask = y_raw != 0
        if np.sum(mask) > 0:
            mape = mean_absolute_percentage_error(y_raw[mask], y_pred[mask]) * 100
        else:
            mape = 0.0

        # Create model dictionary
        model_data = {
            "name": f"{model_type} - {data_type}",
            "model_type": model_type,
            "data_type": data_type,
            "theta": theta,
            "bias": bias,
            "theta_thesis": theta_thesis,
            "bias_thesis": bias_thesis,
            "theta_correct": theta_correct,
            "bias_correct": bias_correct,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "poly_transformer": poly_transformer,
            "metrics": {
                "r2": r2,
                "mae": mae,
                "rmse": rmse_val,
                "mape": mape
            },
            "hyperparameters": {
                "learning_rate": learning_rate,
                "max_epochs": max_epochs,
                "tolerance": tolerance
            },
            "history": epoch_history
        }

        # Save to model registry
        self.model_repo.save_model(model_data["name"], model_data)
        
        return model_data
