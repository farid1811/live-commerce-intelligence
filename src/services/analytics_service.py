import pandas as pd
import numpy as np

class AnalyticsService:
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame, window=7) -> pd.DataFrame:
        """Calculates moving averages for Durasi_Jam, Penonton Aktif, and Produk Terjual."""
        df_sorted = df.copy()
        # Ensure it has an index or timeline
        df_sorted['Moving_Avg_Sales'] = df_sorted['Produk Terjual'].rolling(window=window, min_periods=1).mean()
        df_sorted['Moving_Avg_Viewers'] = df_sorted['Penonton Aktif'].rolling(window=window, min_periods=1).mean()
        df_sorted['Moving_Avg_Duration'] = df_sorted['Durasi_Jam'].rolling(window=window, min_periods=1).mean()
        return df_sorted

    @staticmethod
    def calculate_growth_rates(df: pd.DataFrame) -> dict:
        """Calculates month-on-month growth rate for sales and viewer count."""
        # Aggregate monthly
        from src.services.data_service import DataService
        df_monthly = DataService.get_monthly_data(df)
        
        # Calculate percentage changes
        df_monthly['Sales_Growth'] = df_monthly['Produk Terjual'].pct_change() * 100
        df_monthly['Viewers_Growth'] = df_monthly['Penonton Aktif'].pct_change() * 100
        
        # Get latest growth rates
        if len(df_monthly) > 1:
            latest_sales = df_monthly['Sales_Growth'].iloc[-1]
            latest_viewers = df_monthly['Viewers_Growth'].iloc[-1]
            latest_month = df_monthly['Bulan'].iloc[-1]
        else:
            latest_sales = 0.0
            latest_viewers = 0.0
            latest_month = "N/A"
            
        return {
            "monthly_data": df_monthly,
            "latest_sales_growth": latest_sales,
            "latest_viewers_growth": latest_viewers,
            "latest_month": latest_month
        }

    @staticmethod
    def detect_outliers_regression_residuals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag outliers where actual products sold differ significantly
        from a basic expected prediction (using residuals threshold).
        """
        df_out = df.copy()
        
        X = df_out[['Durasi_Jam', 'Penonton Aktif']].values
        y = df_out['Produk Terjual'].values
        
        # fit a quick standard linear model to compute residuals
        # using normal equations: theta = (X^T X)^-1 X^T y
        X_design = np.hstack([np.ones((X.shape[0], 1)), X])
        try:
            weights = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
            predictions = X_design @ weights
            residuals = y - predictions
            std_resid = np.std(residuals)
            
            df_out['Expected_Sales'] = np.maximum(0, np.round(predictions, 1))
            df_out['Residual'] = residuals
            # Flag sessions where the residual is > 2 standard deviations away
            df_out['Is_Outlier_Session'] = np.abs(residuals) > (1.96 * std_resid)
            df_out['Outlier_Type'] = np.where(
                df_out['Is_Outlier_Session'],
                np.where(residuals > 0, "Underpredicted (High Performer)", "Overpredicted (Low Performer)"),
                "Normal"
            )
        except Exception:
            df_out['Expected_Sales'] = y
            df_out['Residual'] = 0.0
            df_out['Is_Outlier_Session'] = False
            df_out['Outlier_Type'] = "Normal"
            
        return df_out

    @staticmethod
    def analyze_seasonality(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates and formats seasonality trends by month."""
        from src.services.data_service import DataService
        df_monthly = DataService.get_monthly_data(df)
        
        # Sort in standard calendar order
        bulan_order = ['januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus', 'september', 'oktober', 'november', 'desember']
        df_monthly['Bulan'] = pd.Categorical(df_monthly['Bulan'], categories=bulan_order, ordered=True)
        df_monthly = df_monthly.sort_values('Bulan').reset_index(drop=True)
        
        return df_monthly
