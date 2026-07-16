import os
import pandas as pd
import numpy as np
from src.utils.config import DEFAULT_DATASET

class DatasetRepository:
    def __init__(self, filepath=None):
        self.filepath = filepath or DEFAULT_DATASET

    def load_data(self) -> pd.DataFrame:
        """Loads data from the excel or csv file."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset not found at {self.filepath}")
        
        if self.filepath.endswith('.xlsx') or self.filepath.endswith('.xls'):
            # The skripsi excel has a header at row index 1 (second row)
            # Try to read with header=1, if columns don't match, try header=0
            try:
                df = pd.read_excel(self.filepath, sheet_name="Sheet1", header=1)
                # Verify required columns
                required = ['Durasi_Jam', 'Penonton Aktif', 'Produk Terjual']
                if not all(col in df.columns for col in required):
                    # Try header=0
                    df = pd.read_excel(self.filepath, sheet_name="Sheet1", header=0)
            except Exception:
                df = pd.read_excel(self.filepath, header=0)
        else:
            df = pd.read_csv(self.filepath)
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Ensure Bulan column exists
        if 'Bulan' not in df.columns:
            df['Bulan'] = 'september' # default fallback
            
        return df

    def save_data(self, df: pd.DataFrame, target_path: str) -> None:
        """Saves a dataframe to target_path."""
        if target_path.endswith('.xlsx') or target_path.endswith('.xls'):
            df.to_excel(target_path, index=False)
        else:
            df.to_csv(target_path, index=False)

    def validate_schema(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Validates if the dataframe has the correct columns and types."""
        required = ['Durasi_Jam', 'Penonton Aktif', 'Produk Terjual']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return False, f"Missing required columns: {', '.join(missing)}"
        
        # Check data types
        for col in required:
            if not np.issubdtype(df[col].dtype, np.number):
                return False, f"Column '{col}' must contain numeric data."
            
        return True, "Dataset schema is valid."
