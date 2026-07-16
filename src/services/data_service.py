import pandas as pd
import numpy as np

class DataService:
    @staticmethod
    def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Removes outliers using the IQR method."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    @classmethod
    def clean_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Data cleaning: remove sales = 0 and outliers in features."""
        df_clean = df[df['Produk Terjual'] > 0].copy()
        for col in ['Durasi_Jam', 'Penonton Aktif', 'Produk Terjual']:
            df_clean = cls.remove_outliers_iqr(df_clean, col)
        return df_clean

    @staticmethod
    def get_weekly_data(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates data weekly."""
        bulan_order = ['januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus', 'september', 'oktober', 'november', 'desember']
        
        df_copy = df.copy()
        df_copy['Bulan'] = pd.Categorical(df_copy['Bulan'], categories=bulan_order, ordered=True)
        df_copy['Bulan_Str'] = df_copy['Bulan'].astype(str)
        
        df_sorted = df_copy.sort_values('Bulan')
        df_sorted['Week_Number'] = (df_sorted.groupby('Bulan_Str').cumcount() // 7) + 1
        df_sorted['Month_Week'] = df_sorted['Bulan_Str'] + '_Week' + df_sorted['Week_Number'].astype(str)
        
        df_weekly = df_sorted.groupby('Month_Week', observed=False).agg({
            'Durasi_Jam': 'mean',
            'Penonton Aktif': 'mean', 
            'Produk Terjual': 'mean',
            'Bulan_Str': 'first'
        }).reset_index()
        
        # Keep Bulan column
        df_weekly['Bulan'] = pd.Categorical(df_weekly['Bulan_Str'], categories=bulan_order, ordered=True)
        df_weekly = df_weekly.sort_values('Bulan')
        return df_weekly

    @staticmethod
    def get_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates data monthly."""
        bulan_order = ['januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus', 'september', 'oktober', 'november', 'desember']
        
        df_copy = df.copy()
        df_copy['Bulan'] = pd.Categorical(df_copy['Bulan'], categories=bulan_order, ordered=True)
        
        df_monthly = df_copy.groupby('Bulan', observed=False).agg({
            'Durasi_Jam': 'mean',
            'Penonton Aktif': 'mean', 
            'Produk Terjual': 'mean'
        }).reset_index()
        return df_monthly

    @classmethod
    def get_preprocessed_dataset(cls, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Returns the preprocessed dataset based on type: All Data, Cleaned Data, Weekly Data, Monthly Data."""
        if data_type == "Cleaned Data":
            return cls.clean_data(df)
        elif data_type == "Weekly Data":
            return cls.get_weekly_data(df)
        elif data_type == "Monthly Data":
            return cls.get_monthly_data(df)
        else:
            return df.copy()

    @staticmethod
    def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates rich summary statistics for the dataset."""
        cols = ['Durasi_Jam', 'Penonton Aktif', 'Produk Terjual']
        stats_list = []
        for col in cols:
            if col in df.columns:
                series = df[col]
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_b = q1 - 1.5 * iqr
                upper_b = q3 + 1.5 * iqr
                outliers = series[(series < lower_b) | (series > upper_b)].count()
                missing = series.isna().sum()
                
                stats_list.append({
                    "Feature": col,
                    "Count": len(series),
                    "Mean": series.mean(),
                    "Std Dev": series.std(),
                    "Min": series.min(),
                    "25%": q1,
                    "Median": series.median(),
                    "75%": q3,
                    "Max": series.max(),
                    "Missing Values": missing,
                    "Outliers (IQR)": outliers
                })
        return pd.DataFrame(stats_list)

    @staticmethod
    def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the correlation matrix of numerical features."""
        cols = ['Durasi_Jam', 'Penonton Aktif', 'Produk Terjual']
        existing_cols = [c for c in cols if c in df.columns]
        return df[existing_cols].corr()
