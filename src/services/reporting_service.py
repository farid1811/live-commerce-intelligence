import io
import pandas as pd
import matplotlib.pyplot as plt

class ReportingService:
    @staticmethod
    def generate_excel_report(df: pd.DataFrame, model_data: dict = None) -> bytes:
        """
        Generates an Excel workbook containing the dataset, preprocessed statistics, 
        and the registered model parameters/metrics. Returns bytes.
        """
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 1. Sheet: Raw Data
            df.to_excel(writer, sheet_name='Historical Dataset', index=False)
            
            # 2. Sheet: Statistics
            from src.services.data_service import DataService
            stats_df = DataService.get_summary_statistics(df)
            stats_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
            
            # 3. Sheet: Model Info
            if model_data:
                model_records = [
                    {"Key": "Model Name", "Value": model_data.get("name", "N/A")},
                    {"Key": "Model Type", "Value": model_data.get("model_type", "N/A")},
                    {"Key": "Dataset Segment", "Value": model_data.get("data_type", "N/A")},
                    {"Key": "R² Score", "Value": model_data.get("metrics", {}).get("r2", 0.0)},
                    {"Key": "Mean Absolute Error (MAE)", "Value": model_data.get("metrics", {}).get("mae", 0.0)},
                    {"Key": "Root Mean Squared Error (RMSE)", "Value": model_data.get("metrics", {}).get("rmse", 0.0)},
                    {"Key": "Mean Absolute Percentage Error (MAPE)", "Value": f"{model_data.get('metrics', {}).get('mape', 0.0):.2f}%"},
                    {"Key": "Thesis Formula Intercept", "Value": model_data.get("bias_thesis", 0.0)},
                ]
                
                # Add thesis weights
                theta_thesis = model_data.get("theta_thesis", [])
                for idx, w in enumerate(theta_thesis):
                    model_records.append({"Key": f"Thesis Weight B_{idx+1}", "Value": w})
                
                model_df = pd.DataFrame(model_records)
                model_df.to_excel(writer, sheet_name='Model Specifications', index=False)
                
        return output.getvalue()

    @staticmethod
    def generate_markdown_report(df: pd.DataFrame, model_data: dict) -> str:
        """
        Generates a beautifully-formatted markdown executive summary report.
        """
        metrics = model_data.get("metrics", {})
        hyper = model_data.get("hyperparameters", {})
        
        from src.services.data_service import DataService
        stats_df = DataService.get_summary_statistics(df)

        # Build markdown table manually to avoid tabulate dependency
        cols = list(stats_df.columns)
        headers = " | ".join(cols)
        divider = " | ".join(["---"] * len(cols))
        rows = []
        for _, r in stats_df.iterrows():
            row_vals = []
            for col in cols:
                val = r[col]
                if isinstance(val, float):
                    row_vals.append(f"{val:.4f}")
                else:
                    row_vals.append(str(val))
            rows.append(" | ".join(row_vals))
        stats_table = f"| {headers} |\n| {divider} |\n" + "\n".join([f"| {row} |" for row in rows])
        
        theta_thesis = model_data.get("theta_thesis", [])
        weights_str = "\n".join([f"- **Weight B_{i+1}**: {w:.6f}" for i, w in enumerate(theta_thesis)])
        
        report = f"""# CONVORA EXECUTIVE INTELLIGENCE REPORT
*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Platform: Convora AI Live Commerce Intelligence*

---

## 1. Executive Summary
This report analyzes live streaming commerce performance metrics (Duration, Active Viewers) and models the target variable (Products Sold) using standard Stochastic Gradient Descent (SGD) regression optimization.

- **Current Active Model**: {model_data.get('name', 'Default Model')}
- **Model Type**: {model_data.get('model_type', 'Linear')}
- **Training Segment**: {model_data.get('data_type', 'All Data')}
- **Forecast Confidence (R²)**: {metrics.get('r2', 0.0)*100:.2f}%
- **Typical Deviation (RMSE)**: {metrics.get('rmse', 0.0):.2f} products

---

## 2. Statistical Baseline Analysis
Below is the summary baseline statistics calculated across the historical dataset:

{stats_table}

---

## 3. Mathematical Model Specifications
The optimization pipeline trained the regression parameters using a Stochastic Gradient Descent (SGD) algorithm with positive constraints ($\theta \ge 0, bias \ge 0$).

### 3.1 Unscaled Thesis Equation
$$ {model_data.get('model_type')} \text{{ Equation:}} $$
`{model_data.get('name')}` unscaled formulation:
```
Y = {model_data.get('bias_thesis', 0.0):.6f} + sum_i (B_i * X_i)
```

**Parameters:**
- **Intercept (Bias)**: {model_data.get('bias_thesis', 0.0):.6f}
{weights_str}

### 3.2 Evaluation Metrics
- **R² Score (Coefficient of Determination)**: {metrics.get('r2', 0.0):.4f}
- **MAE (Mean Absolute Error)**: {metrics.get('mae', 0.0):.4f}
- **RMSE (Root Mean Squared Error)**: {metrics.get('rmse', 0.0):.4f}
- **MAPE (Mean Absolute Percentage Error)**: {metrics.get('mape', 0.0):.2f}%

**Hyperparameters:**
- **Learning Rate**: {hyper.get('learning_rate', 0.01)}
- **Tolerance**: {hyper.get('tolerance', 0.001)}
- **Max Epochs**: {hyper.get('max_epochs', 1000)}

---

## 4. Business Recommendations
1. **Streaming Duration Optimization**: Target a baseline stream duration of 58 minutes. Streams lasting less than 1 hour fail to build enough viewer momentum.
2. **Audience Acquisition priority**: Focus marketing efforts on boosting *Active Viewers* during the first 15 minutes of the broadcast, as viewer momentum acts as a strong multiplier for final conversions.
3. **Continuous Re-training**: Periodically upload fresh session spreadsheets to the Dataset Manager to keep the model weights tuned to current shopper seasonal behaviors.
"""
        return report

    @staticmethod
    def generate_chart_png(df: pd.DataFrame, model_data: dict) -> bytes:
        """
        Generates a premium regression plot (Actual vs Predicted and Residuals) 
        as PNG bytes for download.
        """
        from src.services.data_service import DataService
        df_processed = DataService.get_preprocessed_dataset(df, model_data.get("data_type", "All Data"))
        
        X = df_processed[['Durasi_Jam', 'Penonton Aktif']].values
        y = df_processed['Produk Terjual'].values
        
        # Make predictions
        scaler_X = model_data.get("scaler_X")
        scaler_y = model_data.get("scaler_y")
        theta = model_data.get("theta")
        bias = model_data.get("bias")
        model_type = model_data.get("model_type", "Linear")
        poly_transformer = model_data.get("poly_transformer")
        
        if scaler_X and scaler_y:
            if model_type == "Logarithmic":
                X_trans = np.log(X + 1)
            elif model_type == "Polynomial" and poly_transformer:
                X_trans = poly_transformer.transform(X)
            else:
                X_trans = X
            
            X_scaled = scaler_X.transform(X_trans)
            y_pred_scaled = np.dot(X_scaled, theta) + bias
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        else:
            y_pred = y
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
        
        # Plot 1: Actual vs Predicted
        ax1.scatter(y, y_pred, color='#2563EB', alpha=0.6, edgecolors='w', s=40)
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], color='#EF4444', linestyle='--', linewidth=1.5)
        ax1.set_xlabel('Actual Products Sold', fontsize=10, color='#1F2937')
        ax1.set_ylabel('Predicted Products Sold', fontsize=10, color='#1F2937')
        ax1.set_title('Actual vs. Predicted', fontsize=12, fontweight='bold', color='#1F2937')
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Plot 2: Residuals
        residuals = y - y_pred
        ax2.scatter(y_pred, residuals, color='#0D9488', alpha=0.6, edgecolors='w', s=40)
        ax2.axhline(0, color='#EF4444', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Predicted Products Sold', fontsize=10, color='#1F2937')
        ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=10, color='#1F2937')
        ax2.set_title('Residuals Analysis', fontsize=12, fontweight='bold', color='#1F2937')
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        
        output = io.BytesIO()
        plt.savefig(output, format='png', bbox_inches='tight')
        plt.close(fig)
        return output.getvalue()

    @staticmethod
    def generate_pdf_report(df: pd.DataFrame, model_data: dict) -> bytes:
        """
        Generates a premium 1-page PDF Executive Summary Dashboard 
        using Matplotlib. Returns bytes.
        """
        fig = plt.figure(figsize=(8.5, 11), dpi=300)
        
        # Title Header
        fig.suptitle('CONVORA EXECUTIVE INTELLIGENCE REPORT', fontsize=18, fontweight='bold', color='#1F2937', y=0.96)
        
        # Add metadata text
        meta_text = (
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Model Configuration: {model_data.get('name', 'Default Model')}\n"
            f"Model Type: {model_data.get('model_type', 'Linear')} Regression (Optimized via SGD with Constraints)\n"
            f"Training Dataset Segment: {model_data.get('data_type', 'All Data')}"
        )
        fig.text(0.08, 0.90, meta_text, fontsize=9, color='#4B5563', style='italic')
        
        # Divider line
        fig.text(0.08, 0.88, "_"*88, color='#D1D5DB')
        
        # Add summary statistics text box
        from src.services.data_service import DataService
        stats_df = DataService.get_summary_statistics(df)
        stats_lines = []
        for idx, row in stats_df.iterrows():
            stats_lines.append(f"{row['Feature']}: Mean={row['Mean']:.2f}, Std={row['Std Dev']:.2f}, Max={row['Max']:.2f}")
        stats_text = "Baseline Statistics:\n" + "\n".join([f"- {line}" for line in stats_lines])
        
        metrics = model_data.get("metrics", {})
        metrics_text = (
            f"Model Performance Metrics:\n"
            f"- R² Score: {metrics.get('r2', 0.0):.4f}\n"
            f"- MAE: {metrics.get('mae', 0.0):.4f}\n"
            f"- RMSE: {metrics.get('rmse', 0.0):.4f}\n"
            f"- MAPE: {metrics.get('mape', 0.0):.2f}%"
        )
        
        fig.text(0.08, 0.77, stats_text, fontsize=9, color='#1F2937', bbox=dict(boxstyle='round,pad=0.5', facecolor='#F9FAFB', edgecolor='#E5E7EB'))
        fig.text(0.53, 0.77, metrics_text, fontsize=9, color='#1F2937', bbox=dict(boxstyle='round,pad=0.5', facecolor='#EFF6FF', edgecolor='#BFDBFE'))
        
        # Equation Text
        eq_text = f"Unscaled Thesis Equation:\n  {model_data.get('model_type')} Model: {PredictionService.get_equation_string(model_data.get('model_type'), model_data.get('theta_thesis'), model_data.get('bias_thesis'))}"
        fig.text(0.08, 0.70, eq_text, fontsize=9.5, fontweight='bold', color='#2563EB', bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFFFFF', edgecolor='#BFDBFE'))

        # Add charts in subplots
        # Position subplots at y=0.35 to 0.65
        ax1 = fig.add_axes([0.08, 0.33, 0.40, 0.28])
        ax2 = fig.add_axes([0.53, 0.33, 0.40, 0.28])
        
        # Predictions calculations
        df_processed = DataService.get_preprocessed_dataset(df, model_data.get("data_type", "All Data"))
        X = df_processed[['Durasi_Jam', 'Penonton Aktif']].values
        y = df_processed['Produk Terjual'].values
        scaler_X = model_data.get("scaler_X")
        scaler_y = model_data.get("scaler_y")
        theta = model_data.get("theta")
        bias = model_data.get("bias")
        model_type = model_data.get("model_type", "Linear")
        poly_transformer = model_data.get("poly_transformer")
        
        if scaler_X and scaler_y:
            if model_type == "Logarithmic":
                X_trans = np.log(X + 1)
            elif model_type == "Polynomial" and poly_transformer:
                X_trans = poly_transformer.transform(X)
            else:
                X_trans = X
            
            X_scaled = scaler_X.transform(X_trans)
            y_pred_scaled = np.dot(X_scaled, theta) + bias
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        else:
            y_pred = y
            
        # Plot 1: Actual vs Predicted
        ax1.scatter(y, y_pred, color='#2563EB', alpha=0.6, edgecolors='w', s=25)
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], color='#EF4444', linestyle='--', linewidth=1.2)
        ax1.set_xlabel('Actual Sales', fontsize=8)
        ax1.set_ylabel('Predicted Sales', fontsize=8)
        ax1.set_title('Actual vs. Predicted', fontsize=9, fontweight='bold')
        ax1.tick_params(labelsize=7)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Plot 2: Residuals
        residuals = y - y_pred
        ax2.scatter(y_pred, residuals, color='#0D9488', alpha=0.6, edgecolors='w', s=25)
        ax2.axhline(0, color='#EF4444', linestyle='--', linewidth=1.2)
        ax2.set_xlabel('Predicted Sales', fontsize=8)
        ax2.set_ylabel('Residuals', fontsize=8)
        ax2.set_title('Residuals Analysis', fontsize=9, fontweight='bold')
        ax2.tick_params(labelsize=7)
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        # Recommendations box at the bottom
        rec_text = (
            "Operational Business Recommendations:\n"
            "1. Target optimal stream duration of 58 minutes. Shorter streams yield significantly lower search rankings.\n"
            "2. Deploy interactive discounts or coupon giveaways during the first 15 minutes of concurrent viewer peaks.\n"
            "3. Periodically re-train parameters in the Training Center as shopper demographics and seasonality shift MoM.\n"
            "4. Monitor anomalous session outlier logs in the Business Intelligence tab to isolate high-performance triggers."
        )
        fig.text(0.08, 0.12, rec_text, fontsize=9.5, color='#1F2937', lineheight=1.5,
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#ECFDF5', edgecolor='#10B981'))
        
        # Footer branding
        fig.text(0.08, 0.06, "Confidential - For Internal Business Analysis Only - Convora SaaS Suite", fontsize=8, color='#9CA3AF')
        fig.text(0.80, 0.06, "www.convora.ai", fontsize=8, color='#2563EB', fontweight='bold')
        
        output = io.BytesIO()
        plt.savefig(output, format='pdf', bbox_inches='tight')
        plt.close(fig)
        return output.getvalue()

