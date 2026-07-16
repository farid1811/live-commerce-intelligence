import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.utils.ui_utils import apply_global_styles, render_page_header, render_alert
from src.services.training_service import TrainingService

apply_global_styles()

render_page_header("Training Center", "Configure, train, and register custom regression models using Stochastic Gradient Descent (SGD) with positive constraints.")

# Load active dataset
df = st.session_state.get("dataset")
if df is None:
    st.warning("No dataset loaded. Please upload a dataset in the Dataset Manager.")
    st.stop()

# Repository & service setup
model_repo = st.session_state.get("model_repo")
training_service = TrainingService(model_repo)

# -----------------------------------------------------
# CONFIGURATION SIDEBAR / PANEL
# -----------------------------------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### Hyperparameters")
    
    model_type = st.selectbox(
        "Model Architecture",
        ["Linear", "Polynomial", "Logarithmic"],
        help="Linear: Y = B1*X1 + B2*X2 + Intercept. Polynomial: includes squared terms and cross-product. Logarithmic: transforms inputs using log(x+1)."
    )
    
    data_type = st.selectbox(
        "Training Dataset Segment",
        ["All Data", "Cleaned Data", "Weekly Data", "Monthly Data"],
        help="All Data: entire dataset. Cleaned Data: outlier removed. Weekly/Monthly: aggregated averages."
    )
    
    lr = st.slider("Learning Rate (alpha)", 0.001, 0.100, 0.010, step=0.001)
    max_epochs = st.slider("Max Epochs", 100, 2000, 1000, step=50)
    tolerance = st.selectbox("Convergence Tolerance", [0.0001, 0.001, 0.01, 0.1], index=1)
    
    start_btn = st.button("🚀 Run Training Pipeline", use_container_width=True)

with col_right:
    st.markdown("### Training Visualization")
    
    # Placeholders for real-time training feedback
    progress_bar_placeholder = st.empty()
    metrics_row_placeholder = st.empty()
    chart_placeholder = st.empty()
    log_header_placeholder = st.empty()
    log_placeholder = st.empty()

# -----------------------------------------------------
# PIPELINE EXECUTION ENGINE
# -----------------------------------------------------
if start_btn:
    # 1. Initialization and validation
    logs = []
    def log_message(msg):
        logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        log_placeholder.code("\n".join(logs), language="plaintext")

    log_header_placeholder.markdown("#### Real-time Training Log")
    log_message("Initializing Convora ML Training Pipeline...")
    
    # Step 1: Schema check
    from src.repositories.dataset_repository import DatasetRepository
    ds_repo = DatasetRepository()
    is_valid, msg = ds_repo.validate_schema(df)
    if not is_valid:
        log_message(f"❌ Schema validation failed: {msg}")
        st.error(msg)
        st.stop()
    log_message("✓ Dataset schema validation succeeded.")

    # Step 2: Data Preprocessing
    log_message(f"Extracting segment: '{data_type}'...")
    from src.services.data_service import DataService
    df_processed = DataService.get_preprocessed_dataset(df, data_type)
    log_message(f"✓ Extracted {len(df_processed)} samples for training.")

    # Step 3: Feature Engineering
    log_message(f"Transforming features for {model_type} Regression...")
    X_raw = df_processed[['Durasi_Jam', 'Penonton Aktif']].values
    y_raw = df_processed['Produk Terjual'].values

    if model_type == "Logarithmic":
        X_trans = np.log(X_raw + 1)
        log_message("✓ Completed log(x+1) transformation.")
    elif model_type == "Polynomial":
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_trans = poly.fit_transform(X_raw)
        log_message("✓ Generated Polynomial degree=2 features (X1, X2, X1^2, X1*X2, X2^2).")
    else:
        X_trans = X_raw
        log_message("✓ Continuous features mapped directly.")

    # Step 4: Standardization
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_trans)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
    log_message("✓ Features standardized via StandardScaler.")

    # Step 5: Initialize Model Weights
    m, n = X_scaled.shape
    theta = np.zeros(n)
    bias = 0.1
    rmse_prev = float('inf')
    losses = []
    epochs_run = 0
    start_time = time.time()
    
    log_message(f"Initialized weights: theta={theta}, bias={bias}")
    log_message("Starting Stochastic Gradient Descent loop with positive constraints...")
    
    # Pre-populate empty plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='Loss (MSE)', line=dict(color='#2563EB', width=2)))
    fig.update_layout(
        title="Training Loss Curve",
        xaxis_title="Epoch",
        yaxis_title="MSE Loss (Scaled)",
        template="simple_white",
        height=320,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    # Real-time SGD loop
    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        
        # Shuffle for Stochastic SGD
        indices = np.random.permutation(m)
        X_shuff = X_scaled[indices]
        y_shuff = y_scaled[indices]
        
        for i in range(m):
            y_pred = np.dot(X_shuff[i], theta) + bias
            error = y_pred - y_shuff[i]
            
            grad_theta = error * X_shuff[i]
            grad_bias = error
            
            theta = theta - lr * grad_theta
            bias = bias - lr * grad_bias
            
            # POSITIVE WEIGHT CONSTRAINTS (Thesis requirement)
            theta = np.maximum(theta, 0)
            bias = max(bias, 0)
            
        # Evaluate Epoch
        y_pred_all = np.dot(X_scaled, theta) + bias
        loss = float(np.mean((y_pred_all - y_scaled) ** 2))
        losses.append(loss)
        
        rmse = float(np.sqrt(np.mean((y_scaled - y_pred_all) ** 2)))
        delta = abs(rmse_prev - rmse)
        
        # Calculate ETA
        elapsed = time.time() - start_time
        avg_time_per_epoch = elapsed / epoch
        eta = avg_time_per_epoch * (max_epochs - epoch)
        
        # Update progress bar
        progress_pct = int((epoch / max_epochs) * 100)
        progress_bar_placeholder.progress(epoch / max_epochs, text=f"Epoch {epoch}/{max_epochs} | Progress: {progress_pct}% | ETA: {eta:.1f}s")
        
        # Update metric cards
        metrics_row_placeholder.markdown(
            f"""
            <div style="display: flex; gap: 16px; margin-bottom: 20px;">
                <div style="flex: 1; background: white; border: 1px solid #E5E7EB; border-radius: 8px; padding: 12px; text-align: center;">
                    <span style="font-size: 0.75rem; color: #6B7280; text-transform: uppercase;">Current Loss</span>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #EF4444;">{loss:.6f}</div>
                </div>
                <div style="flex: 1; background: white; border: 1px solid #E5E7EB; border-radius: 8px; padding: 12px; text-align: center;">
                    <span style="font-size: 0.75rem; color: #6B7280; text-transform: uppercase;">Scaled RMSE</span>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #2563EB;">{rmse:.6f}</div>
                </div>
                <div style="flex: 1; background: white; border: 1px solid #E5E7EB; border-radius: 8px; padding: 12px; text-align: center;">
                    <span style="font-size: 0.75rem; color: #6B7280; text-transform: uppercase;">Convergence Delta</span>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #10B981;">{delta:.6f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Update loss curve every 20 epochs or final epoch to reduce Plotly rendering overhead
        if epoch % 20 == 0 or epoch == max_epochs or delta < tolerance:
            epoch_x = list(range(1, len(losses) + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epoch_x, y=losses, mode='lines', name='MSE Loss', line=dict(color='#2563EB', width=2)))
            fig.update_layout(
                title=f"Training Loss Curve (Epoch {epoch})",
                xaxis_title="Epoch",
                yaxis_title="MSE Loss (Scaled)",
                template="simple_white",
                height=320,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
        if delta < tolerance:
            log_message(f"✅ SGD CONVERGED early at epoch {epoch} (delta {delta:.6f} < tol {tolerance})")
            break
            
        rmse_prev = rmse
        # Sleep slightly to make training visualization smooth
        time.sleep(0.002)
        
    if epochs_run == max_epochs:
        log_message("Reached maximum epochs limit.")
        
    total_duration = time.time() - start_time
    log_message(f"SGD loop completed. Elapsed time: {total_duration:.2f} seconds.")

    # Save registered model parameters
    log_message("Evaluating metrics and registering trained model...")
    
    # Save & Register the model
    model_data = training_service.train_model(
        df, model_type, data_type, 
        learning_rate=lr, max_epochs=max_epochs, tolerance=tolerance,
        callback=None # we ran training loop manually for real visual UI progress
    )
    
    # Update active model in session state
    st.session_state["active_model"] = model_data
    
    metrics = model_data["metrics"]
    log_message(f"✓ Model successfully registered as: {model_data['name']}")
    log_message(f"Evaluation Metrics: R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}%")
    
    render_alert(f"Successfully trained and registered **{model_data['name']}**!", type="success")
    
    # Force rerun to show updated model in history
    st.rerun()

# -----------------------------------------------------
# MODEL REGISTRY / HISTORY SECTION
# -----------------------------------------------------
st.markdown("---")
st.markdown("### Registered Models Registry")

history = model_repo.list_models()
if not history:
    st.info("No registered models found. Train a model above to populate the registry.")
else:
    # Build dataframe for nice history table
    records = []
    for m in history:
        active_tag = "🟢 ACTIVE" if m["is_active"] else "⚪ INACTIVE"
        records.append({
            "Active Status": active_tag,
            "Model Name": m["name"],
            "Type": m["model_type"],
            "Dataset Segment": m["data_type"],
            "R² Score": f"{m['metrics'].get('r2', 0.0):.4f}",
            "MAE": f"{m['metrics'].get('mae', 0.0):.2f}",
            "RMSE": f"{m['metrics'].get('rmse', 0.0):.2f}",
            "MAPE": f"{m['metrics'].get('mape', 0.0):.2f}%",
            "Training Timestamp": m["timestamp"],
            "filename": m["filename"]
        })
    
    history_df = pd.DataFrame(records)
    
    # Display table (excluding filename)
    st.table(history_df.drop(columns=["filename"]))
    
    # Allow user to activate a model
    c_sel, c_act = st.columns([3, 1])
    with c_sel:
        select_to_activate = st.selectbox("Select model to activate", history_df["Model Name"].tolist())
    with c_act:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        activate_btn = st.button("Set Selected Active", use_container_width=True)
        if activate_btn:
            # find filename
            selected_filename = history_df[history_df["Model Name"] == select_to_activate]["filename"].values[0]
            model_repo.set_active_model(selected_filename)
            st.session_state["active_model"] = model_repo.load_model(selected_filename)
            render_alert(f"Activated model: {select_to_activate}", type="success")
            st.rerun()
