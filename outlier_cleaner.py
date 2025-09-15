# outlier_cleaner.py - Module for outlier detection and cleaning in Curve Fitting App

import streamlit as st
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

def outlier_cleaner_ui():
    """
    Render UI for Outlier Detection and Cleaning mode.
    Returns: Dictionary of parameters for outlier detection and fitting
    """
    params = {}
    st.markdown("Select an outlier detection method or specify the number of outliers to remove, then choose a fitting method for the cleaned data.")

    # Outlier detection method
    detection_method = st.selectbox("Outlier Detection Method", 
                                    ["Z-score", "IQR", "Isolation Forest", "Fixed Number"])
    params['detection_method'] = detection_method

    if detection_method == "Z-score":
        params['z_threshold'] = st.slider("Z-score Threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    elif detection_method == "IQR":
        params['iqr_multiplier'] = st.slider("IQR Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    elif detection_method == "IsolationForest":
        params['contamination'] = st.slider("Contamination (proportion of outliers)", 
                                           min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    elif detection_method == "Fixed Number":
        params['n_outliers'] = st.number_input("Number of Outliers to Remove", 
                                              min_value=0, max_value=100, value=1)

    # Fitting method for cleaned data
    method_options = ["Polynomial", "Exponential", "Logarithmic", "Compound Poly+Log", 
                      "Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", 
                      "Gaussian Smoothing", "Wavelet Denoising", "Random Forest"]
    params['fit_method'] = st.selectbox("Fitting Method for Cleaned Data", method_options)

    # Parameters for the fitting method
    if params['fit_method'] == "Polynomial":
        params['degree'] = st.number_input("Polynomial Degree", min_value=1, max_value=10, value=2)
    elif params['fit_method'] == "Spline":
        params['degree'] = st.number_input("Spline Degree (1-5, 3=cubic)", min_value=1, max_value=5, value=3)
    elif params['fit_method'] == "Savitzky-Golay":
        params['window'] = st.number_input("Window length (odd number >=3)", min_value=3, max_value=101, value=5, step=2)
        params['polyorder'] = st.number_input("Polynomial order", min_value=0, max_value=5, value=2)
    elif params['fit_method'] == "LOWESS":
        params['frac'] = st.slider("Fraction of data for local fit", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
    elif params['fit_method'] == "Exponential Smoothing":
        params['alpha'] = st.slider("Smoothing factor (alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    elif params['fit_method'] == "Gaussian Smoothing":
        params['sigma'] = st.number_input("Gaussian sigma", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    elif params['fit_method'] == "Wavelet Denoising":
        params['wavelet'] = st.selectbox("Wavelet family", options=['db4', 'haar', 'sym4', 'coif1'], index=0)
        params['level'] = st.number_input("Decomposition level", min_value=1, max_value=5, value=1)
        params['threshold'] = st.slider("Threshold factor", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    return params

def detect_outliers(x, y, detection_method, params, fit_func=None):
    """
    Detect outliers in the data.
    Args:
        x, y: Data points
        detection_method: "Z-score", "IQR", "Isolation Forest", or "Fixed Number"
        params: Parameters including detection settings and fit_method
        fit_func: Fitting function for residual-based outlier detection (if needed)
    Returns: Boolean mask (True for non-outliers), indices of outliers
    """
    try:
        if detection_method == "Z-score":
            z_scores = np.abs(zscore(y))
            mask = z_scores <= params['z_threshold']
            outlier_indices = np.where(~mask)[0]
        elif detection_method == "IQR":
            q1, q3 = np.percentile(y, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - params['iqr_multiplier'] * iqr
            upper_bound = q3 + params['iqr_multiplier'] * iqr
            mask = (y >= lower_bound) & (y <= upper_bound)
            outlier_indices = np.where(~mask)[0]
        elif detection_method == "Isolation Forest":
            X = np.column_stack((x, y))
            iso = IsolationForest(contamination=params['contamination'], random_state=42)
            labels = iso.fit_predict(X)
            mask = labels != -1  # -1 indicates outlier
            outlier_indices = np.where(~mask)[0]
        elif detection_method == "Fixed Number":
            n_outliers = min(params['n_outliers'], len(y))
            if fit_func:
                coeffs, _, _ = fit_func(x, y, **{k: v for k, v in params.items() if k not in ['detection_method', 'n_outliers', 'fit_method']})
                if coeffs is not None:
                    if params['fit_method'] == "Polynomial":
                        y_pred = np.polyval(coeffs, x)
                    else:
                        # For non-polynomial methods, assume fit_func returns a callable model
                        y_pred = coeffs(x)
                    residuals = np.abs(y - y_pred)
                    outlier_indices = np.argsort(residuals)[-n_outliers:]
                    mask = np.ones(len(y), dtype=bool)
                    mask[outlier_indices] = False
                else:
                    return None, None, "Failed to fit for residual-based outlier detection"
            else:
                return None, None, "Fit function required for Fixed Number method"
        return mask, outlier_indices
    except Exception as e:
        return None, None, f"Outlier detection failed: {str(e)}"

def plot_cleaned_data(x, y, mask, coeffs, method, params):
    """
    Plot original data, cleaned data, and fitted curve.
    Returns: Matplotlib figure
    """
    fig, ax = plt.subplots()
    # Plot all points
    ax.scatter(x, y, color='blue', label='Original Data', s=50, alpha=0.5)
    # Plot cleaned points
    if mask is not None:
        ax.scatter(x[mask], y[mask], color='green', label='Cleaned Data', s=50)
    # Plot outliers
    outlier_indices = np.where(~mask)[0] if mask is not None else []
    if len(outlier_indices) > 0:
        ax.scatter(x[outlier_indices], y[outlier_indices], color='red', label='Outliers', s=100, marker='x')
    
    # Plot fitted curve (if applicable)
    if coeffs is not None:
        x_smooth = np.linspace(min(x), max(x), 200)
        if method == "Polynomial":
            y_smooth = np.polyval(coeffs, x_smooth)
        elif method in ["Spline", "LOWESS", "Savitzky-Golay", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"]:
            # For these methods, coeffs is a callable
            y_smooth = coeffs(x_smooth)
        elif method == "Random Forest":
            y_smooth = coeffs.predict(x_smooth.reshape(-1, 1))
        else:
            y_smooth = coeffs(x_smooth)  # Assume callable for other methods
        ax.plot(x_smooth, y_smooth, color='black', label=f'Fitted {method}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    return fig

def create_cleaned_excel(cleaned_results):
    """
    Create Excel file with cleaned data.
    Args:
        cleaned_results: List of (line_name, x_clean, y_clean, error_message)
    Returns: BytesIO object with Excel data
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        current_row = 0
        for line_name, x_clean, y_clean, error_message in cleaned_results:
            if error_message:
                continue
            df = pd.DataFrame({'A': [line_name] + [''] + list(x_clean), 
                              'B': [''] + [''] + list(y_clean)})
            df.to_excel(writer, sheet_name='Cleaned_Data', startrow=current_row, index=False, header=False)
            current_row += len(x_clean) + 2  # Data + empty row
    output.seek(0)
    return output
