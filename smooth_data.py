import streamlit as st
import numpy as np
import pandas as pd
import io
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import pywt
from sklearn.ensemble import RandomForestRegressor

def smooth_data_ui():
    """
    Render UI for Download Smoothed Data mode.
    Returns: Dictionary of parameters including n_points
    """
    params = {}
    params['n_points'] = st.number_input("Number of smoothed points per line", min_value=10, max_value=1000, value=200, step=10)
    method = st.session_state.get('method')
    if method == "Polynomial":
        params['degree'] = st.number_input("Polynomial Degree", min_value=1, max_value=10, value=2)
    elif method == "Spline":
        params['degree'] = st.number_input("Spline Degree (1-5, 3=cubic)", min_value=1, max_value=5, value=3)
    elif method == "Savitzky-Golay":
        params['window'] = st.number_input("Window length (odd number >=3)", min_value=3, max_value=101, value=5, step=2)
        params['polyorder'] = st.number_input("Polynomial order", min_value=0, max_value=5, value=2)
    elif method == "LOWESS":
        params['frac'] = st.slider("Fraction of data for local fit", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
    elif method == "Exponential Smoothing":
        params['alpha'] = st.slider("Smoothing factor (alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    elif method == "Gaussian Smoothing":
        params['sigma'] = st.number_input("Gaussian sigma", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    elif method == "Wavelet Denoising":
        params['wavelet'] = st.selectbox("Wavelet family", options=['db4', 'haar', 'sym4', 'coif1'], index=0)
        params['level'] = st.number_input("Decomposition level", min_value=1, max_value=5, value=1)
        params['threshold'] = st.slider("Threshold factor", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    elif method == "Random Forest":
        params['n_estimators'] = st.number_input("Number of trees", min_value=10, max_value=500, value=100, step=10)
    return params

def generate_smoothed_data(lines, method, params, fit_funcs):
    """
    Generate smoothed x and y data for each line using the specified method.
    Args:
        lines: List of tuples (line_name, x, y, has_duplicates, has_invalid_x)
        method: Selected fitting method
        params: Parameters including n_points
        fit_funcs: Dictionary of fitting functions
    Returns: List of (line_name, x_smooth, y_smooth, error_message)
    """
    results = []
    n_points = params.pop('n_points', 200)  # Remove n_points from params for fitting

    for line_name, x, y, has_duplicates, has_invalid_x in lines:
        try:
            if method in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"] and has_duplicates and not params.get('average_duplicates', True):
                results.append((line_name, None, None, f"Skipped for {method} due to duplicate x values"))
                continue

            min_points = 3
            if method == "Polynomial":
                min_points = params.get('degree', 2) + 1
            elif method == "Spline":
                min_points = params.get('degree', 3) + 1
            elif method == "Savitzky-Golay":
                min_points = params.get('window', 5)
            if len(x) < min_points:
                results.append((line_name, None, None, f"Skipped for {method}: Insufficient points (need {min_points})"))
                continue

            x_smooth = np.linspace(min(x), max(x), n_points)
            fit_func = fit_funcs[method]
            coeffs, _, _ = fit_func(x, y, **params)

            if coeffs is None:
                results.append((line_name, None, None, f"{method} fit failed"))
                continue

            # Generate y_smooth based on method
            if method == "Polynomial":
                y_smooth = np.polyval(coeffs, x_smooth)
            elif method == "Exponential":
                a, b, c = coeffs
                y_smooth = a * np.exp(b * x_smooth) + c
            elif method == "Logarithmic":
                a, b = coeffs
                y_smooth = a * np.log(x_smooth) + b
            elif method == "Compound Poly+Log":
                a, b, c = coeffs
                y_smooth = a * x_smooth**2 + b * np.log(x_smooth) + c
            elif method == "Spline":
                spline = UnivariateSpline(x, y, k=params.get('degree', 3), s=0)
                y_smooth = spline(x_smooth)
            elif method == "Savitzky-Golay":
                window = params.get('window', 5)
                polyorder = params.get('polyorder', 2)
                y_smooth = savgol_filter(y, window, polyorder)[::len(y)//n_points][:n_points]
                if len(y_smooth) < n_points:
                    y_smooth = np.pad(y_smooth, (0, n_points - len(y_smooth)), mode='edge')
            elif method == "LOWESS":
                frac = params.get('frac', 0.3)
                lowess_result = lowess(y, x, frac=frac, return_sorted=False)
                y_smooth = np.interp(x_smooth, x, lowess_result)
            elif method == "Exponential Smoothing":
                alpha = params.get('alpha', 0.5)
                y_smooth = np.array([y[0]])
                for i in range(1, n_points):
                    y_smooth = np.append(y_smooth, alpha * y[i % len(y)] + (1 - alpha) * y_smooth[-1])
                if len(y_smooth) < n_points:
                    y_smooth = np.pad(y_smooth, (0, n_points - len(y_smooth)), mode='edge')
            elif method == "Gaussian Smoothing":
                sigma = params.get('sigma', 1.0)
                y_smooth = gaussian_filter1d(y, sigma)[::len(y)//n_points][:n_points]
                if len(y_smooth) < n_points:
                    y_smooth = np.pad(y_smooth, (0, n_points - len(y_smooth)), mode='edge')
            elif method == "Wavelet Denoising":
                wavelet = params.get('wavelet', 'db4')
                level = params.get('level', 1)
                threshold = params.get('threshold', 0.1)
                coeffs = pywt.wavedec(y, wavelet, level=level)
                threshold_value = threshold * np.max(np.abs(coeffs[-1]))
                coeffs = [pywt.threshold(c, threshold_value, mode='soft') for c in coeffs]
                y_smooth = pywt.waverec(coeffs, wavelet)[::len(y)//n_points][:n_points]
                if len(y_smooth) < n_points:
                    y_smooth = np.pad(y_smooth, (0, n_points - len(y_smooth)), mode='edge')
            elif method == "Random Forest":
                n_estimators = coeffs[0]
                model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=42)
                model.fit(x.reshape(-1, 1), y)
                y_smooth = model.predict(x_smooth.reshape(-1, 1))

            results.append((line_name, x_smooth, y_smooth, None))
        except Exception as e:
            results.append((line_name, None, None, f"Failed to generate smoothed data: {str(e)}"))

    return results

def create_smoothed_excel(results):
    """
    Create an Excel file with smoothed data in the same format as input.
    Args:
        results: List of (line_name, x_smooth, y_smooth, error_message)
    Returns: BytesIO object with Excel data
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet('Smoothed_Data')
        row = 0
        for line_name, x_smooth, y_smooth, error_message in results:
            if error_message:
                worksheet.write(row, 0, f"Error: {error_message}")
                row += 2
                continue
            worksheet.write(row, 0, line_name)
            row += 2
            for x_val, y_val in zip(x_smooth, y_smooth):
                worksheet.write(row, 0, x_val)
                worksheet.write(row, 1, y_val)
                row += 1
            row += 1  # Empty row between lines
    output.seek(0)
    return output
