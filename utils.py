# utils.py - Utility functions for parsing and fitting

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import pywt

def parse_excel(uploaded_file, min_points=2, average_duplicates=True):
    try:
        df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {str(e)}")
    lines = []
    skipped_lines = []
    row = 0
    while row < df.shape[0]:
        if not pd.isna(df.iloc[row, 0]) and pd.isna(df.iloc[row, 1]):
            line_name = str(df.iloc[row, 0]).strip()
            x_data = []
            y_data = []
            row += 1
            invalid_x = False
            while row < df.shape[0]:
                if (not pd.isna(df.iloc[row, 0]) and pd.isna(df.iloc[row, 1])) or pd.isna(df.iloc[row, 0]):
                    break
                if not pd.isna(df.iloc[row, 1]):
                    raw_x = df.iloc[row, 0]
                    x_val = pd.to_numeric(raw_x, errors='coerce')
                    y_val = pd.to_numeric(df.iloc[row, 1], errors='coerce')
                    if pd.isna(x_val) or pd.isna(y_val):
                        print(f"Warning: Line '{line_name}' has invalid x or y at row {row+1}: x={raw_x}, y={df.iloc[row, 1]}")
                        invalid_x = True
                    else:
                        x_data.append(float(x_val))
                        y_data.append(float(y_val))
                row += 1
            if len(x_data) >= min_points:
                if average_duplicates:
                    # Aggregate duplicates by averaging y values
                    df_temp = pd.DataFrame({'x': x_data, 'y': y_data})
                    df_temp = df_temp.groupby('x', as_index=False).mean()
                    x_data = df_temp['x'].to_numpy()
                    y_data = df_temp['y'].to_numpy()
                    has_duplicates = len(x_data) != len(np.unique(x_data))
                else:
                    has_duplicates = len(np.unique(x_data)) != len(x_data)
                # Sort by x
                x_data, y_data = zip(*sorted(zip(x_data, y_data)))
                x_data = np.array(x_data)
                y_data = np.array(y_data)
                if has_duplicates and not average_duplicates:
                    print(f"Warning: Line '{line_name}' has duplicate x values, may be skipped for Spline, Savitzky-Golay, LOWESS, Exponential Smoothing, Gaussian Smoothing, or Wavelet Denoising.")
                    skipped_lines.append(line_name)
                lines.append((line_name, x_data, y_data, has_duplicates, invalid_x))
            else:
                print(f"Warning: Line '{line_name}' skipped: only {len(x_data)} valid points (need at least {min_points}).")
        else:
            row += 1
    return lines, skipped_lines

def suggest_best_poly_degree(x, y, max_degree=10):
    best_degree = 1
    best_r2 = -float('inf')
    n_points = len(x)
    max_degree = min(max_degree, n_points - 1)
    for degree in range(1, max_degree + 1):
        if n_points > degree:
            coeffs = np.polyfit(x, y, deg=degree)
            poly = np.poly1d(coeffs)
            y_pred = poly(x)
            r2 = r2_score(y, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_degree = degree
    return best_degree, best_r2

def calculate_adjusted_r2(r2, n, p):
    if n > p + 1:
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return -float('inf')

def suggest_best_method(x, y, has_invalid_x):
    """
    Suggest the best fitting method based on Adjusted R².
    Only Polynomial, Exponential, Logarithmic, and Compound Poly+Log are compared.
    Spline, Savitzky-Golay, LOWESS, Exponential Smoothing, Gaussian Smoothing, and Wavelet Denoising
    are excluded as they are smoothing methods, not parametric fits for model comparison.
    """
    results = []
    n = len(x)
    
    # Polynomial
    try:
        best_poly_degree, best_poly_r2 = suggest_best_poly_degree(x, y)
        coeffs, r2, desc = fit_polynomial(x, y, degree=best_poly_degree)
        p = best_poly_degree + 1
        adj_r2 = calculate_adjusted_r2(r2, n, p)
        results.append(("Polynomial", coeffs, r2, adj_r2, f"Polynomial (degree {best_poly_degree})", {'degree': best_poly_degree}))
    except Exception as e:
        print(f"Polynomial suggestion failed: {str(e)}")
    
    # Exponential
    try:
        coeffs, r2, desc = fit_exponential(x, y)
        if coeffs:
            p = len(coeffs)
            adj_r2 = calculate_adjusted_r2(r2, n, p)
            results.append(("Exponential", coeffs, r2, adj_r2, desc, {}))
    except Exception as e:
        print(f"Exponential suggestion failed: {str(e)}")
    
    # Logarithmic (filter x > 0)
    if not has_invalid_x:
        try:
            valid = x > 0
            x_valid = x[valid]
            y_valid = y[valid]
            if len(x_valid) >= 2:
                coeffs, r2, desc = fit_logarithmic(x_valid, y_valid)
                if coeffs:
                    p = len(coeffs)
                    adj_r2 = calculate_adjusted_r2(r2, len(x_valid), p)
                    results.append(("Logarithmic", coeffs, r2, adj_r2, desc, {}))
        except Exception as e:
            print(f"Logarithmic suggestion failed: {str(e)}")
    
    # Compound Poly+Log (filter x > 0)
    if not has_invalid_x:
        try:
            valid = x > 0
            x_valid = x[valid]
            y_valid = y[valid]
            if len(x_valid) >= 3:
                coeffs, r2, desc = fit_compound_poly_log(x_valid, y_valid)
                if coeffs:
                    p = len(coeffs)
                    adj_r2 = calculate_adjusted_r2(r2, len(x_valid), p)
                    results.append(("Compound Poly+Log", coeffs, r2, adj_r2, desc, {}))
        except Exception as e:
            print(f"Compound Poly+Log suggestion failed: {str(e)}")
    
    # Select best or default to Polynomial if all fail
    if results:
        best_result = max(results, key=lambda x: x[3])
    else:
        best_result = ("Polynomial", None, 0, -float('inf'), "Polynomial failed", {'degree': 1})
    return best_result

def fit_polynomial(x, y, degree):
    try:
        if len(x) <= degree:
            raise ValueError("Insufficient points for polynomial degree")
        coeffs = np.polyfit(x, y, deg=degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)
        r2 = r2_score(y, y_pred)
        return coeffs.tolist(), r2, f"Polynomial (degree {degree})"
    except Exception as e:
        return None, None, f"Polynomial fit failed: {str(e)}"

def fit_exponential(x, y):
    def exp_model(x, a, b, c):
        return a * np.exp(b * x) + c
    try:
        if len(x) < 3:
            raise ValueError("Insufficient points for exponential fit")
        popt, _ = curve_fit(exp_model, x, y, p0=[1, 0.1, 1], maxfev=10000)
        y_pred = exp_model(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt.tolist(), r2, "Exponential: y = a * exp(b*x) + c"
    except Exception as e:
        return None, None, f"Exponential fit failed: {str(e)}"

def fit_logarithmic(x, y):
    def log_model(x, a, b):
        return a * np.log(x) + b
    try:
        valid = x > 0
        x_valid = x[valid]
        y_valid = y[valid]
        if len(x_valid) < 2:
            return None, None, "Logarithmic fit failed: insufficient valid points (x > 0)"
        popt, _ = curve_fit(log_model, x_valid, y_valid, p0=[1, 1])
        y_pred = log_model(x_valid, *popt)
        r2 = r2_score(y_valid, y_pred)
        return popt.tolist(), r2, "Logarithmic: y = a * log(x) + b"
    except Exception as e:
        return None, None, f"Logarithmic fit failed: {str(e)}"

def fit_compound_poly_log(x, y):
    def poly_log_model(x, a, b, c):
        return a * x**2 + b * np.log(x) + c
    try:
        valid = x > 0
        x_valid = x[valid]
        y_valid = y[valid]
        if len(x_valid) < 3:
            return None, None, "Compound Poly+Log fit failed: insufficient valid points (x > 0)"
        popt, _ = curve_fit(poly_log_model, x_valid, y_valid, p0=[1, 1, 1])
        y_pred = poly_log_model(x_valid, *popt)
        r2 = r2_score(y_valid, y_pred)
        return popt.tolist(), r2, "Compound Poly+Log: y = a*x^2 + b*log(x) + c"
    except Exception as e:
        return None, None, f"Compound Poly+Log fit failed: {str(e)}"

def fit_spline(x, y, degree=3):
    try:
        if len(x) < degree + 1:
            raise ValueError(f"Insufficient points for spline degree {degree}")
        spline = UnivariateSpline(x, y, k=degree, s=0)
        y_pred = spline(x)
        r2 = r2_score(y, y_pred)
        coeffs = spline.get_coeffs().tolist()
        knots = spline.get_knots().tolist()
        return coeffs + knots, r2, f"Spline (degree {degree}): coeffs then knots"
    except ValueError as e:
        return None, None, f"Spline fit failed: {str(e)}"

def fit_savgol(x, y, window=5, polyorder=2):
    try:
        if len(x) < max(3, window):
            raise ValueError(f"Insufficient points ({len(x)}) for Savitzky-Golay with window {window}")
        if window % 2 == 0:
            raise ValueError("Window length must be odd")
        if window <= polyorder:
            raise ValueError("Window length must be greater than polynomial order")
        if polyorder < 0:
            raise ValueError("Polynomial order must be non-negative")
        if window > len(y):
            raise ValueError("Window length exceeds data length")
        smoothed_y = savgol_filter(y, window, polyorder)
        r2 = r2_score(y, smoothed_y)
        return [window, polyorder], r2, f"Savitzky-Golay (window {window}, order {polyorder})"
    except Exception as e:
        return None, None, f"Savitzky-Golay fit failed: {str(e)}"

def fit_lowess(x, y, frac=0.3):
    try:
        if len(x) < 3:
            raise ValueError(f"Insufficient points ({len(x)}) for LOWESS")
        if not 0 < frac < 1:
            raise ValueError("Fraction must be strictly between 0 and 1")
        smoothed_y = lowess(y, x, frac=frac, return_sorted=False)
        r2 = r2_score(y, smoothed_y)
        return [frac], r2, f"LOWESS (frac {frac})"
    except Exception as e:
        return None, None, f"LOWESS fit failed: {str(e)}"

def fit_exponential_smoothing(x, y, alpha=0.5):
    try:
        if len(x) < 3:
            raise ValueError(f"Insufficient points ({len(x)}) for Exponential Smoothing")
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be strictly between 0 and 1")
        smoothed_y = pd.Series(y).ewm(alpha=alpha).mean().values
        r2 = r2_score(y, smoothed_y)
        return [alpha], r2, f"Exponential Smoothing (alpha {alpha})"
    except Exception as e:
        return None, None, f"Exponential Smoothing fit failed: {str(e)}"

def fit_gaussian_smoothing(x, y, sigma=1.0):
    try:
        if len(x) < 3:
            raise ValueError(f"Insufficient points ({len(x)}) for Gaussian Smoothing")
        if sigma <= 0:
            raise ValueError("Sigma must be positive")
        smoothed_y = gaussian_filter1d(y, sigma)
        r2 = r2_score(y, smoothed_y)
        return [sigma], r2, f"Gaussian Smoothing (sigma {sigma})"
    except Exception as e:
        return None, None, f"Gaussian Smoothing fit failed: {str(e)}"

def fit_wavelet_denoising(x, y, wavelet='db4', level=1, threshold=0.1):
    try:
        if len(x) < 3:
            raise ValueError(f"Insufficient points ({len(x)}) for Wavelet Denoising")
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Invalid wavelet: {wavelet}")
        max_level = pywt.dwt_max_level(len(y), wavelet)
        if level > max_level:
            raise ValueError(f"Decomposition level {level} exceeds maximum {max_level} for wavelet {wavelet}")
        coeffs_wave = pywt.wavedec(y, wavelet, level=level)
        for i in range(1, len(coeffs_wave)):
            coeffs_wave[i] = pywt.threshold(coeffs_wave[i], threshold * np.max(coeffs_wave[i]), mode='soft')
        smoothed_y = pywt.waverec(coeffs_wave, wavelet)
        if len(smoothed_y) > len(y):
            smoothed_y = smoothed_y[:len(y)]
        elif len(smoothed_y) < len(y):
            smoothed_y = np.pad(smoothed_y, (0, len(y) - len(smoothed_y)), mode='edge')
        r2 = r2_score(y, smoothed_y)
        return [wavelet, level, threshold], r2, f"Wavelet Denoising ({wavelet}, level {level}, threshold {threshold})"
    except Exception as e:
        return None, None, f"Wavelet Denoising fit failed: {str(e)}"

def fit_sine(x, y, frequency_guess=1.0):
    """
    Fit a sine model: y = a * sin(b * x + c) + d
    Args:
        x, y: Data points
        frequency_guess: Initial guess for b (angular frequency, e.g., 2*pi/period)
    Returns: Coefficients [a, b, c, d], R², description
    """
    def sin_model(x, a, b, c, d):
        return a * np.sin(b * x + c) + d
    try:
        if len(x) < 4:
            raise ValueError("Insufficient points for sine fit (need at least 4)")
        p0 = [np.std(y), frequency_guess, 0, np.mean(y)]
        popt, _ = curve_fit(sin_model, x, y, p0=p0, maxfev=10000)
        y_pred = sin_model(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt.tolist(), r2, f"Sine: y = a * sin(b*x + c) + d (freq_guess {frequency_guess})"
    except Exception as e:
        return None, None, f"Sine fit failed: {str(e)}"

def fit_cosine(x, y, frequency_guess=1.0):
    """
    Fit a cosine model: y = a * cos(b * x + c) + d
    Args:
        x, y: Data points
        frequency_guess: Initial guess for b (angular frequency, e.g., 2*pi/period)
    Returns: Coefficients [a, b, c, d], R², description
    """
    def cos_model(x, a, b, c, d):
        return a * np.cos(b * x + c) + d
    try:
        if len(x) < 4:
            raise ValueError("Insufficient points for cosine fit (need at least 4)")
        p0 = [np.std(y), frequency_guess, 0, np.mean(y)]
        popt, _ = curve_fit(cos_model, x, y, p0=p0, maxfev=10000)
        y_pred = cos_model(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt.tolist(), r2, f"Cosine: y = a * cos(b*x + c) + d (freq_guess {frequency_guess})"
    except Exception as e:
        return None, None, f"Cosine fit failed: {str(e)}"

def fit_tangent(x, y, frequency_guess=1.0):
    """
    Fit a tangent model: y = a * tan(b * x + c) + d
    Args:
        x, y: Data points
        frequency_guess: Initial guess for b (angular frequency, e.g., 2*pi/period)
    Returns: Coefficients [a, b, c, d], R², description
    """
    def tan_model(x, a, b, c, d):
        return a * np.tan(b * x + c) + d
    try:
        if len(x) < 4:
            raise ValueError("Insufficient points for tangent fit (need at least 4)")
        # Bounds to avoid singularities (tan has poles at (n+0.5)*pi)
        bounds = ([-np.inf, 0, -np.pi, -np.inf], [np.inf, np.inf, np.pi, np.inf])
        p0 = [np.std(y), frequency_guess, 0, np.mean(y)]
        popt, _ = curve_fit(tan_model, x, y, p0=p0, bounds=bounds, maxfev=10000)
        y_pred = tan_model(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt.tolist(), r2, f"Tangent: y = a * tan(b*x + c) + d (freq_guess {frequency_guess})"
    except Exception as e:
        return None, None, f"Tangent fit failed: {str(e)}"

def fit_cotangent(x, y, frequency_guess=1.0):
    """
    Fit a cotangent model: y = a * cot(b * x + c) + d
    Args:
        x, y: Data points
        frequency_guess: Initial guess for b (angular frequency, e.g., 2*pi/period)
    Returns: Coefficients [a, b, c, d], R², description
    """
    def cot_model(x, a, b, c, d):
        return a / np.tan(b * x + c) + d
    try:
        if len(x) < 4:
            raise ValueError("Insufficient points for cotangent fit (need at least 4)")
        # Bounds to avoid singularities (cot has poles at n*pi)
        bounds = ([-np.inf, 0, -np.pi, -np.inf], [np.inf, np.inf, np.pi, np.inf])
        p0 = [np.std(y), frequency_guess, 0, np.mean(y)]
        popt, _ = curve_fit(cot_model, x, y, p0=p0, bounds=bounds, maxfev=10000)
        y_pred = cot_model(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt.tolist(), r2, f"Cotangent: y = a * cot(b*x + c) + d (freq_guess {frequency_guess})"
    except Exception as e:
        return None, None, f"Cotangent fit failed: {str(e)}"

def fit_sinh(x, y, scaling_guess=1.0):
    """
    Fit a hyperbolic sine model: y = a * sinh(b * x + c) + d
    Args:
        x, y: Data points
        scaling_guess: Initial guess for b (scaling factor)
    Returns: Coefficients [a, b, c, d], R², description
    """
    def sinh_model(x, a, b, c, d):
        return a * np.sinh(b * x + c) + d
    try:
        if len(x) < 4:
            raise ValueError("Insufficient points for sinh fit (need at least 4)")
        p0 = [np.std(y), scaling_guess, 0, np.mean(y)]
        popt, _ = curve_fit(sinh_model, x, y, p0=p0, maxfev=10000)
        y_pred = sinh_model(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt.tolist(), r2, f"Sinh: y = a * sinh(b*x + c) + d (scaling_guess {scaling_guess})"
    except Exception as e:
        return None, None, f"Sinh fit failed: {str(e)}"

def fit_cosh(x, y, scaling_guess=1.0):
    """
    Fit a hyperbolic cosine model: y = a * cosh(b * x + c) + d
    Args:
        x, y: Data points
        scaling_guess: Initial guess for b (scaling factor)
    Returns: Coefficients [a, b, c, d], R², description
    """
    def cosh_model(x, a, b, c, d):
        return a * np.cosh(b * x + c) + d
    try:
        if len(x) < 4:
            raise ValueError("Insufficient points for cosh fit (need at least 4)")
        p0 = [np.std(y), scaling_guess, 0, np.mean(y)]
        popt, _ = curve_fit(cosh_model, x, y, p0=p0, maxfev=10000)
        y_pred = cosh_model(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt.tolist(), r2, f"Cosh: y = a * cosh(b*x + c) + d (scaling_guess {scaling_guess})"
    except Exception as e:
        return None, None, f"Cosh fit failed: {str(e)}"

def fit_tanh(x, y, scaling_guess=1.0):
    """
    Fit a hyperbolic tangent model: y = a * tanh(b * x + c) + d
    Args:
        x, y: Data points
        scaling_guess: Initial guess for b (scaling factor)
    Returns: Coefficients [a, b, c, d], R², description
    """
    def tanh_model(x, a, b, c, d):
        return a * np.tanh(b * x + c) + d
    try:
        if len(x) < 4:
            raise ValueError("Insufficient points for tanh fit (need at least 4)")
        p0 = [np.std(y), scaling_guess, 0, np.mean(y)]
        popt, _ = curve_fit(tanh_model, x, y, p0=p0, maxfev=10000)
        y_pred = tanh_model(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt.tolist(), r2, f"Tanh: y = a * tanh(b*x + c) + d (scaling_guess {scaling_guess})"
    except Exception as e:
        return None, None, f"Tanh fit failed: {str(e)}"
