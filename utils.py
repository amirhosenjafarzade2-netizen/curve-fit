# utils.py - Utility functions for parsing and fitting

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

def parse_excel(uploaded_file, min_points=2):
    df = pd.read_excel(uploaded_file, header=None)
    lines = []
    row = 0
    while row < df.shape[0]:
        if not pd.isna(df.iloc[row, 0]) and pd.isna(df.iloc[row, 1]):
            line_name = str(df.iloc[row, 0]).strip()
            x_data = []
            y_data = []
            row += 1
            while row < df.shape[0]:
                if (not pd.isna(df.iloc[row, 0]) and pd.isna(df.iloc[row, 1])) or pd.isna(df.iloc[row, 0]):
                    break
                if not pd.isna(df.iloc[row, 1]):
                    x_val = pd.to_numeric(df.iloc[row, 0], errors='coerce')
                    y_val = pd.to_numeric(df.iloc[row, 1], errors='coerce')
                    if not pd.isna(x_val) and not pd.isna(y_val):
                        x_data.append(float(x_val))
                        y_data.append(float(y_val))
                row += 1
            if len(x_data) >= min_points:
                lines.append((line_name, np.array(x_data), np.array(y_data)))
        else:
            row += 1
    return lines

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

def suggest_best_method(x, y, max_poly_degree=10):
    results = []
    n = len(x)
    
    # Polynomial
    best_poly_degree, best_poly_r2 = suggest_best_poly_degree(x, y, max_poly_degree)
    coeffs, r2, desc = fit_polynomial(x, y, degree=best_poly_degree)
    p = best_poly_degree + 1
    adj_r2 = calculate_adjusted_r2(r2, n, p)
    results.append(("Polynomial", coeffs, r2, adj_r2, f"Polynomial (degree {best_poly_degree})", {'degree': best_poly_degree}))
    
    # Exponential
    coeffs, r2, desc = fit_exponential(x, y)
    if coeffs:
        p = len(coeffs)
        adj_r2 = calculate_adjusted_r2(r2, n, p)
        results.append(("Exponential", coeffs, r2, adj_r2, desc, {}))
    
    # Logarithmic
    coeffs, r2, desc = fit_logarithmic(x, y)
    if coeffs:
        p = len(coeffs)
        adj_r2 = calculate_adjusted_r2(r2, n, p)
        results.append(("Logarithmic", coeffs, r2, adj_r2, desc, {}))
    
    # Compound Poly+Log
    coeffs, r2, desc = fit_compound_poly_log(x, y)
    if coeffs:
        p = len(coeffs)
        adj_r2 = calculate_adjusted_r2(r2, n, p)
        results.append(("Compound Poly+Log", coeffs, r2, adj_r2, desc, {}))
    
    # Spline (default degree 3)
    coeffs, r2, desc = fit_spline(x, y, degree=3)
    p = len(coeffs) // 2  # Approximate parameters
    adj_r2 = calculate_adjusted_r2(r2, n, p)
    results.append(("Spline", coeffs, r2, adj_r2, "Cubic Spline: coeffs then knots", {'degree': 3}))
    
    # Select best
    best_result = max(results, key=lambda x: x[3])
    return best_result

def fit_polynomial(x, y, degree):
    coeffs = np.polyfit(x, y, deg=degree)
    poly = np.poly1d(coeffs)
    y_pred = poly(x)
    r2 = r2_score(y, y_pred)
    return coeffs.tolist(), r2, f"Polynomial (degree {degree})"

def fit_exponential(x, y):
    def exp_model(x, a, b, c):
        return a * np.exp(b * x) + c
    try:
        popt, _ = curve_fit(exp_model, x, y, p0=[1, 0.1, 1], maxfev=10000)
        y_pred = exp_model(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt.tolist(), r2, "Exponential: y = a * exp(b*x) + c"
    except:
        return None, None, "Exponential fit failed"

def fit_logarithmic(x, y):
    def log_model(x, a, b):
        return a * np.log(x + 1e-10) + b
    try:
        popt, _ = curve_fit(log_model, x, y, p0=[1, 1])
        y_pred = log_model(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt.tolist(), r2, "Logarithmic: y = a * log(x) + b"
    except:
        return None, None, "Logarithmic fit failed"

def fit_compound_poly_log(x, y):
    def poly_log_model(x, a, b, c):
        return a * x**2 + b * np.log(x + 1e-10) + c
    try:
        popt, _ = curve_fit(poly_log_model, x, y, p0=[1, 1, 1])
        y_pred = poly_log_model(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt.tolist(), r2, "Compound Poly+Log: y = a*x^2 + b*log(x) + c"
    except:
        return None, None, "Compound fit failed"

def fit_spline(x, y, degree=3):
    spline = UnivariateSpline(x, y, k=degree, s=0)
    y_pred = spline(x)
    r2 = r2_score(y, y_pred)
    coeffs = spline.get_coeffs().tolist()
    knots = spline.get_knots().tolist()
    return coeffs + knots, r2, f"Spline (degree {degree}): coeffs then knots"
