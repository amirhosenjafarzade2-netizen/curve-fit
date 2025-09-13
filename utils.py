# utils.py - Utility functions for parsing and fitting

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

def parse_excel(uploaded_file, min_points=2):
    """
    Parse Excel file to extract lines with x,y points.
    Each line starts with a name in column A (B empty), followed by x in A, y in B.
    """
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
    """
    Suggest the best polynomial degree based on R².
    """
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
    return coeffs + knots, r2, "Cubic Spline: coeffs then knots"
