# constrained_optimization.py - Standalone module for constrained coefficient optimization

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import streamlit as st
import io
import pandas as pd
import sympy as sp
from scipy.stats import iqr

def constrained_optimization_ui():
    """
    Render UI for Constrained Coefficient Optimization mode.
    Returns: Dictionary of parameters including constraints
    """
    params = {}
    st.markdown("Specify the fitting method, parameters, and restrictive points (x, y) that the curve must pass through. Multiple points can be added.")

    method_options = ["Polynomial"]
    params['method'] = st.selectbox("Choose Fitting Method", method_options)

    if params['method'] == "Polynomial":
        params['degree'] = st.number_input("Polynomial Degree", min_value=1, max_value=10, value=3)

    num_points = st.number_input("Number of restrictive points", min_value=0, max_value=10, value=2)
    params['constraints'] = []
    for i in range(num_points):
        col1, col2 = st.columns(2)
        with col1:
            x_val = st.number_input(f"Restrictive Point {i+1}: x", value=i * 1000.0, key=f"x_{i}")
        with col2:
            y_val = st.number_input(f"Restrictive Point {i+1}: y", value=i * 15000.0, key=f"y_{i}")
        params['constraints'].append((x_val, y_val))

    params['lambda_reg'] = st.slider("Regularization Strength", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    params['show_diagnostics'] = st.checkbox("Show detailed diagnostics", value=False)

    return params

def optimize_coefficients(x, y, method, params):
    """
    Optimize coefficients under constraints with maximum accuracy.
    Args:
        x, y: Data points (numpy arrays)
        method: Selected method (e.g., "Polynomial")
        params: Parameters including degree, constraints, lambda_reg, show_diagnostics
    Returns: coeffs, r2, model_desc or None, None, error_message
    """
    try:
        if method == "Polynomial":
            degree = params['degree']
            constraints = params['constraints']
            lambda_reg = params['lambda_reg']
            show_diagnostics = params.get('show_diagnostics', False)

            if len(x) < degree + 1:
                return None, None, f"Insufficient data points: need at least {degree + 1} for degree {degree}"

            if len(constraints) > degree + 1:
                return None, None, f"Too many constraints ({len(constraints)}) for degree {degree}"

            # Robust scaling using median and IQR
            x_med, x_iqr = np.median(x), iqr(x) or 1.0
            y_med, y_iqr = np.median(y), iqr(y) or 1.0
            x_scaled = (x - x_med) / (x_iqr / 1.349)  # 1.349 normalizes IQR to std for normal data
            y_scaled = (y - y_med) / (y_iqr / 1.349)
            constraints_scaled = [((cons_x - x_med) / (x_iqr / 1.349), (cons_y - y_med) / (y_iqr / 1.349)) for cons_x, cons_y in constraints]

            # Hybrid initial guess: Least-squares fit as starting point
            initial_coeffs = np.polyfit(x_scaled, y_scaled, degree)

            # Adaptive weighting based on data density and proximity to constraints
            def compute_weights(x_scaled, constraints_scaled):
                weights = np.ones_like(x_scaled)
                if constraints_scaled:
                    for cons_x, _ in constraints_scaled:
                        dist = (x_scaled - cons_x) ** 2
                        weights += 20.0 * np.exp(-dist / (2 * 0.2**2))  # Wider Gaussian (sigma=0.2)
                # Normalize weights
                weights /= np.mean(weights)
                # Boost weights for dense regions
                kernel = np.exp(-((x_scaled[:, None] - x_scaled[None, :]) ** 2) / (2 * 0.1**2))
                density = np.sum(kernel, axis=1)
                weights *= density / np.mean(density)
                return weights

            weights = compute_weights(x_scaled, constraints_scaled)

            # Objective function with weighted MSE
            def objective(coeffs):
                y_pred = np.polyval(coeffs, x_scaled)
                mse = np.average((y_scaled - y_pred) ** 2, weights=weights)
                reg = lambda_reg * np.sum(coeffs ** 2)  # L2 regularization
                return mse + reg

            # Constraints with high precision
            cons = [{'type': 'eq', 'fun': lambda coeffs, cx=cons_x, cy=cons_y: np.polyval(coeffs, cx) - cy} for cons_x, cons_y in constraints_scaled]

            # Optimize with tighter bounds and more iterations
            bounds = [(-50, 50) for _ in range(degree + 1)]  # Tighter bounds for stability
            res = minimize(objective, initial_coeffs, method='SLSQP', constraints=cons, bounds=bounds, options={'disp': show_diagnostics, 'maxiter': 2000, 'ftol': 1e-8})

            if res.success:
                coeffs_scaled = res.x
                # Rescale coefficients with sympy
                x_var = sp.symbols('x')
                z = (x_var - x_med) / (x_iqr / 1.349)
                p_scaled = sum([coeffs_scaled[i] * z**(degree - i) for i in range(degree + 1)])
                p_original = p_scaled * (y_iqr / 1.349) + y_med
                p_expanded = sp.expand(p_original)
                coeffs = [float(p_expanded.coeff(x_var ** (degree - i))) if p_expanded.coeff(x_var ** (degree - i)) else 0.0 for i in range(degree + 1)]

                y_pred = np.polyval(coeffs, x)
                r2 = r2_score(y, y_pred)

                # Verify constraints with high precision
                for cons_x, cons_y in constraints:
                    pred_y = np.polyval(coeffs, cons_x)
                    if abs(pred_y - cons_y) > 1e-8:
                        error_msg = f"Constraint not satisfied: at x={cons_x}, predicted y={pred_y:.8f} != {cons_y:.8f}"
                        if show_diagnostics:
                            error_msg += f"\nScaled coeffs={coeffs_scaled}"
                        return None, None, error_msg

                return coeffs, r2, f"Constrained Polynomial (degree {degree})"
            else:
                error_msg = f"Optimization failed: {res.message}"
                if show_diagnostics:
                    error_msg += f"\nDiagnostics: Initial coeffs={initial_coeffs}, Constraints={constraints_scaled}, Status={res.status}"
                return None, None, error_msg

        return None, None, f"Method {method} not supported"

    except Exception as e:
        error_msg = f"Optimization failed: {str(e)}"
        if params.get('show_diagnostics', False):
            error_msg += f"\nDiagnostics: x={x}, y={y}, params={params}"
        return None, None, error_msg

def plot_constrained_fit(x, y, coeffs, method, params):
    """
    Plot the constrained fit with enhanced visualization.
    Returns: Matplotlib figure
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data Points', s=50, alpha=0.6)
    x_smooth = np.linspace(min(x), max(x), 300)
    if method == "Polynomial":
        y_smooth = np.polyval(coeffs, x_smooth)
        ax.plot(x_smooth, y_smooth, color='red', label=f'Constrained Polynomial (deg {params.get("degree", "?")})')
    for cons_x, cons_y in params.get('constraints', []):
        ax.scatter(cons_x, cons_y, color='green', label='Restrictive Point', s=100, marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def create_constrained_excel(results):
    """
    Create an Excel file with optimized coefficients and R².
    Args:
        results: List of (line_name, coeffs, r2, model_desc)
    Returns: BytesIO object with Excel data
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        max_coeffs = max(len(row[1]) for row in results if row[1] is not None) if results else 0
        columns = ['Line Name', 'Model Desc'] + [f'coeff_{i}' for i in range(max_coeffs)] + ['R2']
        data = []
        for line_name, coeffs, r2, model_desc in results:
            row = [line_name, model_desc] + (coeffs if coeffs else []) + [r2]
            data.append(row)
        output_df = pd.DataFrame(data, columns=columns)
        output_df.to_excel(writer, index=False, sheet_name='Constrained_Fits')
    output.seek(0)
    return output
