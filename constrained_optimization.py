# constrained_optimization.py - Standalone module for constrained coefficient optimization

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import streamlit as st
import io
import pandas as pd

def constrained_optimization_ui():
    """
    Render UI for Constrained Coefficient Optimization mode.
    Returns: Dictionary of parameters including constraints
    """
    params = {}
    st.markdown("Specify the fitting method, parameters, and restrictive points (x, y) that the curve must pass through. Multiple points can be added.")

    # Method selection (limited to methods that can be constrained, e.g., Polynomial)
    method_options = ["Polynomial"]  # Extendable to Exponential, etc., in future
    params['method'] = st.selectbox("Choose Fitting Method", method_options)

    if params['method'] == "Polynomial":
        params['degree'] = st.number_input("Polynomial Degree", min_value=1, max_value=10, value=5)

    # Input for restrictive points
    num_points = st.number_input("Number of restrictive points", min_value=0, max_value=10, value=1)
    params['constraints'] = []
    for i in range(num_points):
        col1, col2 = st.columns(2)
        with col1:
            x_val = st.number_input(f"Restrictive Point {i+1}: x", value=0.0, key=f"x_{i}")
        with col2:
            y_val = st.number_input(f"Restrictive Point {i+1}: y", value=0.0, key=f"y_{i}")
        params['constraints'].append((x_val, y_val))

    # Regularization strength to preserve trajectory (optional)
    params['lambda_reg'] = st.slider("Regularization Strength (to preserve trajectory)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    return params

def optimize_coefficients(x, y, method, params):
    """
    Optimize coefficients under constraints.
    Args:
        x, y: Data points
        method: Selected method (e.g., "Polynomial")
        params: Parameters including degree, constraints, lambda_reg
    Returns: coeffs, r2, model_desc or None, None, error_message
    """
    try:
        if method == "Polynomial":
            degree = params['degree']
            constraints = params['constraints']
            lambda_reg = params['lambda_reg']

            # Objective function: MSE + regularization
            def objective(coeffs):
                y_pred = np.polyval(coeffs, x)
                mse = np.mean((y - y_pred)**2)
                reg = lambda_reg * np.sum(coeffs**2)  # L2 regularization to preserve trajectory
                return mse + reg

            # Constraints: Equality for each restrictive point
            cons = []
            for cons_x, cons_y in constraints:
                cons.append({'type': 'eq', 'fun': lambda coeffs: np.polyval(coeffs, cons_x) - cons_y})

            # Initial guess: Standard polyfit without constraints
            initial_coeffs = np.polyfit(x, y, degree)

            # Optimize
            res = minimize(objective, initial_coeffs, method='SLSQP', constraints=cons)
            if res.success:
                coeffs = res.x
                y_pred = np.polyval(coeffs, x)
                r2 = r2_score(y, y_pred)
                return coeffs.tolist(), r2, f"Constrained Polynomial (degree {degree})"
            else:
                return None, None, "Optimization failed"

        # Add other methods here if extended (e.g., Exponential with constraints)

    except Exception as e:
        return None, None, f"Constrained optimization failed: {str(e)}"

def plot_constrained_fit(x, y, coeffs, method, params):
    """
    Plot the constrained fit.
    Returns: Matplotlib figure
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data Points', s=50)

    x_smooth = np.linspace(min(x), max(x), 200)

    if method == "Polynomial":
        y_smooth = np.polyval(coeffs, x_smooth)
        ax.plot(x_smooth, y_smooth, color='red', label=f'Constrained Polynomial (deg {params.get("degree", "?")})')

    # Plot restrictive points
    for cons_x, cons_y in params.get('constraints', []):
        ax.scatter(cons_x, cons_y, color='green', label='Restrictive Point', s=100, marker='x')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
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
