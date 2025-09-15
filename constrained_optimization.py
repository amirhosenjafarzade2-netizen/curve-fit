# constrained_optimization.py - Standalone module for constrained coefficient optimization

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import streamlit as st
import io
import pandas as pd
import sympy as sp

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

    # Option to show detailed diagnostics
    params['show_diagnostics'] = st.checkbox("Show detailed diagnostics for optimization failures", value=False)

    return params

def optimize_coefficients(x, y, method, params):
    """
    Optimize coefficients under constraints with improved stability and error handling.
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

            # Check for sufficient data points
            min_points = degree + 1
            if len(x) < min_points:
                return None, None, f"Insufficient data points: need at least {min_points} for degree {degree} polynomial"

            # Check for reasonable number of constraints
            if len(constraints) > degree + 1:
                return None, None, f"Too many constraints ({len(constraints)}) for degree {degree} polynomial"

            # Scale data to improve numerical stability
            x_mean = np.mean(x)
            x_std = np.std(x) if np.std(x) != 0 else 1.0
            y_mean = np.mean(y)
            y_std = np.std(y) if np.std(y) != 0 else 1.0
            x_scaled = (x - x_mean) / x_std
            y_scaled = (y - y_mean) / y_std
            constraints_scaled = [((cons_x - x_mean) / x_std, (cons_y - y_mean) / y_std) for cons_x, cons_y in constraints]

            # Weighted objective function to emphasize points near restrictive points
            def objective(coeffs):
                y_pred = np.polyval(coeffs, x_scaled)
                # Base weights
                weights = np.ones_like(x_scaled)
                # Add Gaussian weighting for each restrictive point to emphasize nearby data points
                if constraints_scaled:
                    for cons_x, _ in constraints_scaled:
                        dist = (x_scaled - cons_x) ** 2
                        weights += 10.0 * np.exp(-dist / (2 * 0.1**2))  # Adjust sigma (0.1) and multiplier (10.0) as needed
                # Normalize weights to prevent scaling issues
                weights /= np.mean(weights)
                mse = np.average((y_scaled - y_pred)**2, weights=weights)
                reg = lambda_reg * np.sum(coeffs**2)  # L2 regularization
                return mse + reg

            # Constraints: Equality for each restrictive point
            cons = []
            for cons_x, cons_y in constraints_scaled:
                cons.append({'type': 'eq', 'fun': lambda coeffs, cx=cons_x, cy=cons_y: np.polyval(coeffs, cx) - cy})

            # Initial guess: Standard polyfit on scaled data
            initial_coeffs = np.polyfit(x_scaled, y_scaled, degree)

            # Optimize with bounds to prevent extreme coefficients
            bounds = [(-100, 100) for _ in range(degree + 1)]  # Reasonable bounds for scaled coefficients
            res = minimize(objective, initial_coeffs, method='SLSQP', constraints=cons, bounds=bounds, options={'disp': show_diagnostics, 'maxiter': 1000})

            if res.success:
                coeffs_scaled = res.x
                # Rescale coefficients back to original scale using sympy for accurate expansion
                x_var = sp.symbols('x')
                z = (x_var - x_mean) / x_std
                p_scaled = sum([coeffs_scaled[i] * z**(degree - i) for i in range(degree + 1)])
                p_original = p_scaled * y_std + y_mean
                p_expanded = sp.expand(p_original)
                coeffs = [float(p_expanded.coeff(x_var ** (degree - i))) if p_expanded.coeff(x_var ** (degree - i)) else 0.0 for i in range(degree + 1)]

                y_pred = np.polyval(coeffs, x)
                r2 = r2_score(y, y_pred)

                # Verify constraints are satisfied (within tighter tolerance)
                for cons_x, cons_y in constraints:
                    pred_y = np.polyval(coeffs, cons_x)
                    if abs(pred_y - cons_y) > 1e-6:  # Tightened tolerance
                        error_msg = f"Constraint not satisfied: at x={cons_x}, predicted y={pred_y:.6f} != {cons_y:.6f}"
                        if show_diagnostics:
                            error_msg += f"\nTolerance exceeded. Scaled coeffs={coeffs_scaled}"
                        return None, None, error_msg

                return coeffs, r2, f"Constrained Polynomial (degree {degree})"
            else:
                error_msg = f"Optimization failed: {res.message}"
                if show_diagnostics:
                    error_msg += f"\nDiagnostics: Initial coeffs={initial_coeffs}, Constraints={constraints_scaled}, Status={res.status}"
                return None, None, error_msg

        # Add other methods here if extended (e.g., Exponential with constraints)
        return None, None, f"Method {method} not supported for constrained optimization"

    except Exception as e:
        error_msg = f"Constrained optimization failed: {str(e)}"
        if params.get('show_diagnostics', False):
            error_msg += f"\nDiagnostics: x={x}, y={y}, params={params}"
        return None, None, error_msg

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
