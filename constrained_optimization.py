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

def constrained_optimization_ui(x, y, line_name=None):
    """
    Render UI for Constrained Coefficient Optimization mode.
    Args:
        x, y: Data points (numpy arrays) for auto-generating constraints
        line_name: Optional string to create unique widget keys per line
    Returns: Dictionary of parameters including constraints
    """
    params = {}
    st.markdown("Specify the fitting method, parameters, and restrictive points (x, y) that the curve must pass through. Multiple points can be added.")

    # Use line_name to create unique key for selectbox
    key_suffix = f"_{line_name}" if line_name else ""
    method_options = ["Polynomial"]
    params['method'] = st.selectbox("Choose Fitting Method", method_options, key=f"method_select{key_suffix}")

    if params['method'] == "Polynomial":
        params['degree'] = st.number_input("Polynomial Degree", min_value=1, max_value=10, value=3, key=f"degree{key_suffix}")

    # Input for manual restrictive points
    num_points = st.number_input("Number of manual restrictive points", min_value=0, max_value=10, value=1, key=f"num_points{key_suffix}")
    params['constraints'] = []
    for i in range(num_points):
        col1, col2 = st.columns(2)
        with col1:
            x_val = st.number_input(f"Restrictive Point {i+1}: x", value=0.0, key=f"x_{i}{key_suffix}")
        with col2:
            y_val = st.number_input(f"Restrictive Point {i+1}: y", value=0.0, key=f"y_{i}{key_suffix}")
        params['constraints'].append((x_val, y_val))

    # If only 1 manual constraint, offer auto-add options
    if num_points == 1 and len(x) > 1:
        # Sort data by x for consistent range selection
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        st.markdown("### Auto-Add Constraints (since only 1 manual point specified)")
        
        add_last_point = st.checkbox("Add the last data point as a constraint?", value=True, key=f"add_last{key_suffix}")
        if add_last_point:
            last_x, last_y = x_sorted[-1], y_sorted[-1]
            if (last_x, last_y) not in params['constraints']:
                params['constraints'].append((last_x, last_y))
                st.info(f"Added last point: ({last_x:.2f}, {last_y:.2f})")

        add_random_points = st.checkbox("Add random constraint points from data?", value=False, key=f"add_random{key_suffix}")
        if add_random_points:
            n_random = st.number_input("Number of random points (n)", min_value=1, max_value=5, value=2, key=f"n_random{key_suffix}")
            min_percent, max_percent = st.slider("Select range (%) of data for random points", 0, 100, (40, 90), step=5, key=f"range_slider{key_suffix}")
            
            # Calculate index range
            num_data = len(x_sorted)
            start_idx = int(num_data * (min_percent / 100))
            end_idx = int(num_data * (max_percent / 100))
            if start_idx >= end_idx:
                st.warning("Invalid range; using full data.")
                start_idx, end_idx = 0, num_data
            
            # Select random indices from range, get points, avoid duplicates
            candidate_indices = np.arange(start_idx, end_idx)
            if len(candidate_indices) < n_random:
                st.warning(f"Not enough points in range; using {len(candidate_indices)} points.")
                n_random = len(candidate_indices)
            
            random_indices = np.random.choice(candidate_indices, n_random, replace=False)
            for idx in random_indices:
                rand_x, rand_y = x_sorted[idx], y_sorted[idx]
                if (rand_x, rand_y) not in params['constraints']:
                    params['constraints'].append((rand_x, rand_y))
                    st.info(f"Added random point: ({rand_x:.2f}, {rand_y:.2f}) from {min_percent}%-{max_percent}% range")

    params['lambda_reg'] = st.slider("Regularization Strength", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key=f"lambda_reg{key_suffix}")
    params['show_diagnostics'] = st.checkbox("Show detailed diagnostics", value=False, key=f"diagnostics{key_suffix}")

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
            x_scaled = (x - x_med) / (x_iqr / 1.349)
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
                        weights += 20.0 * np.exp(-dist / (2 * 0.2**2))  # Wider Gaussian
                weights /= np.mean(weights)
                kernel = np.exp(-((x_scaled[:, None] - x_scaled[None, :]) ** 2) / (2 * 0.1**2))
                density = np.sum(kernel, axis=1)
                weights *= density / np.mean(density)
                return weights

            weights = compute_weights(x_scaled, constraints_scaled)

            # Objective function with weighted MSE
            def objective(coeffs):
                y_pred = np.polyval(coeffs, x_scaled)
                mse = np.average((y_scaled - y_pred) ** 2, weights=weights)
                reg = lambda_reg * np.sum(coeffs ** 2)
                return mse + reg

            # Constraints with high precision
            cons = [{'type': 'eq', 'fun': lambda coeffs, cx=cons_x, cy=cons_y: np.polyval(coeffs, cx) - cy} for cons_x, cons_y in constraints_scaled]

            # Optimize with tighter bounds and more iterations
            bounds = [(-50, 50) for _ in range(degree + 1)]
            res = minimize(objective, initial_coeffs, method='SLSQP', constraints=cons, bounds=bounds, options={'disp': show_diagnostics, 'maxiter': 2000, 'ftol': 1e-8})

            if res.success:
                coeffs_scaled = res.x
                x_var = sp.symbols('x')
                z = (x_var - x_med) / (x_iqr / 1.349)
                p_scaled = sum([coeffs_scaled[i] * z**(degree - i) for i in range(degree + 1)])
                p_original = p_scaled * (y_iqr / 1.349) + y_med
                p_expanded = sp.expand(p_original)
                coeffs = [float(p_expanded.coeff(x_var ** (degree - i))) if p_expanded.coeff(x_var ** (degree - i)) else 0.0 for i in range(degree + 1)]

                y_pred = np.polyval(coeffs, x)
                r2 = r2_score(y, y_pred)

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
