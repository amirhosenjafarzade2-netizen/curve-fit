# parametric_modes.py - Module for parametric and path-based fitting modes

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import pandas as pd
import io

def parametric_ui():
    """
    Render UI for Parametric Fitting mode.
    Returns: Dictionary of parameters including sub_mode and n_points
    """
    params = {}
    sub_mode_options = ["Parametric Spline", "Path Interpolation", "Bezier Spline", "Gaussian Process"]
    params['sub_mode'] = st.selectbox("Choose Parametric Sub-Mode", sub_mode_options)
    params['n_points'] = st.number_input("Number of smoothed points per line", min_value=10, max_value=1000, value=200, step=10)
    
    if params['sub_mode'] == "Gaussian Process":
        params['length_scale'] = st.slider("RBF Length Scale", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    
    return params

def generate_parametric_data(lines, params):
    """
    Generate smoothed parametric data for each line using the specified sub-mode.
    Args:
        lines: List of tuples (line_name, x, y, has_duplicates, has_invalid_x) - but ignores duplicates/invalid for parametric
        params: Parameters including sub_mode and n_points
    Returns: List of (line_name, x_smooth, y_smooth, error_message)
    """
    results = []
    n_points = params['n_points']

    for line_name, x, y, _, _ in lines:
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        
        if n < 2:
            results.append((line_name, None, None, "Insufficient points (need at least 2)"))
            continue
        
        try:
            sub_mode = params['sub_mode']
            
            if sub_mode == "Parametric Spline" or sub_mode == "Bezier Spline":  # Bezier-like using B-spline
                # Parameterize with normalized u
                u = np.linspace(0, 1, n)
                tck, _ = splprep([x, y], u=u, s=0, k=3)  # Cubic spline interpolation
                u_new = np.linspace(0, 1, n_points)
                x_smooth, y_smooth = splev(u_new, tck)
                results.append((line_name, x_smooth, y_smooth, None))
            
            elif sub_mode == "Path Interpolation":
                # Arc-length parameterization
                points = np.column_stack((x, y))
                segments = np.diff(points, axis=0)
                lengths = np.linalg.norm(segments, axis=1)
                cumlen = np.cumsum(lengths)
                total_len = cumlen[-1] if len(cumlen) > 0 else 0
                
                if total_len == 0:
                    results.append((line_name, None, None, "Zero path length"))
                    continue
                
                dist_new = np.linspace(0, total_len, n_points)
                cumlen_with0 = np.insert(cumlen, 0, 0)
                x_smooth = []
                y_smooth = []
                
                for d in dist_new:
                    i = np.searchsorted(cumlen_with0, d)
                    if i == 0:
                        p = points[0]
                    elif i > len(points) - 1:
                        p = points[-1]
                    else:
                        prev_cum = cumlen_with0[i - 1]
                        ratio = (d - prev_cum) / lengths[i - 1]
                        p = points[i - 1] + ratio * (points[i] - points[i - 1])
                    x_smooth.append(p[0])
                    y_smooth.append(p[1])
                
                results.append((line_name, np.array(x_smooth), np.array(y_smooth), None))
            
            elif sub_mode == "Gaussian Process":
                # Fit separate GPs for x(t) and y(t)
                t = np.linspace(0, 1, n).reshape(-1, 1)
                length_scale = params.get('length_scale', 1.0)
                kernel = ConstantKernel() * RBF(length_scale=length_scale)
                
                gp_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
                gp_x.fit(t, x)
                
                gp_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
                gp_y.fit(t, y)
                
                t_new = np.linspace(0, 1, n_points).reshape(-1, 1)
                x_smooth = gp_x.predict(t_new)
                y_smooth = gp_y.predict(t_new)
                
                results.append((line_name, x_smooth, y_smooth, None))
        
        except Exception as e:
            results.append((line_name, None, None, f"Failed to generate parametric data: {str(e)}"))
    
    return results

def plot_parametric(x, y, x_smooth, y_smooth, sub_mode):
    """
    Plot original points and smoothed parametric curve.
    Returns: Matplotlib figure
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Original Points', s=50)
    if x_smooth is not None and y_smooth is not None:
        ax.plot(x_smooth, y_smooth, color='red', label=f'Smoothed {sub_mode}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)
    return fig

def create_parametric_excel(results):
    """
    Create Excel file with parametric/smoothed data in the same format as input.
    Args:
        results: List of (line_name, x_smooth, y_smooth, error_message)
    Returns: BytesIO object with Excel data
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet('Parametric_Data')
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
