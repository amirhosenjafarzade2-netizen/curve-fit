import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, Akima1DInterpolator, PchipInterpolator, CubicHermiteSpline
import pandas as pd
import io

def parametric_ui():
    """
    Render UI for Parametric Fitting mode.
    Returns: Dictionary of parameters including sub_mode, n_points, and smoothness
    """
    params = {}
    sub_mode_options = ["Parametric Spline", "Path Interpolation", "Bezier Spline", 
                        "Catmull-Rom Spline", "Cubic Hermite Spline", "NURBS (Non-Uniform Rational B-Spline)",
                        "Akima Spline", "PCHIP (Piecewise Cubic Hermite)"]
    params['sub_mode'] = st.selectbox("Choose Parametric Sub-Mode", sub_mode_options)
    params['n_points'] = st.number_input("Number of smoothed points per line", min_value=10, max_value=1000, value=200, step=10)
    params['smoothness'] = st.slider("Smoothness (higher values reduce overfitting)", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                                    help="For Parametric/Bezier Spline: Controls smoothing (0 = exact fit, higher = smoother). For Path Interpolation: Affects point density indirectly.")
    
    return params

def generate_parametric_data(lines, params):
    """
    Generate smoothed parametric data for each line using the specified sub-mode.
    Args:
        lines: List of tuples (line_name, x, y, has_duplicates, has_invalid_x) - but ignores duplicates/invalid for parametric
        params: Parameters including sub_mode, n_points, and smoothness
    Returns: List of (line_name, x_smooth, y_smooth, error_message)
    """
    results = []
    n_points = int(params['n_points'])  # Ensure it's an integer
    smoothness = params['smoothness']

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
                u = np.linspace(0, 1, n)
                tck, _ = splprep([x, y], u=u, s=smoothness, k=3)  # Cubic spline with adjustable smoothness
                u_new = np.linspace(0, 1, n_points)
                x_smooth, y_smooth = splev(u_new, tck)
                results.append((line_name, x_smooth, y_smooth, None))
            
            elif sub_mode == "Catmull-Rom Spline":
                # Catmull-Rom spline: passes through all points with smooth tangents
                if n < 4:
                    results.append((line_name, None, None, "Catmull-Rom needs at least 4 points"))
                    continue
                u = np.linspace(0, 1, n)
                tck, _ = splprep([x, y], u=u, s=smoothness, k=min(3, n-1))
                u_new = np.linspace(0, 1, n_points)
                x_smooth, y_smooth = splev(u_new, tck)
                results.append((line_name, x_smooth, y_smooth, None))
            
            elif sub_mode == "Cubic Hermite Spline":
                # Cubic Hermite: interpolates with specified derivatives
                if n < 2:
                    results.append((line_name, None, None, "Need at least 2 points"))
                    continue
                t = np.linspace(0, 1, n)
                # Estimate derivatives using finite differences
                dx = np.gradient(x, t)
                dy = np.gradient(y, t)
                # Apply smoothness by dampening derivatives
                if smoothness > 0:
                    damping = 1.0 / (1.0 + smoothness * 0.5)
                    dx = dx * damping
                    dy = dy * damping
                t_new = np.linspace(0, 1, n_points)
                hermite_x = CubicHermiteSpline(t, x, dx)
                hermite_y = CubicHermiteSpline(t, y, dy)
                x_smooth = hermite_x(t_new)
                y_smooth = hermite_y(t_new)
                results.append((line_name, x_smooth, y_smooth, None))
            
            elif sub_mode == "NURBS (Non-Uniform Rational B-Spline)":
                # NURBS approximation using weighted B-spline
                if n < 4:
                    results.append((line_name, None, None, "NURBS needs at least 4 points"))
                    continue
                u = np.linspace(0, 1, n)
                weights = np.ones(n)  # Uniform weights (can be modified for emphasis)
                tck, _ = splprep([x, y], u=u, w=weights, s=smoothness, k=3)
                u_new = np.linspace(0, 1, n_points)
                x_smooth, y_smooth = splev(u_new, tck)
                results.append((line_name, x_smooth, y_smooth, None))
            
            elif sub_mode == "Akima Spline":
                # Akima spline: avoids overshooting, good for sharp curves
                if n < 3:
                    results.append((line_name, None, None, "Akima needs at least 3 points"))
                    continue
                # If smoothness > 0, pre-smooth the data before Akima interpolation
                if smoothness > 0:
                    t = np.linspace(0, 1, n)
                    tck_x, _ = splprep([t, x], s=smoothness, k=min(3, n-1))
                    tck_y, _ = splprep([t, y], s=smoothness, k=min(3, n-1))
                    t_smooth = np.linspace(0, 1, n)
                    _, x_pre = splev(t_smooth, tck_x)
                    _, y_pre = splev(t_smooth, tck_y)
                else:
                    t = np.linspace(0, 1, n)
                    x_pre, y_pre = x, y
                akima_x = Akima1DInterpolator(t, x_pre)
                akima_y = Akima1DInterpolator(t, y_pre)
                t_new = np.linspace(0, 1, n_points)
                x_smooth = akima_x(t_new)
                y_smooth = akima_y(t_new)
                results.append((line_name, x_smooth, y_smooth, None))
            
            elif sub_mode == "PCHIP (Piecewise Cubic Hermite)":
                # PCHIP: preserves monotonicity and shape, no overshooting
                if n < 2:
                    results.append((line_name, None, None, "PCHIP needs at least 2 points"))
                    continue
                # If smoothness > 0, pre-smooth the data before PCHIP interpolation
                if smoothness > 0:
                    t = np.linspace(0, 1, n)
                    tck_x, _ = splprep([t, x], s=smoothness, k=min(3, n-1))
                    tck_y, _ = splprep([t, y], s=smoothness, k=min(3, n-1))
                    t_smooth = np.linspace(0, 1, n)
                    _, x_pre = splev(t_smooth, tck_x)
                    _, y_pre = splev(t_smooth, tck_y)
                else:
                    t = np.linspace(0, 1, n)
                    x_pre, y_pre = x, y
                pchip_x = PchipInterpolator(t, x_pre)
                pchip_y = PchipInterpolator(t, y_pre)
                t_new = np.linspace(0, 1, n_points)
                x_smooth = pchip_x(t_new)
                y_smooth = pchip_y(t_new)
                results.append((line_name, x_smooth, y_smooth, None))
            
            elif sub_mode == "Path Interpolation":
                points = np.column_stack((x, y))
                segments = np.diff(points, axis=0)
                lengths = np.linalg.norm(segments, axis=1)
                cumlen = np.cumsum(lengths)
                total_len = cumlen[-1] if len(cumlen) > 0 else 0
                
                if total_len == 0:
                    results.append((line_name, None, None, "Zero path length"))
                    continue
                
                # Adjust n_points based on smoothness for Path Interpolation (higher smoothness reduces points)
                adjusted_n_points = max(10, int(n_points * (1 - smoothness / 10)))
                dist_new = np.linspace(0, total_len, adjusted_n_points)
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
            # Write actual data points from x_smooth and y_smooth arrays
            for i in range(len(x_smooth)):
                worksheet.write(row, 0, float(x_smooth[i]))
                worksheet.write(row, 1, float(y_smooth[i]))
                row += 1
            row += 1  # Empty row between lines
    output.seek(0)
    return output

def compare_parametric_modes(lines, n_points=200):
    """
    Compare all parametric sub-modes and display their smoothed curves for each line.
    Args:
        lines: List of tuples (line_name, x, y, has_duplicates, has_invalid_x)
        n_points: Number of smoothed points (default 200)
    """
    st.subheader("Parametric Visual Comparison")
    st.markdown("This mode compares all parametric sub-modes with default parameters for visual inspection.")
    
    # Define parametric methods with default params
    param_methods = [
        ("Parametric Spline", {"n_points": n_points, "smoothness": 0.0}),
        ("Path Interpolation", {"n_points": n_points, "smoothness": 0.0}),
        ("Bezier Spline", {"n_points": n_points, "smoothness": 0.0}),
        ("Catmull-Rom Spline", {"n_points": n_points, "smoothness": 0.0}),
        ("Cubic Hermite Spline", {"n_points": n_points, "smoothness": 0.0}),
        ("NURBS (Non-Uniform Rational B-Spline)", {"n_points": n_points, "smoothness": 0.0}),
        ("Akima Spline", {"n_points": n_points, "smoothness": 0.0}),
        ("PCHIP (Piecewise Cubic Hermite)", {"n_points": n_points, "smoothness": 0.0})
    ]
    
    results = {}
    for method, method_params in param_methods:
        method_params['sub_mode'] = method
        results[method] = generate_parametric_data(lines, method_params)
    
    for line_name, x, y, _, _ in lines:
        st.markdown(f"### Line: {line_name}")
        if len(x) < 2:
            st.warning(f"Line '{line_name}': Skipped due to insufficient points (need at least 2).")
            continue
        for method, method_params in param_methods:
            result = next((r for r in results[method] if r[0] == line_name), None)
            if result and result[3] is None:
                x_smooth, y_smooth = result[1], result[2]
                st.write(f"{method} Fit")
                fig = plot_parametric(x, y, x_smooth, y_smooth, method)
                st.pyplot(fig)
            else:
                st.warning(f"Line '{line_name}' - {method}: {result[3] if result else 'No result'}")
