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

def calculate_fit_metrics(x_orig, y_orig, x_smooth, y_smooth):
    """
    Calculate goodness of fit metrics for parametric curves.
    Returns: Dictionary with various fit quality metrics
    """
    from scipy.spatial.distance import cdist
    
    # Original points
    orig_points = np.column_stack((x_orig, y_orig))
    smooth_points = np.column_stack((x_smooth, y_smooth))
    
    # 1. Average perpendicular distance (main metric for parametric curves)
    distances = cdist(orig_points, smooth_points, metric='euclidean')
    min_distances = np.min(distances, axis=1)
    avg_distance = np.mean(min_distances)
    max_distance = np.max(min_distances)
    
    # 2. Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean(min_distances**2))
    
    # 3. Normalized RMSE (as percentage of data range)
    data_range = np.sqrt((np.max(x_orig) - np.min(x_orig))**2 + (np.max(y_orig) - np.min(y_orig))**2)
    nrmse = (rmse / data_range * 100) if data_range > 0 else 0
    
    # 4. R² analog for parametric curves (based on distance variance)
    total_variance = np.var(min_distances)
    ss_tot = np.sum((min_distances - np.mean(min_distances))**2)
    ss_res = np.sum(min_distances**2)
    r2_analog = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 0 else 0
    
    # 5. Fit Quality Score (0-100, higher is better)
    # Based on normalized distance: 100 = perfect fit, decreases with distance
    fit_score = max(0, 100 * (1 - nrmse / 100))
    
    return {
        'avg_distance': avg_distance,
        'max_distance': max_distance,
        'rmse': rmse,
        'nrmse': nrmse,
        'r2_analog': r2_analog,
        'fit_score': fit_score
    }

def generate_parametric_data(lines, params, include_metrics=True):
    """
    Generate smoothed parametric data for each line using the specified sub-mode.
    Args:
        lines: List of tuples (line_name, x, y, has_duplicates, has_invalid_x) - but ignores duplicates/invalid for parametric
        params: Parameters including sub_mode, n_points, and smoothness
        include_metrics: If True, returns fit_metrics; if False, returns None for backward compatibility
    Returns: List of (line_name, x_smooth, y_smooth, error_message, fit_metrics) if include_metrics=True
             List of (line_name, x_smooth, y_smooth, error_message) if include_metrics=False
    """
    results = []
    n_points = int(params['n_points'])  # Ensure it's an integer
    smoothness = params['smoothness']

    for line_name, x, y, _, _ in lines:
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        
        if n < 2:
            if include_metrics:
                results.append((line_name, None, None, "Insufficient points (need at least 2)", None))
            else:
                results.append((line_name, None, None, "Insufficient points (need at least 2)"))
            continue
        
        try:
            sub_mode = params['sub_mode']
            
            if sub_mode == "Parametric Spline" or sub_mode == "Bezier Spline":  # Bezier-like using B-spline
                u = np.linspace(0, 1, n)
                tck, _ = splprep([x, y], u=u, s=smoothness, k=3)  # Cubic spline with adjustable smoothness
                u_new = np.linspace(0, 1, n_points)
                x_smooth, y_smooth = splev(u_new, tck)
                if include_metrics:
                    fit_metrics = calculate_fit_metrics(x, y, x_smooth, y_smooth)
                    results.append((line_name, x_smooth, y_smooth, None, fit_metrics))
                else:
                    results.append((line_name, x_smooth, y_smooth, None))
            
            elif sub_mode == "Catmull-Rom Spline":
                # Catmull-Rom spline: passes through all points with smooth tangents
                if n < 4:
                    if include_metrics:
                        results.append((line_name, None, None, "Catmull-Rom needs at least 4 points", None))
                    else:
                        results.append((line_name, None, None, "Catmull-Rom needs at least 4 points"))
                    continue
                u = np.linspace(0, 1, n)
                tck, _ = splprep([x, y], u=u, s=smoothness, k=min(3, n-1))
                u_new = np.linspace(0, 1, n_points)
                x_smooth, y_smooth = splev(u_new, tck)
                if include_metrics:
                    fit_metrics = calculate_fit_metrics(x, y, x_smooth, y_smooth)
                    results.append((line_name, x_smooth, y_smooth, None, fit_metrics))
                else:
                    results.append((line_name, x_smooth, y_smooth, None))
            
            elif sub_mode == "Cubic Hermite Spline":
                # Cubic Hermite: interpolates with specified derivatives
                if n < 2:
                    if include_metrics:
                        results.append((line_name, None, None, "Need at least 2 points", None))
                    else:
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
                if include_metrics:
                    fit_metrics = calculate_fit_metrics(x, y, x_smooth, y_smooth)
                    results.append((line_name, x_smooth, y_smooth, None, fit_metrics))
                else:
                    results.append((line_name, x_smooth, y_smooth, None))
            
            elif sub_mode == "NURBS (Non-Uniform Rational B-Spline)":
                # NURBS approximation using weighted B-spline
                if n < 4:
                    if include_metrics:
                        results.append((line_name, None, None, "NURBS needs at least 4 points", None))
                    else:
                        results.append((line_name, None, None, "NURBS needs at least 4 points"))
                    continue
                u = np.linspace(0, 1, n)
                weights = np.ones(n)  # Uniform weights (can be modified for emphasis)
                tck, _ = splprep([x, y], u=u, w=weights, s=smoothness, k=3)
                u_new = np.linspace(0, 1, n_points)
                x_smooth, y_smooth = splev(u_new, tck)
                if include_metrics:
                    fit_metrics = calculate_fit_metrics(x, y, x_smooth, y_smooth)
                    results.append((line_name, x_smooth, y_smooth, None, fit_metrics))
                else:
                    results.append((line_name, x_smooth, y_smooth, None))
            
            elif sub_mode == "Akima Spline":
                # Akima spline: avoids overshooting, good for sharp curves
                if n < 3:
                    if include_metrics:
                        results.append((line_name, None, None, "Akima needs at least 3 points", None))
                    else:
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
                if include_metrics:
                    fit_metrics = calculate_fit_metrics(x, y, x_smooth, y_smooth)
                    results.append((line_name, x_smooth, y_smooth, None, fit_metrics))
                else:
                    results.append((line_name, x_smooth, y_smooth, None))
            
            elif sub_mode == "PCHIP (Piecewise Cubic Hermite)":
                # PCHIP: preserves monotonicity and shape, no overshooting
                if n < 2:
                    if include_metrics:
                        results.append((line_name, None, None, "PCHIP needs at least 2 points", None))
                    else:
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
                if include_metrics:
                    fit_metrics = calculate_fit_metrics(x, y, x_smooth, y_smooth)
                    results.append((line_name, x_smooth, y_smooth, None, fit_metrics))
                else:
                    results.append((line_name, x_smooth, y_smooth, None))
            
            elif sub_mode == "Path Interpolation":
                points = np.column_stack((x, y))
                segments = np.diff(points, axis=0)
                lengths = np.linalg.norm(segments, axis=1)
                cumlen = np.cumsum(lengths)
                total_len = cumlen[-1] if len(cumlen) > 0 else 0
                
                if total_len == 0:
                    if include_metrics:
                        results.append((line_name, None, None, "Zero path length", None))
                    else:
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
                
                if include_metrics:
                    fit_metrics = calculate_fit_metrics(x, y, np.array(x_smooth), np.array(y_smooth))
                    results.append((line_name, np.array(x_smooth), np.array(y_smooth), None, fit_metrics))
                else:
                    results.append((line_name, np.array(x_smooth), np.array(y_smooth), None))
        
        except Exception as e:
            if include_metrics:
                results.append((line_name, None, None, f"Failed to generate parametric data: {str(e)}", None))
            else:
                results.append((line_name, None, None, f"Failed to generate parametric data: {str(e)}"))
    
    return results

def plot_parametric(x, y, x_smooth, y_smooth, sub_mode, fit_metrics=None):
    """
    Plot original points and smoothed parametric curve.
    Returns: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    if x_smooth is not None and y_smooth is not None:
        ax.plot(x_smooth, y_smooth, color='red', label=f'Smoothed {sub_mode}', linewidth=1.5, zorder=5)
    ax.scatter(x, y, color='blue', label='Original Points', s=50, zorder=3, alpha=1.0)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add fit metrics as text box
    if fit_metrics is not None:
        textstr = f"Fit Quality Metrics:\n"
        textstr += f"Fit Score: {fit_metrics['fit_score']:.2f}/100\n"
        textstr += f"Avg Distance: {fit_metrics['avg_distance']:.4f}\n"
        textstr += f"Max Distance: {fit_metrics['max_distance']:.4f}\n"
        textstr += f"RMSE: {fit_metrics['rmse']:.4f}\n"
        textstr += f"NRMSE: {fit_metrics['nrmse']:.2f}%\n"
        textstr += f"R² Analog: {fit_metrics['r2_analog']:.4f}"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    return fig

def create_parametric_excel(results):
    """
    Create Excel file with parametric/smoothed data in the same format as input.
    Args:
        results: List of (line_name, x_smooth, y_smooth, error_message, fit_metrics)
    Returns: BytesIO object with Excel data
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet('Parametric_Data')
        metrics_worksheet = workbook.add_worksheet('Fit_Metrics')
        
        # Write smoothed data
        row = 0
        for line_name, x_smooth, y_smooth, error_message, fit_metrics in results:
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
        
        # Write fit metrics
        metrics_row = 0
        metrics_worksheet.write(metrics_row, 0, "Line Name")
        metrics_worksheet.write(metrics_row, 1, "Fit Score (0-100)")
        metrics_worksheet.write(metrics_row, 2, "Avg Distance")
        metrics_worksheet.write(metrics_row, 3, "Max Distance")
        metrics_worksheet.write(metrics_row, 4, "RMSE")
        metrics_worksheet.write(metrics_row, 5, "NRMSE (%)")
        metrics_worksheet.write(metrics_row, 6, "R² Analog")
        metrics_row += 1
        
        for line_name, x_smooth, y_smooth, error_message, fit_metrics in results:
            if error_message or fit_metrics is None:
                continue
            metrics_worksheet.write(metrics_row, 0, line_name)
            metrics_worksheet.write(metrics_row, 1, fit_metrics['fit_score'])
            metrics_worksheet.write(metrics_row, 2, fit_metrics['avg_distance'])
            metrics_worksheet.write(metrics_row, 3, fit_metrics['max_distance'])
            metrics_worksheet.write(metrics_row, 4, fit_metrics['rmse'])
            metrics_worksheet.write(metrics_row, 5, fit_metrics['nrmse'])
            metrics_worksheet.write(metrics_row, 6, fit_metrics['r2_analog'])
            metrics_row += 1
            
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
    
    # Create comparison table
    comparison_data = []
    for method, method_params in param_methods:
        for result in results[method]:
            line_name, x_smooth, y_smooth, error_message, fit_metrics = result
            if fit_metrics:
                comparison_data.append({
                    'Line': line_name,
                    'Method': method,
                    'Fit Score': f"{fit_metrics['fit_score']:.2f}",
                    'Avg Distance': f"{fit_metrics['avg_distance']:.4f}",
                    'RMSE': f"{fit_metrics['rmse']:.4f}",
                    'NRMSE (%)': f"{fit_metrics['nrmse']:.2f}",
                    'R² Analog': f"{fit_metrics['r2_analog']:.4f}"
                })
    
    if comparison_data:
        st.subheader("Fit Quality Comparison Table")
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    for line_name, x, y, _, _ in lines:
        st.markdown(f"### Line: {line_name}")
        if len(x) < 2:
            st.warning(f"Line '{line_name}': Skipped due to insufficient points (need at least 2).")
            continue
        for method, method_params in param_methods:
            result = next((r for r in results[method] if r[0] == line_name), None)
            if result and result[3] is None:
                x_smooth, y_smooth, fit_metrics = result[1], result[2], result[4]
                st.write(f"{method} Fit")
                fig = plot_parametric(x, y, x_smooth, y_smooth, method, fit_metrics)
                st.pyplot(fig)
            else:
                st.warning(f"Line '{line_name}' - {method}: {result[3] if result else 'No result'}")
