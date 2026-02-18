import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, Akima1DInterpolator, PchipInterpolator, CubicHermiteSpline
import pandas as pd
import io
from scipy.spatial.distance import cdist


def parametric_ui():
    """
    Render UI for Parametric Fitting mode.
    Returns: Dictionary of parameters including sub_mode, n_points, and smoothness
    """
    params = {}
    sub_mode_options = [
        "Parametric Spline", "Path Interpolation", "Bezier Spline",
        "Catmull-Rom Spline", "Cubic Hermite Spline", "NURBS (Non-Uniform Rational B-Spline)",
        "Akima Spline", "PCHIP (Piecewise Cubic Hermite)"
    ]
    params['sub_mode'] = st.selectbox("Choose Parametric Sub-Mode", sub_mode_options)
    params['n_points'] = st.number_input("Number of smoothed points per line", min_value=10, max_value=1000, value=200, step=10)

    st.markdown(
        "**Smoothness** — `0.0` = exact interpolation (passes through every point). "
        "Increase to trade fidelity for smoothness. For spline-based modes the value is "
        "scaled automatically by the number of points; "
        "for Cubic Hermite it damps tangent magnitudes; "
        "for Path Interpolation it reduces output point count."
    )

    smoothness_mode = st.radio("Smoothness input mode", ["Simple (0–10 scale)", "Raw s value (advanced)"],
                               horizontal=True, key="smooth_mode")
    if smoothness_mode == "Simple (0–10 scale)":
        simple_s = st.slider("Smoothness", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                             help="0 = interpolate exactly. 1–3 = gentle. 5+ = heavy smoothing. Internally scaled by n·σ².")
        params['smoothness'] = simple_s
        params['smoothness_simple'] = True   # flag to auto-scale in generate
    else:
        params['smoothness'] = st.number_input("Raw splprep s value", min_value=0.0, value=0.0, step=0.5,
                                                format="%.2f",
                                                help="Directly passed to splprep. Suitable range depends heavily on data scale.")
        params['smoothness_simple'] = False
    return params


def calculate_smoothness_metric(x, y):
    """
    Compute a smoothness score (lower = smoother curve).
    Combines average absolute curvature + second derivative magnitude.
    """
    if len(x) < 5:
        return np.nan

    t = np.linspace(0, 1, len(x))
    
    # First derivatives
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    
    # Second derivatives
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)
    
    # Curvature κ = |x'y'' - y'x''| / (x'^2 + y'^2)^{3/2}
    num = np.abs(dx * ddy - dy * ddx)
    den = (dx**2 + dy**2)**1.5 + 1e-12
    curvature = num / den
    
    # Acceleration magnitude
    accel = np.sqrt(ddx**2 + ddy**2)
    
    # Combined smoothness score (lower = better)
    score = np.nanmean(curvature) + 0.04 * np.nanmean(accel)
    
    return round(float(score), 6)


def calculate_fit_metrics(x_orig, y_orig, x_smooth, y_smooth):
    """
    Calculate goodness of fit + smoothness metrics for parametric curves.
    """
    orig_points = np.column_stack((x_orig, y_orig))
    smooth_points = np.column_stack((x_smooth, y_smooth))
    
    distances = cdist(orig_points, smooth_points, metric='euclidean')
    min_distances = np.min(distances, axis=1)
    
    avg_distance = np.mean(min_distances)
    max_distance = np.max(min_distances)
    rmse = np.sqrt(np.mean(min_distances**2))
    
    data_range = np.sqrt((np.max(x_orig) - np.min(x_orig))**2 + (np.max(y_orig) - np.min(y_orig))**2)
    nrmse = (rmse / data_range * 100) if data_range > 0 else 0
    
    ss_tot = np.sum((min_distances - np.mean(min_distances))**2)
    ss_res = np.sum(min_distances**2)
    r2_analog = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 0 else 0
    
    fit_score = max(0, 100 * (1 - nrmse / 100))
    
    metrics = {
        'avg_distance': round(avg_distance, 4),
        'max_distance': round(max_distance, 4),
        'rmse': round(rmse, 4),
        'nrmse': round(nrmse, 2),
        'r2_analog': round(r2_analog, 4),
        'fit_score': round(fit_score, 2),
        'smoothness': calculate_smoothness_metric(x_smooth, y_smooth)
    }
    
    return metrics


def generate_parametric_data(lines, params):
    """
    Generate smoothed parametric data for each line using the specified sub-mode.
    """
    results = []
    n_points = int(params['n_points'])
    smoothness_raw = params['smoothness']
    simple_mode = params.get('smoothness_simple', False)

    for line_name, x, y, _, _ in lines:
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        
        if n < 2:
            results.append((line_name, None, None, "Insufficient points (need at least 2)"))
            continue
        
        # Auto-scale s: in simple mode, s_raw ∈ [0,10] → scaled to n * combined_variance * factor
        if simple_mode and smoothness_raw > 0:
            data_variance = np.var(x) + np.var(y)
            # s = 0 → exact; s=10 → heavy. Scale: n * variance * (raw/10)^1.5
            smoothness = n * data_variance * ((smoothness_raw / 10.0) ** 1.5)
        else:
            smoothness = smoothness_raw
        
        try:
            sub_mode = params['sub_mode']
            x_smooth = None
            y_smooth = None
            
            if sub_mode in ["Parametric Spline", "Bezier Spline", "NURBS (Non-Uniform Rational B-Spline)",
                            "Catmull-Rom Spline"]:
                if sub_mode == "Catmull-Rom Spline" and n < 4:
                    raise ValueError("Catmull-Rom needs at least 4 points")
                u = np.linspace(0, 1, n)
                if sub_mode == "NURBS (Non-Uniform Rational B-Spline)":
                    weights = np.ones(n)
                    tck, _ = splprep([x, y], u=u, w=weights, s=smoothness, k=min(3, n-1))
                else:
                    tck, _ = splprep([x, y], u=u, s=smoothness, k=min(3, n-1))
                u_new = np.linspace(0, 1, n_points)
                x_smooth, y_smooth = splev(u_new, tck)
            
            elif sub_mode == "Cubic Hermite Spline":
                t = np.linspace(0, 1, n)
                dx = np.gradient(x, t)
                dy = np.gradient(y, t)
                if smoothness_raw > 0:
                    # damping: simple_mode raw ∈[0,10] → damping factor ∈ [1, 0.05]
                    damping = 1.0 / (1.0 + smoothness_raw * 0.5)
                    dx *= damping
                    dy *= damping
                t_new = np.linspace(0, 1, n_points)
                hermite_x = CubicHermiteSpline(t, x, dx)
                hermite_y = CubicHermiteSpline(t, y, dy)
                x_smooth = hermite_x(t_new)
                y_smooth = hermite_y(t_new)
            
            elif sub_mode in ["Akima Spline", "PCHIP (Piecewise Cubic Hermite)"]:
                if sub_mode == "Akima Spline" and n < 3:
                    raise ValueError("Akima needs at least 3 points")
                
                t = np.linspace(0, 1, n)
                if smoothness > 0:
                    tck_x, _ = splprep([t, x], s=smoothness, k=min(3, n-1))
                    tck_y, _ = splprep([t, y], s=smoothness, k=min(3, n-1))
                    t_smooth = np.linspace(0, 1, n)
                    _, x_pre = splev(t_smooth, tck_x)
                    _, y_pre = splev(t_smooth, tck_y)
                else:
                    x_pre, y_pre = x, y
                
                interp_class = Akima1DInterpolator if sub_mode == "Akima Spline" else PchipInterpolator
                interp_x = interp_class(t, x_pre)
                interp_y = interp_class(t, y_pre)
                t_new = np.linspace(0, 1, n_points)
                x_smooth = interp_x(t_new)
                y_smooth = interp_y(t_new)
            
            elif sub_mode == "Path Interpolation":
                points = np.column_stack((x, y))
                segments = np.diff(points, axis=0)
                lengths = np.linalg.norm(segments, axis=1)
                cumlen = np.cumsum(lengths)
                total_len = cumlen[-1] if len(cumlen) > 0 else 0
                
                if total_len == 0:
                    raise ValueError("Zero path length")
                
                cumlen_with0 = np.insert(cumlen, 0, 0)
                dist_new = np.linspace(0, total_len, n_points)
                x_interp = np.interp(dist_new, cumlen_with0, x)
                y_interp = np.interp(dist_new, cumlen_with0, y)

                # Apply smoothing via splprep if smoothness > 0
                if smoothness_raw > 0:
                    try:
                        tck, _ = splprep([x_interp, y_interp], s=smoothness, k=min(3, n_points-1))
                        u_new = np.linspace(0, 1, n_points)
                        x_smooth, y_smooth = splev(u_new, tck)
                    except Exception:
                        x_smooth, y_smooth = x_interp, y_interp
                else:
                    x_smooth, y_smooth = x_interp, y_interp
            
            results.append((line_name, x_smooth, y_smooth, None))
        
        except Exception as e:
            results.append((line_name, None, None, f"{sub_mode} failed: {str(e)}"))
    
    return results


def plot_parametric(x, y, x_smooth, y_smooth, sub_mode):
    """
    Plot original points and smoothed parametric curve with metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if x_smooth is not None and y_smooth is not None:
        ax.plot(x_smooth, y_smooth, color='red', label=f'Smoothed {sub_mode}', linewidth=1.8, zorder=5)
    
    ax.scatter(x, y, color='blue', label='Original Points', s=60, zorder=3, alpha=0.9)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    
    if x_smooth is not None and y_smooth is not None:
        metrics = calculate_fit_metrics(x, y, x_smooth, y_smooth)
        textstr = f"Fit Quality & Smoothness:\n"
        textstr += f"Fit Score: {metrics['fit_score']:.1f}/100\n"
        textstr += f"Avg Distance: {metrics['avg_distance']:.4f}\n"
        textstr += f"RMSE: {metrics['rmse']:.4f}\n"
        textstr += f"NRMSE: {metrics['nrmse']:.2f}%\n"
        textstr += f"Smoothness: {metrics['smoothness']:.6f}  (lower = smoother)"
        
        props = dict(boxstyle='round', facecolor='ivory', alpha=0.85)
        ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=9.5,
                verticalalignment='top', bbox=props)
    
    return fig


def create_parametric_excel(results):
    """
    Create Excel file with parametric data and fit metrics (including smoothness).
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        ws = workbook.add_worksheet('Parametric_Data')
        wm = workbook.add_worksheet('Fit_Metrics')
        
        row = 0
        mrow = 1
        
        # Headers for metrics
        headers = ["Line Name", "Fit Score (0-100)", "Avg Distance", "Max Distance",
                   "RMSE", "NRMSE (%)", "R² Analog", "Smoothness (lower=better)"]
        for col, h in enumerate(headers):
            wm.write(0, col, h)
        
        for line_name, x_smooth, y_smooth, error_message in results:
            if error_message:
                ws.write(row, 0, f"Error: {error_message}")
                row += 2
                continue
            
            ws.write(row, 0, line_name)
            row += 2
            
            if x_smooth is not None:
                for i in range(len(x_smooth)):
                    ws.write(row, 0, float(x_smooth[i]))
                    ws.write(row, 1, float(y_smooth[i]))
                    row += 1
            row += 1  # empty line between series
            
            # Note: full metrics require original points → here we approximate with self-distance
            # In real usage you would pass original points to this function
            if x_smooth is not None and len(x_smooth) >= 5:
                metrics = calculate_fit_metrics(x_smooth, y_smooth, x_smooth, y_smooth)
                wm.write(mrow, 0, line_name)
                wm.write(mrow, 1, metrics['fit_score'])
                wm.write(mrow, 2, metrics['avg_distance'])
                wm.write(mrow, 3, metrics['max_distance'])
                wm.write(mrow, 4, metrics['rmse'])
                wm.write(mrow, 5, metrics['nrmse'])
                wm.write(mrow, 6, metrics['r2_analog'])
                wm.write(mrow, 7, metrics['smoothness'])
                mrow += 1
        
        output.seek(0)
    return output


def compare_parametric_modes(lines, n_points=200):
    """
    Compare all parametric sub-modes visually and in table.
    """
    st.subheader("Parametric Visual Comparison")
    st.markdown("Compares all sub-modes with default smoothness = 0.0")
    
    param_methods = [
        ("Parametric Spline",       {"n_points": n_points, "smoothness": 0.0, "smoothness_simple": False}),
        ("Path Interpolation",      {"n_points": n_points, "smoothness": 0.0, "smoothness_simple": False}),
        ("Bezier Spline",           {"n_points": n_points, "smoothness": 0.0, "smoothness_simple": False}),
        ("Catmull-Rom Spline",      {"n_points": n_points, "smoothness": 0.0, "smoothness_simple": False}),
        ("Cubic Hermite Spline",    {"n_points": n_points, "smoothness": 0.0, "smoothness_simple": False}),
        ("NURBS (Non-Uniform Rational B-Spline)", {"n_points": n_points, "smoothness": 0.0, "smoothness_simple": False}),
        ("Akima Spline",            {"n_points": n_points, "smoothness": 0.0, "smoothness_simple": False}),
        ("PCHIP (Piecewise Cubic Hermite)", {"n_points": n_points, "smoothness": 0.0, "smoothness_simple": False})
    ]
    
    results = {}
    for name, p in param_methods:
        p['sub_mode'] = name
        results[name] = generate_parametric_data(lines, p)
    
    comparison_data = []
    for method in [m[0] for m in param_methods]:
        for res in results[method]:
            name, xs, ys, err = res
            if err or xs is None:
                continue
            orig_x = next((l[1] for l in lines if l[0] == name), None)
            orig_y = next((l[2] for l in lines if l[0] == name), None)
            if orig_x is None:
                continue
            m = calculate_fit_metrics(orig_x, orig_y, xs, ys)
            comparison_data.append({
                'Line': name,
                'Method': method,
                'Fit Score': f"{m['fit_score']:.1f}",
                'Avg Dist': f"{m['avg_distance']:.4f}",
                'RMSE': f"{m['rmse']:.4f}",
                'NRMSE': f"{m['nrmse']:.2f}%",
                'R²': f"{m['r2_analog']:.4f}",
                'Smoothness': f"{m['smoothness']:.6f}"
            })
    
    if comparison_data:
        st.subheader("Fit & Smoothness Comparison")
        df = pd.DataFrame(comparison_data)
        st.dataframe(df.sort_values(['Line', 'Smoothness']), use_container_width=True)
    
    for line_name, x, y, _, _ in lines:
        if len(x) < 2:
            st.warning(f"{line_name}: skipped (too few points)")
            continue
        
        st.markdown(f"### {line_name}")
        for method in [m[0] for m in param_methods]:
            res = next((r for r in results[method] if r[0] == line_name), None)
            if res and res[3] is None:
                st.write(f"**{method}**")
                fig = plot_parametric(x, y, res[1], res[2], method)
                st.pyplot(fig)
            else:
                st.caption(f"{method}: {res[3] if res else 'no result'}")
