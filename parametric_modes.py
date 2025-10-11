import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import pandas as pd
import io

def parametric_ui():
    """
    Render UI for Parametric Fitting mode.
    Returns: Dictionary of parameters including sub_mode, n_points, and smoothness
    """
    params = {}
    sub_mode_options = ["Parametric Spline", "Path Interpolation", "Bezier Spline"]
    params['sub_mode'] = st.selectbox("Choose Parametric Sub-Mode", sub_mode_options)
    params['n_points'] = st.number_input("Number of smoothed points per line", min_value=10, max_value=1000, value=200, step=10)
    params['smoothness'] = st.slider("Smoothness (higher values reduce overfitting)", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                                    help="For Parametric/Bezier Spline: Controls smoothing (0 = exact fit, higher = smoother). For Path Interpolation: Affects point density indirectly.")
    return params

def generate_parametric_data(lines, params):
    """
    Generate smoothed parametric data for each line using the specified sub-mode.
    Args:
        lines: List of tuples (line_name, x, y, has_duplicates, has_invalid_x)
        params: Parameters including sub_mode, n_points, and smoothness
    Returns: List of (line_name, x_smooth, y_smooth, error_message)
    """
    results = []
    n_points = params['n_points']
    smoothness = params['smoothness']

    for line_name, x, y, has_duplicates, has_invalid_x in lines:
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        
        if n < 2:
            results.append((line_name, None, None, "Insufficient points (need at least 2)"))
            continue
        
        # Check for duplicate x-values
        if has_duplicates or len(np.unique(x)) < len(x):
            if params['sub_mode'] in ["Parametric Spline", "Bezier Spline"]:
                results.append((line_name, None, None, "Duplicate x-values detected, which may affect spline-based methods"))
                continue
        
        try:
            sub_mode = params['sub_mode']
            
            if sub_mode == "Parametric Spline" or sub_mode == "Bezier Spline":
                u = np.linspace(0, 1, n)
                tck, _ = splprep([x, y], u=u, s=smoothness, k=3)
                u_new = np.linspace(0, 1, n_points)
                x_smooth, y_smooth = splev(u_new, tck)
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
            for x_val, y_val in zip(x_smooth, y_smooth):
                worksheet.write(row, 0, x_val)
                worksheet.write(row, 1, y_val)
                row += 1
            row += 1
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
    
    param_methods = [
        ("Parametric Spline", {"n_points": n_points, "smoothness": 0.0}),
        ("Path Interpolation", {"n_points": n_points, "smoothness": 0.0}),
        ("Bezier Spline", {"n_points": n_points, "smoothness": 0.0})
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
        
        # Create a single figure with subplots for all methods
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        fig.suptitle(f"Comparison for Line: {line_name}")
        
        for idx, (method, method_params) in enumerate(param_methods):
            result = next((r for r in results[method] if r[0] == line_name), None)
            ax = axes[idx]
            ax.scatter(x, y, color='blue', label='Original Points', s=50)
            if result and result[3] is None:
                x_smooth, y_smooth = result[1], result[2]
                ax.plot(x_smooth, y_smooth, color='red', label=f'Smoothed {method}')
            else:
                ax.text(0.5, 0.5, f"Error: {result[3] if result else 'No result'}", 
                        ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            ax.legend()
            ax.grid(True)
            ax.set_title(method)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Parametric Fitting App")
    params = parametric_ui()
    # Replace with your actual data source for lines_param
    lines_param = [
        ("Line1", [0, 1, 2, 3], [0, 1, 4, 9], False, False),
        ("Line2", [0, 1, 2], [0, 2, 0], False, False)
    ]
    
    # Check for duplicate x-values in data
    for line_name, x, y, _, _ in lines_param:
        if len(np.unique(x)) < len(x):
            st.warning(f"Line '{line_name}' has duplicate x-values: {x}")
    
    # Generate and plot smoothed data
    results = generate_parametric_data(lines_param, params)
    for line_name, x_smooth, y_smooth, error_message in results:
        if error_message:
            st.error(f"Line '{line_name}': {error_message}")
            continue
        st.markdown(f"### Line: {line_name}")
        line_data = next((line for line in lines_param if line[0] == line_name), None)
        if line_data:
            fig = plot_parametric(line_data[1], line_data[2], x_smooth, y_smooth, params['sub_mode'])
            st.pyplot(fig)
            plt.close(fig)
    
    # Button to trigger comparison
    if st.button("Compare Parametric Modes"):
        compare_parametric_modes(lines_param)

if __name__ == "__main__":
    main()
