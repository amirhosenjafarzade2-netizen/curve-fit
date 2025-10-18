# app.py - Main Streamlit app for curve fitting
# Run with: streamlit run app.py
# Requirements: streamlit, pandas, numpy, scikit-learn, scipy, xlsxwriter, matplotlib, openpyxl, statsmodels, pywavelets, sympy
# Install via: pip install -r requirements.txt

import streamlit as st
import pandas as pd
import io
import random
from utils import parse_excel, suggest_best_method, fit_polynomial, fit_exponential, fit_logarithmic, fit_compound_poly_log, fit_spline, fit_savgol, fit_lowess, fit_exponential_smoothing, fit_gaussian_smoothing, fit_wavelet_denoising
from plotting import plot_fit
from random_forest import fit_random_forest, plot_random_forest, random_forest_ui
from smooth_data import smooth_data_ui, generate_smoothed_data, create_smoothed_excel
from outlier_cleaner import outlier_cleaner_ui, detect_outliers, plot_cleaned_data, create_cleaned_excel
from parametric_modes import parametric_ui, generate_parametric_data, plot_parametric, create_parametric_excel, compare_parametric_modes
from trig_polar_fits import trig_polar_ui, fit_trigonometric, fit_polar, plot_trig_polar

# Custom CSS for button styling and cleaner expander styling
st.markdown("""
<style>
.stButton > button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 8px 16px;
    margin: 5px;
    font-size: 14px;
    font-weight: 500;
    transition: background-color 0.3s;
}
.stButton > button:hover {
    background-color: #45a049;
}
.stButton > button:focus {
    outline: none;
    box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
}
.stExpander {
    border: 1px solid #d3d3d3;
    border-radius: 5px;
    background-color: #f9f9f9;
    padding: 10px;
    margin-bottom: 10px;
}
.stExpander > div > div {
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("Curve Fitting App")

# Define fitting functions
fit_funcs = {
    "Polynomial": fit_polynomial,
    "Exponential": fit_exponential,
    "Logarithmic": fit_logarithmic,
    "Compound Poly+Log": fit_compound_poly_log,
    "Spline": fit_spline,
    "Savitzky-Golay": fit_savgol,
    "LOWESS": fit_lowess,
    "Exponential Smoothing": fit_exponential_smoothing,
    "Gaussian Smoothing": fit_gaussian_smoothing,
    "Wavelet Denoising": fit_wavelet_denoising,
    "Random Forest": fit_random_forest,
    "Trigonometric": fit_trigonometric,
    "Polar": fit_polar
}

# Single collapsible window for all guides
with st.expander("View Guides", expanded=False):
    guide_options = [
        "General Instructions",
        "Excel Format Guide",
        "Curve Fit Formulas",
        "Outlier Detection Guide",
        "Parametric Fitting Guide",
        "Trigonometric and Polar Fitting Guide"
    ]
    selected_guide = st.selectbox("Select a Guide", guide_options)

    if selected_guide == "General Instructions":
        st.markdown("""
        1. Upload an Excel file with line data: Line name in column A (B empty), then x in A, y in B below it (next line after an empty row, similarly).
        2. Choose a mode: Curve Fit (single method with parameters), Visual Comparison with Graphs (compare all methods), Download Smoothed Data (generate smoothed curves), Outlier Detection and Cleaning (remove outliers and fit), Parametric Fitting (fit non-functional curves like circles or vertical lines), or Trigonometric and Polar Fitting (fit using trig or polar equations).
        3. For Curve Fit, select a fitting method and parameters. For Visual Comparison, choose all lines or n random lines to compare methods. For Download Smoothed Data, select a method, parameters, and number of points. For Outlier Detection, choose an outlier detection method or number of outliers. For Parametric Fitting, select a sub-mode (e.g., Parametric Spline, Path Interpolation) and parameters. For Trigonometric and Polar Fitting, select a fit type (Trigonometric or Polar) and sub-model with parameters.
        4. For functional modes (Curve Fit, Visual Comparison, etc.), optionally enable averaging of y values for duplicate x values to enable splines and smoothing methods.
        5. For functional modes, optionally view suggestions for the best method based on Adjusted R² (only Polynomial, Exponential, Logarithmic, Compound Poly+Log, and Trigonometric compared).
        6. Fit curves, view graphs, or download the output Excel.
        """)
    elif selected_guide == "Excel Format Guide":
        st.markdown("""
        **Output Excel Guide (Curve Fit, Visual Comparison, Trigonometric and Polar Fitting):**
        - **Columns**: 'Line Name', then coefficients/parameters, followed by R².
        - **Smoothed Data, Outlier Cleaning, and Parametric Fitting Excel Format**: Similar to input: Line name in column A (B empty), then x in A, y in B, empty rows.
        """)
    elif selected_guide == "Curve Fit Formulas":
        st.markdown("""
        - **Polynomial**: Coefficients from highest degree to constant (e.g., degree 2: a_2, a_1, a_0 for y = a_2*x^2 + a_1*x + a_0).
        - **Exponential**: a, b, c for y = a * exp(b*x) + c.
        - **Logarithmic**: a, b for y = a * log(x) + b (requires x > 0; points with x ≤ 0 are ignored).
        - **Compound Poly+Log**: a, b, c for y = a*x^2 + b*log(x) + c (requires x > 0; points with x ≤ 0 are ignored).
        - **Spline**: Spline coefficients followed by knots (variable length; interpret with scipy.interpolate.UnivariateSpline). Requires strictly increasing x values unless duplicates are averaged.
        - **Savitzky-Golay**: window_length, polyorder. Requires strictly increasing x values unless duplicates are averaged.
        - **LOWESS**: frac. Requires strictly increasing x values unless duplicates are averaged.
        - **Exponential Smoothing**: alpha. Requires strictly increasing x values unless duplicates are averaged.
        - **Gaussian Smoothing**: sigma. Requires strictly increasing x values unless duplicates are averaged.
        - **Wavelet Denoising**: wavelet_name, level, threshold. Requires strictly increasing x values unless duplicates are averaged.
        - **Random Forest**: n_estimators (number of trees).
        - **Trigonometric**:
          - Cosine + Sine: a_i*cos(b_i*x) + c_i*sin(d_i*x) + c, for i terms.
          - Tan + Cot: a*tan(b*x) + c/tan(d*x).
          - Hyperbolic: a_i*cosh(b_i*x) + c_i*sinh(d_i*x) + c, for i terms.
          - Inverse Trig: a*arctan(b*x) + c*arcsin(d*x/(1+|d*x|)).
          - Complex Trig: a_i*cos(b_i*x)*tan(c_i*x) + c, for i terms.
        - **Polar**:
          - Power Spiral: r = a_0 + a_1*theta + ... + a_n*theta^n.
          - Rose Curve: r = a*cos(n*theta).
          - Archimedean Spiral: r = a + b*theta.
          - Complex Polar: r = a_i*cos(b_i*theta)*theta^i + c, for i terms.
        """)
    elif selected_guide == "Outlier Detection Guide":
        st.markdown("""
        - **Purpose**: Detects and removes outliers using Z-score, IQR, Isolation Forest, or a fixed number, then fits curves to cleaned data.
        - **Parameters**:
          - **Z-score**: Threshold (e.g., 3.0) for absolute Z-score.
          - **IQR**: Multiplier (e.g., 1.5) for interquartile range bounds.
          - **Isolation Forest**: Contamination (proportion of outliers, e.g., 0.1).
          - **Fixed Number**: Number of outliers to remove (based on residuals from preliminary fit).
          - Select fitting method and parameters for cleaned data.
        - **Output**: Excel with cleaned data (line name in A, B empty, x in A, y in B, empty rows).
        """)
    elif selected_guide == "Parametric Fitting Guide":
        st.markdown("""
        - **Purpose**: Fits non-functional curves (e.g., circles, vertical lines, loops) by treating x and y as parametric functions of t (x=f(t), y=g(t)).
        - **Sub-Modes**:
          - **Parametric Spline**: Fits a smooth B-spline curve through points, preserving order.
          - **Path Interpolation**: Interpolates points along the path based on arc length.
          - **Bezier Spline**: Fits a B-spline approximating Bezier-like curves.
          - **Gaussian Process**: Fits x(t) and y(t) using Gaussian Processes for noisy data.
        - **Parameters**:
          - Number of smoothed points per line.
          - For Gaussian Process: RBF length scale.
        - **Output**: Excel with smoothed parametric data (line name in A, B empty, x in A, y in B, empty rows).
        - **Note**: Points are not sorted by x, and duplicates are preserved to maintain curve geometry.
        """)
    elif selected_guide == "Trigonometric and Polar Fitting Guide":
        st.markdown("""
        - **Purpose**: Fits data using trigonometric functions (cos, sin, tan, cot, hyperbolic, inverse) or polar equations (power spiral, rose curve, Archimedean spiral, complex polar).
        - **Trigonometric Sub-Modes**:
          - **Cosine + Sine**: y = sum(a_i*cos(b_i*x) + c_i*sin(d_i*x)) + c.
          - **Tan + Cot**: y = a*tan(b*x) + c/tan(d*x).
          - **Hyperbolic**: y = sum(a_i*cosh(b_i*x) + c_i*sinh(d_i*x)) + c.
          - **Inverse Trig**: y = a*arctan(b*x) + c*arcsin(d*x/(1+|d*x|)).
          - **Complex Trig**: y = sum(a_i*cos(b_i*x)*tan(c_i*x)) + c.
        - **Polar Sub-Modes**:
          - **Power Spiral**: r = a_0 + a_1*theta + ... + a_n*theta^n.
          - **Rose Curve**: r = a*cos(n*theta).
          - **Archimedean Spiral**: r = a + b*theta.
          - **Complex Polar**: r = sum(a_i*cos(b_i*theta)*theta^i) + c.
        - **Parameters**:
          - Number of terms (for some models), scale factors, or specific constants.
        - **Output**: Excel with coefficients, R², and LaTeX formula displayed in plots.
        """)

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
average_duplicates = st.checkbox("Average y values for duplicate x values (enables splines and other smoothing methods for functional modes)", value=True)

if uploaded_file:
    try:
        lines, skipped_lines = parse_excel(uploaded_file, average_duplicates=average_duplicates)
        if not lines:
            st.error("No valid lines found in the file.")
        else:
            st.success(f"Found {len(lines)} valid lines.")
            if skipped_lines and not average_duplicates:
                st.warning(f"Warnings for some lines: {', '.join(skipped_lines)}")

            mode = st.selectbox("Select Mode", [
                "Curve Fit",
                "Visual Comparison with Graphs",
                "Download Smoothed Data",
                "Outlier Detection and Cleaning",
                "Parametric Fitting",
                "Trigonometric and Polar Fitting"
            ])

            fit_methods = [
                "Polynomial", "Exponential", "Logarithmic", "Compound Poly+Log",
                "Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing",
                "Gaussian Smoothing", "Wavelet Denoising", "Random Forest",
                "Trigonometric", "Polar"
            ]

            if mode == "Curve Fit":
                st.subheader("Curve Fitting")
                st.markdown("Select a fitting method and line, then specify parameters.")
                selected_line = st.selectbox("Select Line", [line[0] for line in lines])
                method = st.selectbox("Fitting Method", fit_methods)
                st.session_state['method'] = method

                # Get parameters based on method
                params = {}
                min_points = 3
                if method == "Polynomial":
                    params = {'degree': st.number_input("Polynomial Degree", min_value=1, max_value=10, value=2)}
                    min_points = params['degree'] + 1
                elif method == "Spline":
                    params = {'degree': st.number_input("Spline Degree (1-5, 3=cubic)", min_value=1, max_value=5, value=3)}
                    min_points = params['degree'] + 1
                elif method == "Savitzky-Golay":
                    params = {
                        'window': st.number_input("Window length (odd number >=3)", min_value=3, max_value=101, value=5, step=2),
                        'polyorder': st.number_input("Polynomial order", min_value=0, max_value=5, value=2)
                    }
                    min_points = params['window']
                elif method == "LOWESS":
                    params = {'frac': st.slider("Fraction of data for local fit", min_value=0.1, max_value=1.0, value=0.3, step=0.05)}
                elif method == "Exponential Smoothing":
                    params = {'alpha': st.slider("Smoothing factor (alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)}
                elif method == "Gaussian Smoothing":
                    params = {'sigma': st.number_input("Gaussian sigma", min_value=0.1, max_value=10.0, value=1.0, step=0.1)}
                elif method == "Wavelet Denoising":
                    params = {
                        'wavelet': st.selectbox("Wavelet family", options=['db4', 'haar', 'sym4', 'coif1'], index=0),
                        'level': st.number_input("Decomposition level", min_value=1, max_value=5, value=1),
                        'threshold': st.slider("Threshold factor", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
                    }
                elif method == "Random Forest":
                    params = random_forest_ui()
                elif method in ["Trigonometric", "Polar"]:
                    params = trig_polar_ui()

                show_suggestions = st.checkbox("Show best method suggestion (Adjusted R²)", value=False)
                if show_suggestions and method not in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"]:
                    line_data = next((line for line in lines if line[0] == selected_line), None)
                    if line_data:
                        x, y, _, has_invalid_x = line_data[1], line_data[2], line_data[3], line_data[4]
                        suggestion = suggest_best_method(x, y, has_invalid_x)
                        if suggestion:
                            st.write(f"Suggested best method: {suggestion[0]} (Adjusted R²: {suggestion[3]:.4f}) - {suggestion[4]}")

                if st.button("Fit Curve"):
                    line_data = next((line for line in lines if line[0] == selected_line), None)
                    if line_data:
                        x, y, has_duplicates, has_invalid_x = line_data[1], line_data[2], line_data[3], line_data[4]
                        if len(x) < min_points:
                            st.warning(f"Line '{selected_line}' has only {len(x)} points; need at least {min_points} for {method}.")
                        elif method in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"] and has_duplicates and not average_duplicates:
                            st.warning(f"Line '{selected_line}' has duplicate x values, cannot fit {method} without averaging duplicates.")
                        else:
                            fit_func = fit_funcs[method]
                            if method in ["Trigonometric", "Polar"]:
                                coeffs, r2, desc, formula = fit_func(x, y, params.get('trig_model') or params.get('polar_model'), params)
                            else:
                                coeffs, r2, desc = fit_func(x, y, **params)
                                formula = None
                            if coeffs is not None:
                                st.write(f"{desc}, R² = {r2:.4f}")
                                if formula:
                                    st.markdown(f"Formula: ${formula}$")
                                    st.markdown(f"```latex\n{formula}\n```")
                                if method == "Random Forest":
                                    fig = plot_random_forest(x, y, coeffs, method, params)
                                elif method in ["Trigonometric", "Polar"]:
                                    fig = plot_trig_polar(x, y, coeffs, method, params, formula)
                                else:
                                    fig = plot_fit(x, y, coeffs, method, params)
                                st.pyplot(fig)
                            else:
                                st.warning(f"Fit failed: {desc}")

            elif mode == "Visual Comparison with Graphs":
                st.subheader("Visual Comparison")
                st.markdown("Compare all fitting methods for selected lines (or n random lines).")
                show_all = st.checkbox("Show all lines", value=True)
                n_lines = 0
                if not show_all:
                    n_lines = st.number_input("Number of random lines", min_value=1, max_value=len(lines), value=min(3, len(lines)))
                show_suggestions = st.checkbox("Show best method suggestion per line (Adjusted R²)", value=False)

                if st.button("Generate Comparison"):
                    selected_lines = lines if show_all else random.sample(lines, min(n_lines, len(lines)))
                    results = []
                    for line_name, x, y, has_duplicates, has_invalid_x in selected_lines:
                        st.markdown(f"### Line: {line_name}")
                        if len(x) < 3:
                            st.warning(f"Line '{line_name}' skipped: only {len(x)} points (need at least 3).")
                            continue
                        if show_suggestions:
                            suggestion = suggest_best_method(x, y, has_invalid_x)
                            if suggestion:
                                st.write(f"Suggested best method: {suggestion[0]} (Adjusted R²: {suggestion[3]:.4f}) - {suggestion[4]}")
                        for method in fit_methods:
                            min_points = 3
                            if method == "Polynomial":
                                params = {'degree': 2}
                                min_points = params['degree'] + 1
                            elif method == "Spline":
                                params = {'degree': 3}
                                min_points = params['degree'] + 1
                            elif method == "Savitzky-Golay":
                                params = {'window': 5, 'polyorder': 2}
                                min_points = params['window']
                            elif method == "LOWESS":
                                params = {'frac': 0.3}
                            elif method == "Exponential Smoothing":
                                params = {'alpha': 0.5}
                            elif method == "Gaussian Smoothing":
                                params = {'sigma': 1.0}
                            elif method == "Wavelet Denoising":
                                params = {'wavelet': 'db4', 'level': 1, 'threshold': 0.1}
                            elif method == "Random Forest":
                                params = {'n_estimators': 100}
                            elif method == "Trigonometric":
                                params = {'trig_model': 'Cosine + Sine', 'n_terms': 2}
                            elif method == "Polar":
                                params = {'polar_model': 'Rose Curve', 'n_petals': 4}
                            if len(x) < min_points:
                                st.warning(f"Line '{line_name}' skipped for {method}: only {len(x)} points (need at least {min_points}).")
                                continue
                            if method in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"] and has_duplicates and not average_duplicates:
                                st.warning(f"Line '{line_name}' skipped for {method}: duplicate x values.")
                                continue
                            fit_func = fit_funcs[method]
                            if method in ["Trigonometric", "Polar"]:
                                coeffs, r2, desc, formula = fit_func(x, y, params.get('trig_model') or params.get('polar_model'), params)
                            else:
                                coeffs, r2, desc = fit_func(x, y, **params)
                                formula = None
                            if coeffs is not None:
                                st.write(f"{method}: {desc}, R² = {r2:.4f}")
                                if formula:
                                    st.markdown(f"Formula: ${formula}$")
                                    st.markdown(f"```latex\n{formula}\n```")
                                if method == "Random Forest":
                                    fig = plot_random_forest(x, y, coeffs, method, params)
                                elif method in ["Trigonometric", "Polar"]:
                                    fig = plot_trig_polar(x, y, coeffs, method, params, formula)
                                else:
                                    fig = plot_fit(x, y, coeffs, method, params)
                                st.pyplot(fig)
                                results.append((line_name, coeffs, r2, desc))
                            else:
                                st.warning(f"Line '{line_name}' - {method}: {desc}")
                    if results:
                        max_coeffs = max(len(row[1]) for row in results if row[1] is not None)
                        columns = ['Line Name', 'Model Desc'] + [f'param_{i}' for i in range(max_coeffs)] + ['R2']
                        output_df = pd.DataFrame([[r[0], r[3]] + r[1] + [r[2]] for r in results], columns=columns)
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            output_df.to_excel(writer, index=False, sheet_name='Fits')
                        output.seek(0)
                        st.download_button(
                            label="Download Fit Results Excel",
                            data=output,
                            file_name="fit_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

            elif mode == "Download Smoothed Data":
                st.subheader("Download Smoothed Data")
                st.markdown("Generate smoothed data for all lines using the selected method.")
                method = st.selectbox("Smoothing Method", fit_methods)
                st.session_state['method'] = method
                params = smooth_data_ui()
                if st.button("Generate Smoothed Data"):
                    smoothed_results = generate_smoothed_data(lines, method, params, fit_funcs)
                    for line_name, x_smooth, y_smooth, error_message in smoothed_results:
                        if error_message:
                            st.warning(f"Line '{line_name}': {error_message}")
                        else:
                            fig = plot_fit(x_smooth, y_smooth, None, method, params)
                            st.pyplot(fig)
                    if smoothed_results:
                        output = create_smoothed_excel(smoothed_results)
                        st.download_button(
                            label="Download Smoothed Data Excel",
                            data=output,
                            file_name=f"{method.lower().replace(' ', '_')}_smoothed_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

            elif mode == "Outlier Detection and Cleaning":
                st.subheader("Outlier Detection and Cleaning")
                st.markdown("Remove outliers from data and fit curves to the cleaned data.")
                params = outlier_cleaner_ui()
                method = params['fit_method']
                detection_method = params['detection_method']
                min_points = 3
                if method == "Polynomial":
                    min_points = params.get('degree', 2) + 1
                elif method == "Spline":
                    min_points = params.get('degree', 3) + 1
                elif method == "Savitzky-Golay":
                    min_points = params.get('window', 5)
                filtered_lines = [(name, x, y, has_duplicates, has_invalid_x) for name, x, y, has_duplicates, has_invalid_x in lines if len(x) >= min_points]
                if len(filtered_lines) < len(lines):
                    st.warning(f"Some lines skipped due to insufficient points for {method} (need at least {min_points}).")

                if st.button("Detect Outliers and Fit"):
                    cleaned_results = []
                    fit_results = []
                    st.subheader("Outlier Detection and Fit Results")
                    for line_name, x, y, has_duplicates, has_invalid_x in filtered_lines:
                        try:
                            if method in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"] and has_duplicates and not average_duplicates:
                                st.warning(f"Line '{line_name}': Skipped for {method} due to duplicate x values.")
                                cleaned_results.append((line_name, None, None, f"Skipped due to duplicate x values for {method}"))
                                continue
                            fit_func = fit_funcs[method]
                            mask, outlier_indices = detect_outliers(x, y, detection_method, params, fit_func if detection_method == "Fixed Number" else None)
                            if mask is None:
                                st.warning(f"Line '{line_name}': {outlier_indices}")
                                cleaned_results.append((line_name, None, None, outlier_indices))
                                continue
                            x_clean = x[mask]
                            y_clean = y[mask]
                            cleaned_results.append((line_name, x_clean, y_clean, None))
                            st.write(f"Line '{line_name}': Removed {len(outlier_indices)} outliers")

                            if len(x_clean) >= min_points:
                                coeffs, r2, model_desc = fit_func(x_clean, y_clean, **{k: v for k, v in params.items() if k not in ['detection_method', 'fit_method']})
                                if coeffs is not None:
                                    fit_results.append((line_name, coeffs, r2, model_desc))
                                    st.write(f"Line '{line_name}': {model_desc}, R² = {r2:.4f}")
                                    fig = plot_cleaned_data(x, y, mask, coeffs, method, params)
                                    st.pyplot(fig)
                                else:
                                    st.warning(f"Line '{line_name}': Failed to fit {method} on cleaned data")
                            else:
                                st.warning(f"Line '{line_name}': Not enough points after outlier removal (need {min_points})")
                        except Exception as e:
                            st.warning(f"Line '{line_name}': Failed to process: {str(e)}")
                            cleaned_results.append((line_name, None, None, str(e)))

                    if cleaned_results:
                        output = create_cleaned_excel(cleaned_results)
                        st.download_button(
                            label="Download Cleaned Data Excel",
                            data=output,
                            file_name="cleaned_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    if fit_results:
                        max_coeffs = max(len(row[1]) for row in fit_results if row[1] is not None)
                        columns = ['Line Name', 'Model Desc'] + [f'param_{i}' for i in range(max_coeffs)] + ['R2']
                        output_df = pd.DataFrame([[r[0], r[3]] + r[1] + [r[2]] for r in fit_results], columns=columns)
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            output_df.to_excel(writer, index=False, sheet_name='Cleaned_Fits')
                        output.seek(0)
                        st.download_button(
                            label="Download Cleaned Fits Excel",
                            data=output,
                            file_name="cleaned_fits.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    if not cleaned_results and not fit_results:
                        st.error("No data could be processed.")

            elif mode == "Parametric Fitting":
                st.subheader("Parametric and Path Fitting")
                st.markdown("This mode treats data as parametric curves or paths (non-functional, preserves point order). Generates smoothed/interpolated points along the curve/path.")
                
                lines_param, skipped_param = parse_excel(uploaded_file, average_duplicates=False, sort_by_x=False)
                if not lines_param:
                    st.error("No valid lines found for parametric fitting.")
                else:
                    st.success(f"Found {len(lines_param)} valid lines for parametric fitting.")
                    if skipped_param:
                        st.warning(f"Warnings for some lines: {', '.join(skipped_param)}")
                
                compare_parametric = st.checkbox("Enable Parametric Visual Comparison", value=False)
                
                params = parametric_ui()
                
                if compare_parametric:
                    compare_parametric_modes(lines_param)
                else:
                    if st.button("Generate and Plot Parametric Data"):
                        parametric_results = generate_parametric_data(lines_param, params)
                        sub_mode = params['sub_mode']
                        
                        st.subheader("Parametric Results")
                        for line_name, x_smooth, y_smooth, error_message in parametric_results:
                            if error_message:
                                st.warning(f"Line '{line_name}': {error_message}")
                            else:
                                orig_line = next((l for l in lines_param if l[0] == line_name), None)
                                if orig_line:
                                    orig_x, orig_y = orig_line[1], orig_line[2]
                                    fig = plot_parametric(orig_x, orig_y, x_smooth, y_smooth, sub_mode)
                                    st.pyplot(fig)
                        
                        if parametric_results:
                            output = create_parametric_excel(parametric_results)
                            st.download_button(
                                label="Download Parametric Data Excel",
                                data=output,
                                file_name=f"{sub_mode.lower().replace(' ', '_')}_parametric_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

            elif mode == "Trigonometric and Polar Fitting":
                st.subheader("Trigonometric and Polar Fitting")
                st.markdown("Fit data using trigonometric or polar equations. Trigonometric fits use y as a function of x. Polar fits treat data as (r, theta) converted from (x, y).")
                selected_line = st.selectbox("Select Line", [line[0] for line in lines])
                params = trig_polar_ui()
                fit_type = params['fit_type']
                sub_model = params.get('trig_model') or params.get('polar_model')

                if st.button("Fit Curve"):
                    line_data = next((line for line in lines if line[0] == selected_line), None)
                    if line_data:
                        x, y, has_duplicates, has_invalid_x = line_data[1], line_data[2], line_data[3], line_data[4]
                        if len(x) < 3:
                            st.warning(f"Line '{selected_line}' has only {len(x)} points; need at least 3.")
                        else:
                            fit_func = fit_funcs[fit_type]
                            coeffs, r2, desc, formula = fit_func(x, y, sub_model, params)
                            if coeffs is not None:
                                st.write(f"{desc}, R² = {r2:.4f}")
                                st.markdown(f"Formula: ${formula}$")
                                st.markdown(f"```latex\n{formula}\n```")
                                fig = plot_trig_polar(x, y, coeffs, fit_type, params, formula)
                                st.pyplot(fig)
                                # Prepare Excel output
                                results = [(selected_line, coeffs, r2, desc)]
                                max_coeffs = len(coeffs)
                                columns = ['Line Name', 'Model Desc'] + [f'param_{i}' for i in range(max_coeffs)] + ['R2']
                                output_df = pd.DataFrame([[r[0], r[3]] + r[1] + [r[2]] for r in results], columns=columns)
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    output_df.to_excel(writer, index=False, sheet_name='Trig_Polar_Fits')
                                output.seek(0)
                                st.download_button(
                                    label="Download Fit Results Excel",
                                    data=output,
                                    file_name="trig_polar_fits.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            else:
                                st.warning(f"Fit failed: {desc}")

    except ValueError as e:
        st.error(f"Failed to read Excel file: {str(e)}")
