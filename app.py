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
from trig_polar_fits import trig_polar_ui, fit_trigonometric, fit_polar, plot_trig_polar, compare_trig_polar_modes

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
        3. For Curve Fit, select a fitting method and parameters. For Visual Comparison, choose all lines or n random lines to compare methods. For Download Smoothed Data, select a method, parameters, and number of points. For Outlier Detection, choose an outlier detection method or number of outliers. For Parametric Fitting or Trigonometric and Polar Fitting, select a sub-mode with parameters or enable visual comparison of all sub-modes (all lines or n random lines).
        4. For functional modes (Curve Fit, Visual Comparison, etc.), optionally enable averaging of y values for duplicate x values to enable splines and smoothing methods.
        5. For functional modes, optionally view suggestions for the best method based on Adjusted R² (only Polynomial, Exponential, Logarithmic, Compound Poly+Log, and Trigonometric compared).
        6. Fit curves, view graphs, or download the output Excel.
        """)
    elif selected_guide == "Excel Format Guide":
        st.markdown("""
        **Output Excel Guide (Curve Fit, Visual Comparison, Trigonometric and Polar Fitting):**
        - **Columns**: 'Line Name', then coefficients/parameters, followed by R².
        - **Trigonometric/Polar Comparison**: 'Line Name', 'Fit Type', 'Sub-Model', 'Model Desc', coefficients, 'R2'.
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
        - **Trigonometric** (all include multipliers for x, e.g., sin(b*x)):
          - Cosine + Sine: y = sum(a_i*cos(b_i*x) + c_i*sin(d_i*x)) + c.
          - Tan + Cot: y = a*tan(b*x) + c/tan(d*x).
          - Hyperbolic: y = sum(a_i*cosh(b_i*x) + c_i*sinh(d_i*x)) + c.
          - Inverse Trig: y = a*arctan(b*x) + c*arcsin(d*x/(1+|d*x|)).
          - Complex Trig: y = sum(a_i*cos(b_i*x)*tan(c_i*x)) + c.
          - Combined: y = sum(a_i*cos(b_i*x) + c_i*sin(d_i*x) + e_i*cosh(f_i*x) + g_i*sinh(h_i*x)) + c.
        - **Polar** (all include multipliers for theta where applicable):
          - Power Spiral: r = a_0 + a_1*theta + ... + a_n*theta^n.
          - Rose Curve: r = a*cos(n*theta).
          - Archimedean Spiral: r = a + b*theta.
          - Complex Polar: r = sum(a_i*cos(b_i*theta)*theta^i) + c.
          - Combined: r = sum(a_i*theta^i) + b*cos(c*theta) + d*theta + e.
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
        - **Purpose**: Fits data using trigonometric functions (cos, sin, tan, cot, hyperbolic, inverse, all with multipliers for x) or polar equations (power spiral, rose curve, Archimedean spiral, complex polar, combined).
        - **Trigonometric Sub-Modes** (x multipliers, e.g., sin(b*x)):
          - **Cosine + Sine**: y = sum(a_i*cos(b_i*x) + c_i*sin(d_i*x)) + c.
          - **Tan + Cot**: y = a*tan(b*x) + c/tan(d*x).
          - **Hyperbolic**: y = sum(a_i*cosh(b_i*x) + c_i*sinh(d_i*x)) + c.
          - **Inverse Trig**: y = a*arctan(b*x) + c*arcsin(d*x/(1+|d*x|)).
          - **Complex Trig**: y = sum(a_i*cos(b_i*x)*tan(c_i*x)) + c.
          - **Combined**: y = sum(a_i*cos(b_i*x) + c_i*sin(d_i*x) + e_i*cosh(f_i*x) + g_i*sinh(h_i*x)) + c.
        - **Polar Sub-Modes** (theta multipliers where applicable):
          - **Power Spiral**: r = a_0 + a_1*theta + ... + a_n*theta^n.
          - **Rose Curve**: r = a*cos(n*theta).
          - **Archimedean Spiral**: r = a + b*theta.
          - **Complex Polar**: r = sum(a_i*cos(b_i*theta)*theta^i) + c.
          - **Combined**: r = sum(a_i*theta^i) + b*cos(c*theta) + d*theta + e.
        - **Parameters**:
          - Number of terms, scale factors, or specific constants.
          - Visual comparison mode: Compare all sub-modes for all lines or n random lines.
        - **Output**: Excel with coefficients, R², and LaTeX formula displayed in plots. Comparison mode produces an Excel with Line Name, Fit Type, Sub-Model, Model Desc, coefficients, R2.
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
                            output_df.to_excel(writer, index=False, sheet_name='Comparison')
                        output.seek(0)
                        st.download_button(
                            label="Download Comparison Results Excel",
                            data=output,
                            file_name="curve_fit_comparison.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

            elif mode == "Download Smoothed Data":
                st.subheader("Download Smoothed Data")
                st.markdown("Generate smoothed data for all lines using a selected method.")
                method = st.selectbox("Smoothing Method", fit_methods)
                params = smooth_data_ui(method)
                num_points = st.number_input("Number of points per line", min_value=10, max_value=1000, value=100)

                if st.button("Generate Smoothed Data"):
                    smoothed_data = []
                    for line_name, x, y, has_duplicates, has_invalid_x in lines:
                        if len(x) < 3:
                            st.warning(f"Line '{line_name}' skipped: only {len(x)} points (need at least 3).")
                            continue
                        if method in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"] and has_duplicates and not average_duplicates:
                            st.warning(f"Line '{line_name}' skipped for {method}: duplicate x values.")
                            continue
                        try:
                            x_smooth, y_smooth = generate_smoothed_data(x, y, method, params, num_points)
                            smoothed_data.append((line_name, x_smooth, y_smooth))
                        except Exception as e:
                            st.warning(f"Line '{line_name}' failed: {str(e)}")
                    if smoothed_data:
                        output = create_smoothed_excel(smoothed_data)
                        st.download_button(
                            label="Download Smoothed Data Excel",
                            data=output,
                            file_name="smoothed_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

            elif mode == "Outlier Detection and Cleaning":
                st.subheader("Outlier Detection and Cleaning")
                st.markdown("Detect and remove outliers, then fit curves to cleaned data.")
                detection_params, fit_method, fit_params = outlier_cleaner_ui()
                if st.button("Clean and Fit"):
                    cleaned_data = []
                    for line_name, x, y, has_duplicates, has_invalid_x in lines:
                        if len(x) < 3:
                            st.warning(f"Line '{line_name}' skipped: only {len(x)} points (need at least 3).")
                            continue
                        x_clean, y_clean = detect_outliers(x, y, detection_params)
                        if len(x_clean) < 3:
                            st.warning(f"Line '{line_name}' skipped after cleaning: only {len(x_clean)} points remain.")
                            continue
                        if fit_method in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"] and has_duplicates and not average_duplicates:
                            st.warning(f"Line '{line_name}' skipped for {fit_method}: duplicate x values in cleaned data.")
                            continue
                        fit_func = fit_funcs[fit_method]
                        if fit_method in ["Trigonometric", "Polar"]:
                            coeffs, r2, desc, formula = fit_func(x_clean, y_clean, fit_params.get('trig_model') or fit_params.get('polar_model'), fit_params)
                        else:
                            coeffs, r2, desc = fit_func(x_clean, y_clean, **fit_params)
                            formula = None
                        if coeffs is not None:
                            st.write(f"Line '{line_name}' - {fit_method}: {desc}, R² = {r2:.4f}")
                            if formula:
                                st.markdown(f"Formula: ${formula}$")
                                st.markdown(f"```latex\n{formula}\n```")
                            if fit_method == "Random Forest":
                                fig = plot_random_forest(x_clean, y_clean, coeffs, fit_method, fit_params)
                            elif fit_method in ["Trigonometric", "Polar"]:
                                fig = plot_trig_polar(x_clean, y_clean, coeffs, fit_method, fit_params, formula)
                            else:
                                fig = plot_cleaned_data(x, y, x_clean, y_clean, coeffs, fit_method, fit_params)
                            st.pyplot(fig)
                            cleaned_data.append((line_name, x_clean, y_clean))
                        else:
                            st.warning(f"Line '{line_name}' - {fit_method}: {desc}")
                    if cleaned_data:
                        output = create_cleaned_excel(cleaned_data)
                        st.download_button(
                            label="Download Cleaned Data Excel",
                            data=output,
                            file_name="cleaned_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

            elif mode == "Parametric Fitting":
                st.subheader("Parametric Fitting")
                st.markdown("Fit non-functional curves (e.g., circles, loops) using parametric methods.")
                params = parametric_ui()
                if params.get('compare_modes', False):
                    if st.button("Compare Parametric Modes"):
                        compare_parametric_modes(lines, params)
                else:
                    num_points = st.number_input("Number of smoothed points per line", min_value=10, max_value=1000, value=100)
                    if st.button("Generate Parametric Fit"):
                        parametric_data = []
                        for line_name, x, y, _, _ in lines:
                            if len(x) < 3:
                                st.warning(f"Line '{line_name}' skipped: only {len(x)} points (need at least 3).")
                                continue
                            try:
                                x_fit, y_fit = generate_parametric_data(x, y, params, num_points)
                                fig = plot_parametric(x, y, x_fit, y_fit, params)
                                st.pyplot(fig)
                                parametric_data.append((line_name, x_fit, y_fit))
                            except Exception as e:
                                st.warning(f"Line '{line_name}' failed: {str(e)}")
                        if parametric_data:
                            output = create_parametric_excel(parametric_data)
                            st.download_button(
                                label="Download Parametric Data Excel",
                                data=output,
                                file_name="parametric_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

            elif mode == "Trigonometric and Polar Fitting":
                st.subheader("Trigonometric and Polar Fitting")
                st.markdown("Fit data using trigonometric or polar equations, or compare all sub-modes.")
                compare_trig_polar = st.checkbox("Enable Trigonometric and Polar Visual Comparison", value=False)
                params = trig_polar_ui()
                
                if compare_trig_polar:
                    show_all = st.checkbox("Show all lines", value=True)
                    n_lines = 0
                    if not show_all:
                        n_lines = st.number_input("Number of random lines", min_value=1, max_value=len(lines), value=min(3, len(lines)))
                    if st.button("Compare Trigonometric and Polar Modes"):
                        selected_lines = lines if show_all else random.sample(lines, min(n_lines, len(lines)))
                        compare_trig_polar_modes(selected_lines, params)
                else:
                    selected_line = st.selectbox("Select Line", [line[0] for line in lines])
                    if st.button("Fit Curve"):
                        line_data = next((line for line in lines if line[0] == selected_line), None)
                        if line_data:
                            x, y, has_duplicates, has_invalid_x = line_data[1], line_data[2], line_data[3], line_data[4]
                            if len(x) < 3:
                                st.warning(f"Line '{selected_line}' has only {len(x)} points; need at least 3.")
                            else:
                                fit_func = fit_funcs[params['fit_type']]
                                coeffs, r2, desc, formula = fit_func(x, y, params.get('trig_model') or params.get('polar_model'), params)
                                if coeffs is not None:
                                    st.write(f"{params['fit_type']} - {params.get('trig_model') or params.get('polar_model')}: {desc}, R² = {r2:.4f}")
                                    st.markdown(f"Formula: ${formula}$")
                                    st.markdown(f"```latex\n{formula}\n```")
                                    fig = plot_trig_polar(x, y, coeffs, params['fit_type'], params, formula)
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
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
