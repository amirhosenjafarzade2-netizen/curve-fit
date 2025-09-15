# app.py - Main Streamlit app for curve fitting
# Run with: streamlit run app.py
# Requirements: streamlit, pandas, numpy, scikit-learn, scipy, xlsxwriter, matplotlib, openpyxl, statsmodels, pywavelets
# Install via: pip install -r requirements.txt

import streamlit as st
import pandas as pd
import io
import random
from utils import parse_excel, suggest_best_method, fit_polynomial, fit_exponential, fit_logarithmic, fit_compound_poly_log, fit_spline, fit_savgol, fit_lowess, fit_exponential_smoothing, fit_gaussian_smoothing, fit_wavelet_denoising
from plotting import plot_fit
from random_forest import fit_random_forest, plot_random_forest, random_forest_ui
from smooth_data import smooth_data_ui, generate_smoothed_data, create_smoothed_excel
from constrained_optimization import constrained_optimization_ui, optimize_coefficients, plot_constrained_fit, create_constrained_excel
from outlier_cleaner import outlier_cleaner_ui, detect_outliers, plot_cleaned_data, create_cleaned_excel

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

# Single collapsible window for all guides
with st.expander("View Guides", expanded=False):
    guide_options = [
        "General Instructions",
        "Excel Format Guide",
        "Curve Fit Formulas",
        "Constrained Optimization Guide",
        "Outlier Detection Guide"
    ]
    selected_guide = st.selectbox("Select a Guide", guide_options)

    if selected_guide == "General Instructions":
        st.markdown("""
        1. Upload an Excel file with line data: Line name in column A (B empty), then x in A, y in B below it (next line after an empty row, similarly).
        2. Choose a mode: Curve Fit (single method with parameters), Visual Comparison with Graphs (compare all methods), Download Smoothed Data (generate smoothed curves), Constrained Optimization (fit with restrictive points), or Outlier Detection and Cleaning (remove outliers and fit).
        3. For Curve Fit, select a fitting method and parameters. For Visual Comparison, choose all lines or n random lines to compare methods. For Download Smoothed Data, select a method, parameters, and number of points. For Constrained Optimization, specify restrictive points. For Outlier Detection, choose an outlier detection method or number of outliers.
        4. Optionally enable averaging of y values for duplicate x values (enables splines and other smoothing methods).
        5. Optionally view suggestions for the best method based on Adjusted R² (only Polynomial, Exponential, Logarithmic, and Compound Poly+Log compared).
        6. Fit curves, view graphs, or download the output Excel.
        """)
    elif selected_guide == "Excel Format Guide":
        st.markdown("""
        **Output Excel Guide (Curve Fit, Visual Comparison, Constrained Optimization):**
        - **Columns**: 'Line Name', then coefficients/parameters, followed by R².
        - **Smoothed Data and Outlier Cleaning Excel Format**: Similar to input: Line name in column A (B empty), then x in A, y in B below it, empty row, next line, etc.
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
        """)
    elif selected_guide == "Constrained Optimization Guide":
        st.markdown("""
        - **Purpose**: Fits curves (e.g., Polynomial) with coefficients optimized to pass through user-specified points (x, y).
        - **Parameters**: Select method (e.g., Polynomial), degree, number of manual restrictive points, and their (x, y) coordinates. If only 1 manual point is specified, options include adding the last data point as a constraint or adding random points from a user-defined range (e.g., 40%-90% of data).
        - **Output**: Excel with line name, model description, optimized coefficients, and R².
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

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
average_duplicates = st.checkbox("Average y values for duplicate x values (enables splines and other smoothing methods for all lines)", value=True)

if uploaded_file:
    try:
        lines, skipped_lines = parse_excel(uploaded_file, average_duplicates=average_duplicates)
        if not lines:
            st.error("No valid lines found in the file.")
        else:
            st.success(f"Found {len(lines)} valid lines.")
            if skipped_lines and not average_duplicates:
                st.warning(f"Skipped {len(skipped_lines)} lines due to duplicate x values (not suitable for splines or smoothing methods): {', '.join(skipped_lines)}")

            # Option to show suggested best method
            show_suggestions = st.radio("Show suggested best method for each line?", ["No", "Yes"], index=0)

            # Mode selection
            mode = st.radio("Select Mode", ["Curve Fit", "Visual Comparison with Graphs", "Download Smoothed Data", 
                                            "Constrained Optimization", "Outlier Detection and Cleaning"])

            # Define fit and plot functions for reuse
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
                "Random Forest": fit_random_forest
            }
            plot_funcs = {
                "Polynomial": plot_fit,
                "Exponential": plot_fit,
                "Logarithmic": plot_fit,
                "Compound Poly+Log": plot_fit,
                "Spline": plot_fit,
                "Savitzky-Golay": plot_fit,
                "LOWESS": plot_fit,
                "Exponential Smoothing": plot_fit,
                "Gaussian Smoothing": plot_fit,
                "Wavelet Denoising": plot_fit,
                "Random Forest": plot_random_forest
            }

            if mode == "Curve Fit":
                # Display message for logarithmic and compound methods
                if st.session_state.get('method') in ["Logarithmic", "Compound Poly+Log"]:
                    st.info("Points with x ≤ 0 will be ignored for Logarithmic and Compound Poly+Log methods.")

                # Suggest best method (only if user selects "Yes")
                if show_suggestions == "Yes":
                    st.subheader("Suggested Best Method (Based on Adjusted R²)")
                    st.markdown("Note: Only Polynomial, Exponential, Logarithmic, and Compound Poly+Log are compared for best method.")
                    suggestions = []
                    for line_name, x, y, has_duplicates, has_invalid_x in lines:
                        if len(x) > 1:
                            try:
                                best_method, coeffs, r2, adj_r2, desc, _ = suggest_best_method(x, y, has_invalid_x)
                                suggestions.append((line_name, best_method, r2, adj_r2, desc))
                                st.write(f"Line '{line_name}': Best method = {best_method} ({desc}), R² = {r2:.4f}, Adjusted R² = {adj_r2:.4f}")
                            except ValueError as e:
                                st.warning(f"Line '{line_name}': Skipped in suggestion due to error: {str(e)}")

                # Choose method
                method_options = ["Polynomial", "Exponential", "Logarithmic", "Compound Poly+Log", "Spline", 
                                  "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising", "Random Forest"]
                method = st.selectbox("Choose Fitting Method", method_options, key="method")

                params = {}
                min_points = 3  # Default for most
                if method == "Polynomial":
                    params['degree'] = st.number_input("Polynomial Degree", min_value=1, max_value=10, value=2)
                    min_points = params['degree'] + 1
                elif method == "Spline":
                    params['degree'] = st.number_input("Spline Degree (1-5, 3=cubic)", min_value=1, max_value=5, value=3)
                    min_points = params['degree'] + 1
                elif method == "Savitzky-Golay":
                    params['window'] = st.number_input("Window length (odd number >=3)", min_value=3, max_value=101, value=5, step=2)
                    params['polyorder'] = st.number_input("Polynomial order", min_value=0, max_value=5, value=2)
                    min_points = params['window']
                elif method == "LOWESS":
                    params['frac'] = st.slider("Fraction of data for local fit", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
                elif method == "Exponential Smoothing":
                    params['alpha'] = st.slider("Smoothing factor (alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
                elif method == "Gaussian Smoothing":
                    params['sigma'] = st.number_input("Gaussian sigma", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                elif method == "Wavelet Denoising":
                    params['wavelet'] = st.selectbox("Wavelet family", options=['db4', 'haar', 'sym4', 'coif1'], index=0)
                    params['level'] = st.number_input("Decomposition level", min_value=1, max_value=5, value=1)
                    params['threshold'] = st.slider("Threshold factor", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
                elif method == "Random Forest":
                    params = random_forest_ui()
                    min_points = 3

                # Filter lines with enough points
                filtered_lines = [(name, x, y, has_duplicates, has_invalid_x) for name, x, y, has_duplicates, has_invalid_x in lines if len(x) >= min_points]
                if len(filtered_lines) < len(lines):
                    st.warning(f"Some lines skipped due to insufficient points for {method} (need at least {min_points}).")

                if st.button("Fit Curves"):
                    results = []
                    st.subheader("Fit Results and Graphs")
                    for line_name, x, y, has_duplicates, has_invalid_x in filtered_lines:
                        try:
                            if method in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"] and has_duplicates and not average_duplicates:
                                st.warning(f"Line '{line_name}': Skipped for {method} due to duplicate x values.")
                                continue
                            fit_func = fit_funcs[method]
                            coeffs, r2, model_desc = fit_func(x, y, **params)
                            if coeffs is not None:
                                result_row = [line_name] + coeffs + [r2]
                                results.append(result_row)
                                st.write(f"Line '{line_name}': {model_desc}, R² = {r2:.4f}")
                                # Use original x, y for plotting to include all points
                                plot_func = plot_funcs[method]
                                fig = plot_func(x, y, coeffs, method, params)
                                st.pyplot(fig)
                            else:
                                st.warning(f"Line '{line_name}': {model_desc}")
                        except ValueError as e:
                            st.warning(f"Line '{line_name}': Failed to fit {method}: {str(e)}")

                    if results:
                        max_coeffs = max(len(row) - 2 for row in results)
                        columns = ['Line Name'] + [f'param_{i}' for i in range(max_coeffs)] + ['R2']
                        output_df = pd.DataFrame(results, columns=columns)

                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            output_df.to_excel(writer, index=False, sheet_name='Fits')
                        output.seek(0)

                        st.download_button(
                            label="Download Output Excel",
                            data=output,
                            file_name=f"{method.lower().replace(' ', '_')}_fits.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.error("No fits could be performed.")

            elif mode == "Visual Comparison with Graphs":
                st.subheader("Visual Comparison with Graphs")
                st.markdown("This mode shows plots for polynomials (degrees 1-10) and all other methods using default parameters for visual inspection.")

                # Select all or random n lines
                comparison_type = st.radio("Compare", ["All lines", "Random n lines"])
                if comparison_type == "Random n lines":
                    n = st.number_input("Number of random lines", min_value=1, max_value=len(lines), value=min(5, len(lines)))
                    selected_lines = random.sample(lines, min(n, len(lines)))
                else:
                    selected_lines = lines

                # Define all methods with default params
                method_params_list = []
                for deg in range(1, 11):
                    method_params_list.append(("Polynomial", {"degree": deg}))
                method_params_list.extend([
                    ("Exponential", {}),
                    ("Logarithmic", {}),
                    ("Compound Poly+Log", {}),
                    ("Spline", {"degree": 3}),
                    ("Savitzky-Golay", {"window": 5, "polyorder": 2}),
                    ("LOWESS", {"frac": 0.3}),
                    ("Exponential Smoothing", {"alpha": 0.5}),
                    ("Gaussian Smoothing", {"sigma": 1.0}),
                    ("Wavelet Denoising", {"wavelet": "db4", "level": 1, "threshold": 0.1}),
                    ("Random Forest", {"n_estimators": 100})
                ])

                if st.button("Compare All Methods"):
                    results = []
                    st.subheader("Visual Comparison Results")
                    for line_name, x, y, has_duplicates, has_invalid_x in selected_lines:
                        st.markdown(f"### Line: {line_name}")
                        if len(x) < 3:
                            st.warning(f"Line '{line_name}': Skipped due to insufficient points (need at least 3).")
                            continue
                        for method, params in method_params_list:
                            min_points = 3
                            if method == "Polynomial":
                                min_points = params['degree'] + 1
                            elif method == "Spline":
                                min_points = params['degree'] + 1
                            elif method == "Savitzky-Golay":
                                min_points = params['window']
                            if len(x) < min_points:
                                st.warning(f"Line '{line_name}' - {method}: Skipped due to insufficient points (need {min_points}).")
                                continue
                            try:
                                if method in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"] and has_duplicates and not average_duplicates:
                                    st.warning(f"Line '{line_name}' - {method}: Skipped due to duplicate x values.")
                                    continue
                                fit_func = fit_funcs[method]
                                coeffs, r2, model_desc = fit_func(x, y, **params)
                                if coeffs is not None:
                                    result_row = [line_name, model_desc] + coeffs + [r2]
                                    results.append(result_row)
                                    st.write(f"{model_desc}, R² = {r2:.4f}")
                                    plot_func = plot_funcs[method]
                                    fig = plot_func(x, y, coeffs, method, params)
                                    st.pyplot(fig)
                                else:
                                    st.warning(f"Line '{line_name}' - {method}: {model_desc}")
                            except ValueError as e:
                                st.warning(f"Line '{line_name}' - {method}: Failed to fit: {str(e)}")

                    if results:
                        # Create DataFrame with method description included
                        max_coeffs = max(len(row) - 3 for row in results)  # Subtract line_name, model_desc, r2
                        columns = ['Line Name', 'Method'] + [f'param_{i}' for i in range(max_coeffs)] + ['R2']
                        output_df = pd.DataFrame(results, columns=columns)

                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            output_df.to_excel(writer, index=False, sheet_name='Comparison_Fits')
                        output.seek(0)

                        st.download_button(
                            label="Download Comparison Excel",
                            data=output,
                            file_name="comparison_fits.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.error("No fits could be performed in Visual Comparison mode.")

            elif mode == "Download Smoothed Data":
                st.subheader("Download Smoothed Data")
                st.markdown("Select a fitting method and parameters to generate smoothed curves (x_smooth, y_smooth) for each line, then download the data in Excel format matching the input structure.")

                # Choose method
                method_options = ["Polynomial", "Exponential", "Logarithmic", "Compound Poly+Log", "Spline", 
                                  "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising", "Random Forest"]
                method = st.selectbox("Choose Fitting Method", method_options, key="method")

                # Get parameters and number of points
                params = smooth_data_ui()
                params['average_duplicates'] = average_duplicates  # Pass average_duplicates to handle duplicates

                if st.button("Generate and Download Smoothed Data"):
                    smoothed_results = generate_smoothed_data(lines, method, params, fit_funcs)
                    output = create_smoothed_excel(smoothed_results)

                    for line_name, _, _, error_message in smoothed_results:
                        if error_message:
                            st.warning(f"Line '{line_name}': {error_message}")

                    st.download_button(
                        label="Download Smoothed Data Excel",
                        data=output,
                        file_name=f"{method.lower().replace(' ', '_')}_smoothed_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            elif mode == "Constrained Optimization":
                st.subheader("Constrained Optimization")
                st.markdown("Select a fitting method and specify restrictive points that the curve must pass through. The coefficients will be optimized to minimize MSE while satisfying the constraints.")

                # Get parameters once (outside the loop) using a sample line's data
                if lines:
                    sample_line_name, sample_x, sample_y, _, _ = lines[0]
                    params = constrained_optimization_ui(sample_x, sample_y, line_name="global_params")
                    method = params.pop('method')
                else:
                    st.error("No lines available to initialize parameters.")
                    params = {'method': 'Polynomial', 'degree': 3, 'constraints': [], 'lambda_reg': 1.0, 'show_diagnostics': False}
                    method = 'Polynomial'

                if st.button("Optimize Coefficients"):
                    results = []
                    st.subheader("Optimization Results and Graphs")
                    for line_name, x, y, has_duplicates, has_invalid_x in lines:
                        try:
                            # Use the globally defined params, but pass x, y for auto-constraint calculation
                            line_params = params.copy()
                            line_params['constraints'] = constrained_optimization_ui(x, y, line_name=line_name)['constraints']  # Update constraints per line
                            coeffs, r2, model_desc = optimize_coefficients(x, y, method, line_params)
                            if coeffs is not None:
                                result_row = [line_name, model_desc] + coeffs + [r2]
                                results.append((line_name, coeffs, r2, model_desc))
                                st.write(f"Line '{line_name}': {model_desc}, R² = {r2:.4f}")
                                fig = plot_constrained_fit(x, y, coeffs, method, line_params)
                                st.pyplot(fig)
                            else:
                                st.warning(f"Line '{line_name}': {model_desc}")
                        except ValueError as e:
                            st.warning(f"Line '{line_name}': Failed to optimize {method}: {str(e)}")

                    if results:
                        output = create_constrained_excel(results)
                        st.download_button(
                            label="Download Optimized Coefficients Excel",
                            data=output,
                            file_name="constrained_optimization_fits.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.error("No optimizations could be performed.")

            elif mode == "Outlier Detection and Cleaning":
                st.subheader("Outlier Detection and Cleaning")
                st.markdown("Select an outlier detection method or specify the number of outliers to remove, then fit curves to the cleaned data.")

                # Get parameters
                params = outlier_cleaner_ui()
                method = params.pop('fit_method')
                detection_method = params['detection_method']
                min_points = 3
                if method == "Polynomial":
                    min_points = params.get('degree', 2) + 1
                elif method == "Spline":
                    min_points = params.get('degree', 3) + 1
                elif method == "Savitzky-Golay":
                    min_points = params.get('window', 5)

                # Filter lines with enough points
                filtered_lines = [(name, x, y, has_duplicates, has_invalid_x) for name, x, y, has_duplicates, has_invalid_x in lines if len(x) >= min_points]
                if len(filtered_lines) < len(lines):
                    st.warning(f"Some lines skipped due to insufficient points for {method} (need at least {min_points}).")

                if st.button("Detect Outliers and Fit"):
                    cleaned_results = []
                    fit_results = []
                    st.subheader("Outlier Detection and Fit Results")
                    for line_name, x, y, has_duplicates, has_invalid_x in filtered_lines:
                        try:
                            # Detect outliers
                            if method in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"] and has_duplicates and not average_duplicates:
                                st.warning(f"Line '{line_name}': Skipped for {method} due to duplicate x values.")
                                cleaned_results.append((line_name, None, None, f"Skipped due to duplicate x values for {method}"))
                                continue
                            fit_func = fit_funcs[method]
                            mask, outlier_indices = detect_outliers(x, y, detection_method, params, fit_func if detection_method == "Fixed Number" else None)
                            if mask is None:
                                st.warning(f"Line '{line_name}': {outlier_indices}")
                                cleaned_results.append
