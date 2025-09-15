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

st.title("Curve Fitting App")

st.markdown("""
### Instructions
1. Upload an Excel file with line data: Line name in column A (B empty), then x in A, y in B below it 
(next line after an empty row, similarly)
2. Choose a mode: Standard (single method with parameters) or Visual Comparison (compare all methods).
3. For Standard mode, select a fitting method and parameters. For Visual Comparison, choose all lines or n random lines to compare polynomials (degrees 1-10) and other methods.
4. Optionally enable averaging of y values for duplicate x values (enables splines and other smoothing methods for all lines).
5. View suggestions for the best overall method based on Adjusted R² (only Polynomial, Exponential, Logarithmic, and Compound Poly+Log are compared).
6. Fit curves, view graphs, and download the output Excel.

**Output Excel Guide:**
- **Columns**: 'Line Name', then coefficients/parameters, followed by R².
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

            # Mode selection: Standard or Visual Comparison
            mode = st.radio("Select Mode", ["Standard", "Visual Comparison"])

            if mode == "Standard":
                # Display message for logarithmic and compound methods
                if st.session_state.get('method') in ["Logarithmic", "Compound Poly+Log"]:
                    st.info("Points with x ≤ 0 will be ignored for Logarithmic and Compound Poly+Log methods.")

                # Suggest best method
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
                                  "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"]
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
                else:
                    min_points = 3

                # Filter lines with enough points
                filtered_lines = [(name, x, y, has_duplicates, has_invalid_x) for name, x, y, has_duplicates, has_invalid_x in lines if len(x) >= min_points]
                if len(filtered_lines) < len(lines):
                    st.warning(f"Some lines skipped due to insufficient points for {method} (need at least {min_points}).")

                if st.button("Fit Curves"):
                    results = []
                    st.subheader("Fit Results and Graphs")
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
                        "Wavelet Denoising": fit_wavelet_denoising
                    }
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
                                fig = plot_fit(x, y, coeffs, method, params)
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

            elif mode == "Visual Comparison":
                st.subheader("Visual Comparison Mode")
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
                    ("Wavelet Denoising", {"wavelet": "db4", "level": 1, "threshold": 0.1})
                ])

                # Fit functions map
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
                    "Wavelet Denoising": fit_wavelet_denoising
                }

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
                                    # Store results with method name for Excel output
                                    result_row = [line_name, model_desc] + coeffs + [r2]
                                    results.append(result_row)
                                    st.write(f"{model_desc}, R² = {r2:.4f}")
                                    fig = plot_fit(x, y, coeffs, method, params)
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

    except ValueError as e:
        st.error(f"Failed to read Excel file: {str(e)}")
