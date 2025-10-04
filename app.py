# app.py - Main Streamlit app for curve fitting
# Run with: streamlit run app.py
# Requirements: streamlit, pandas, numpy, scikit-learn, scipy, xlsxwriter, matplotlib, openpyxl, statsmodels, pywavelets
# Install via: pip install -r requirements.txt

import streamlit as st
import pandas as pd
import io
import random
from utils import parse_excel, suggest_best_method, fit_polynomial, fit_exponential, fit_logarithmic, fit_compound_poly_log, fit_spline, fit_savgol, fit_lowess, fit_exponential_smoothing, fit_gaussian_smoothing, fit_wavelet_denoising, fit_sine, fit_cosine, fit_tangent, fit_cotangent, fit_sinh, fit_cosh, fit_tanh
from plotting import plot_fit
from random_forest import fit_random_forest, plot_random_forest, random_forest_ui
from smooth_data import smooth_data_ui, generate_smoothed_data, create_smoothed_excel
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
        "Outlier Detection Guide"
    ]
    selected_guide = st.selectbox("Select a Guide", guide_options)

    if selected_guide == "General Instructions":
        st.markdown("""
        1. Upload an Excel file with line data: Line name in column A (B empty), then x in A, y in B below it (next line after an empty row, similarly).
        2. Choose a mode: Curve Fit (single method with parameters), Visual Comparison with Graphs (compare all methods), Download Smoothed Data (generate smoothed curves), or Outlier Detection and Cleaning (remove outliers and fit).
        3. For Curve Fit, select a fitting method and parameters. For Visual Comparison, choose all lines or n random lines to compare methods. For Download Smoothed Data, select a method, parameters, and number of points. For Outlier Detection, choose an outlier detection method or number of outliers.
        4. Optionally enable averaging of y values for duplicate x values (enables splines and other smoothing methods).
        5. Optionally view suggestions for the best method based on Adjusted R² (only Polynomial, Exponential, Logarithmic, and Compound Poly+Log compared).
        6. Fit curves, view graphs, or download the output Excel.
        """)
    elif selected_guide == "Excel Format Guide":
        st.markdown("""
        **Output Excel Guide (Curve Fit, Visual Comparison):**
        - **Columns**: 'Line Name', then coefficients/parameters, followed by R².
        - **Smoothed Data and Outlier Cleaning Excel Format**: Similar to input: Line name in column A (B empty), then x in A, y in B below it, empty rows, next line, etc.
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
        - **Sine**: a, b, c, d for y = a * sin(b*x + c) + d. Requires at least 4 points. Use frequency_guess for initial b (angular frequency).
        - **Cosine**: a, b, c, d for y = a * cos(b*x + c) + d. Requires at least 4 points. Use frequency_guess for initial b (angular frequency).
        - **Tangent**: a, b, c, d for y = a * tan(b*x + c) + d. Requires at least 4 points. Use frequency_guess for initial b; may fail near singularities.
        - **Cotangent**: a, b, c, d for y = a * cot(b*x + c) + d. Requires at least 4 points. Use frequency_guess for initial b; may fail near singularities.
        - **Sinh**: a, b, c, d for y = a * sinh(b*x + c) + d. Requires at least 4 points. Use scaling_guess for initial b.
        - **Cosh**: a, b, c, d for y = a * cosh(b*x + c) + d. Requires at least 4 points. Use scaling_guess for initial b.
        - **Tanh**: a, b, c, d for y = a * tanh(b*x + c) + d. Requires at least 4 points. Use scaling_guess for initial b.
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
                                            "Outlier Detection and Cleaning"])

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
                "Random Forest": fit_random_forest,
                "Sine": fit_sine,
                "Cosine": fit_cosine,
                "Tangent": fit_tangent,
                "Cotangent": fit_cotangent,
                "Sinh": fit_sinh,
                "Cosh": fit_cosh,
                "Tanh": fit_tanh
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
                "Random Forest": plot_random_forest,
                "Sine": plot_fit,
                "Cosine": plot_fit,
                "Tangent": plot_fit,
                "Cotangent": plot_fit,
                "Sinh": plot_fit,
                "Cosh": plot_fit,
                "Tanh": plot_fit
            }

            if mode == "Curve Fit":
                # Display message for logarithmic and compound methods
                if st.session_state.get('method') in ["Logarithmic", "Compound Poly+Log"]:
                    st.info("Points with x ≤ 0 will be ignored for Logarithmic and Compound Poly+Log methods.")
                if st.session_state.get('method') in ["Tangent", "Cotangent"]:
                    st.info("Tangent and Cotangent fits may fail near singularities (poles). Ensure data avoids these regions or adjust frequency_guess.")

                # Suggest best method (only if user selects "Yes")
                if show_suggestions == "Yes":
                    st.subheader("Suggested Best Method (Based on Adjusted R²)")
                    st.markdown("Note: Only Polynomial, Exponential, Logarithmic, and Compound Poly+Log are compared.")
                    for line_name, x, y, _, has_invalid_x in lines:
                        best_method, coeffs, r2, adj_r2, model_desc, params = suggest_best_method(x, y, has_invalid_x)
                        st.write(f"Line '{line_name}': Best method = {model_desc}, Adjusted R² = {adj_r2:.4f}")

                st.subheader("Curve Fit")
                st.markdown("Select a line and a fitting method to fit a curve and view the results.")

                # Choose line
                line_names = [line[0] for line in lines]
                selected_line = st.selectbox("Select Line", line_names)

                # Choose method
                method_options = ["Polynomial", "Exponential", "Logarithmic", "Compound Poly+Log", "Spline", 
                                  "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", 
                                  "Wavelet Denoising", "Random Forest", "Sine", "Cosine", "Tangent", "Cotangent", 
                                  "Sinh", "Cosh", "Tanh"]
                method = st.selectbox("Choose Fitting Method", method_options, key="method")

                # Get parameters
                params = {}
                if method == "Polynomial":
                    params['degree'] = st.number_input("Polynomial Degree", min_value=1, max_value=10, value=2)
                elif method == "Spline":
                    params['degree'] = st.number_input("Spline Degree (1-5, 3=cubic)", min_value=1, max_value=5, value=3)
                elif method == "Savitzky-Golay":
                    params['window'] = st.number_input("Window length (odd number >=3)", min_value=3, max_value=101, value=5, step=2)
                    params['polyorder'] = st.number_input("Polynomial order", min_value=0, max_value=5, value=2)
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
                elif method in ["Sine", "Cosine", "Tangent", "Cotangent"]:
                    params['frequency_guess'] = st.number_input("Frequency Guess (b initial value)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                elif method in ["Sinh", "Cosh", "Tanh"]:
                    params['scaling_guess'] = st.number_input("Scaling Guess (b initial value)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

                # Fit and display results
                if st.button("Fit Curve"):
                    for line_name, x, y, has_duplicates, has_invalid_x in lines:
                        if line_name == selected_line:
                            if method in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"] and has_duplicates and not average_duplicates:
                                st.error(f"Cannot apply {method} to line '{line_name}' due to duplicate x values. Enable 'Average y values for duplicate x values' or choose another method.")
                                break
                            min_points = 3
                            if method == "Polynomial":
                                min_points = params.get('degree', 2) + 1
                            elif method == "Spline":
                                min_points = params.get('degree', 3) + 1
                            elif method == "Savitzky-Golay":
                                min_points = params.get('window', 5)
                            elif method in ["Sine", "Cosine", "Tangent", "Cotangent", "Sinh", "Cosh", "Tanh"]:
                                min_points = 4
                            if len(x) < min_points:
                                st.error(f"Insufficient points ({len(x)}) for {method}. Need at least {min_points} points.")
                                break
                            coeffs, r2, model_desc = fit_funcs[method](x, y, **params)
                            if coeffs is None:
                                st.error(f"Fit failed for line '{line_name}': {model_desc}")
                            else:
                                st.write(f"Line: {line_name}")
                                st.write(f"Model: {model_desc}")
                                st.write(f"Coefficients/Parameters: {coeffs}")
                                st.write(f"R²: {r2:.4f}")
                                fig = plot_funcs[method](x, y, coeffs, method, params)
                                st.pyplot(fig)

                # Export results to Excel
                if st.button("Export Curve Fit Results"):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        results = []
                        for line_name, x, y, has_duplicates, has_invalid_x in lines:
                            if method in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"] and has_duplicates and not average_duplicates:
                                results.append([line_name] + ['Failed: Duplicate x values'] + ['-'])
                                continue
                            min_points = 3
                            if method == "Polynomial":
                                min_points = params.get('degree', 2) + 1
                            elif method == "Spline":
                                min_points = params.get('degree', 3) + 1
                            elif method == "Savitzky-Golay":
                                min_points = params.get('window', 5)
                            elif method in ["Sine", "Cosine", "Tangent", "Cotangent", "Sinh", "Cosh", "Tanh"]:
                                min_points = 4
                            if len(x) < min_points:
                                results.append([line_name] + [f'Failed: Insufficient points (need {min_points})'] + ['-'])
                                continue
                            coeffs, r2, model_desc = fit_funcs[method](x, y, **params)
                            if coeffs is None:
                                results.append([line_name] + [model_desc] + ['-'])
                            else:
                                results.append([line_name] + coeffs + [r2])
                        max_cols = max(len(row) for row in results)
                        columns = ['Line Name'] + [f'Coeff/Param {i+1}' for i in range(max_cols-2)] + ['R²']
                        df = pd.DataFrame(results, columns=columns[:len(results[0])])
                        df.to_excel(writer, index=False, sheet_name='Fit_Results')
                    output.seek(0)
                    st.download_button("Download Curve Fit Results", output, file_name="curve_fit_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            elif mode == "Visual Comparison with Graphs":
                st.subheader("Visual Comparison with Graphs")
                st.markdown("Compare all fitting methods on selected lines (all or n random).")
                line_selection = st.radio("Select Lines", ["All Lines", "Random Lines"], index=0)
                if line_selection == "Random Lines":
                    n_lines = st.number_input("Number of Random Lines", min_value=1, max_value=len(lines), value=min(5, len(lines)))
                    selected_lines = random.sample(lines, n_lines)
                else:
                    selected_lines = lines

                if st.button("Generate Comparison"):
                    for line_name, x, y, has_duplicates, has_invalid_x in selected_lines:
                        st.subheader(f"Line: {line_name}")
                        for method in method_options:
                            if method in ["Logarithmic", "Compound Poly+Log"] and has_invalid_x:
                                st.write(f"{method}: Skipped (x ≤ 0 not allowed)")
                                continue
                            if method in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"] and has_duplicates and not average_duplicates:
                                st.write(f"{method}: Skipped due to duplicate x values")
                                continue
                            min_points = 3
                            if method == "Polynomial":
                                min_points = params.get('degree', 2) + 1
                            elif method == "Spline":
                                min_points = params.get('degree', 3) + 1
                            elif method == "Savitzky-Golay":
                                min_points = params.get('window', 5)
                            elif method in ["Sine", "Cosine", "Tangent", "Cotangent", "Sinh", "Cosh", "Tanh"]:
                                min_points = 4
                            if len(x) < min_points:
                                st.write(f"{method}: Skipped (insufficient points, need {min_points})")
                                continue
                            # Use default params for comparison
                            temp_params = {'degree': 2, 'window': 5, 'polyorder': 2, 'frac': 0.3, 'alpha': 0.5, 'sigma': 1.0, 
                                           'wavelet': 'db4', 'level': 1, 'threshold': 0.1, 'n_estimators': 100, 
                                           'frequency_guess': 1.0, 'scaling_guess': 1.0}
                            coeffs, r2, model_desc = fit_funcs[method](x, y, **{k: v for k, v in temp_params.items() if k in fit_funcs[method].__code__.co_varnames})
                            if coeffs is None:
                                st.write(f"{method}: {model_desc}")
                            else:
                                st.write(f"{method}: {model_desc}, R² = {r2:.4f}")
                                fig = plot_funcs[method](x, y, coeffs, method, temp_params)
                                st.pyplot(fig)

            elif mode == "Download Smoothed Data":
                st.subheader("Download Smoothed Data")
                st.markdown("Select a fitting method and parameters to generate smoothed data for all lines.")
                method = st.selectbox("Choose Fitting Method", method_options, key="smooth_method")
                if method in ["Logarithmic", "Compound Poly+Log"]:
                    st.info("Points with x ≤ 0 will be ignored for Logarithmic and Compound Poly+Log methods.")
                if method in ["Tangent", "Cotangent"]:
                    st.info("Tangent and Cotangent fits may fail near singularities (poles). Ensure data avoids these regions or adjust frequency_guess.")
                params = smooth_data_ui()

                if st.button("Generate and Download Smoothed Data"):
                    results = generate_smoothed_data(lines, method, params, fit_funcs)
                    output = create_smoothed_excel(results)
                    st.download_button("Download Smoothed Data", output, file_name="smoothed_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    for line_name, x_smooth, y_smooth, error_message in results:
                        if error_message:
                            st.write(f"Line '{line_name}': {error_message}")
                        else:
                            st.write(f"Line '{line_name}': Smoothed data generated ({len(x_smooth)} points)")

            elif mode == "Outlier Detection and Cleaning":
                st.subheader("Outlier Detection and Cleaning")
                params = outlier_cleaner_ui()
                method = params['fit_method']
                if method in ["Logarithmic", "Compound Poly+Log"]:
                    st.info("Points with x ≤ 0 will be ignored for Logarithmic and Compound Poly+Log methods.")
                if method in ["Tangent", "Cotangent"]:
                    st.info("Tangent and Cotangent fits may fail near singularities (poles). Ensure data avoids these regions or adjust frequency_guess.")

                if st.button("Detect Outliers and Fit"):
                    cleaned_results = []
                    for line_name, x, y, has_duplicates, has_invalid_x in lines:
                        if method in ["Spline", "Savitzky-Golay", "LOWESS", "Exponential Smoothing", "Gaussian Smoothing", "Wavelet Denoising"] and has_duplicates and not average_duplicates:
                            st.write(f"Line '{line_name}': Skipped for {method} due to duplicate x values")
                            cleaned_results.append((line_name, None, None, f"Skipped for {method} due to duplicate x values"))
                            continue
                        if method in ["Logarithmic", "Compound Poly+Log"] and has_invalid_x:
                            st.write(f"Line '{line_name}': Skipped for {method} (x ≤ 0 not allowed)")
                            cleaned_results.append((line_name, None, None, f"Skipped for {method} (x ≤ 0 not allowed)"))
                            continue
                        min_points = 3
                        if method == "Polynomial":
                            min_points = params.get('degree', 2) + 1
                        elif method == "Spline":
                            min_points = params.get('degree', 3) + 1
                        elif method == "Savitzky-Golay":
                            min_points = params.get('window', 5)
                        elif method in ["Sine", "Cosine", "Tangent", "Cotangent", "Sinh", "Cosh", "Tanh"]:
                            min_points = 4
                        if len(x) < min_points:
                            st.write(f"Line '{line_name}': Skipped (insufficient points, need {min_points})")
                            cleaned_results.append((line_name, None, None, f"Skipped (insufficient points, need {min_points})"))
                            continue
                        mask, outlier_indices = detect_outliers(x, y, params['detection_method'], params, fit_funcs[method])
                        if mask is None:
                            st.write(f"Line '{line_name}': {outlier_indices}")
                            cleaned_results.append((line_name, None, None, outlier_indices))
                            continue
                        x_clean = x[mask]
                        y_clean = y[mask]
                        if len(x_clean) < min_points:
                            st.write(f"Line '{line_name}': Not enough points after outlier removal")
                            cleaned_results.append((line_name, None, None, "Not enough points after outlier removal"))
                            continue
                        coeffs, r2, model_desc = fit_funcs[method](x_clean, y_clean, **{k: v for k, v in params.items() if k not in ['detection_method', 'fit_method']})
                        if coeffs is None:
                            st.write(f"Line '{line_name}': Fit failed - {model_desc}")
                            cleaned_results.append((line_name, None, None, model_desc))
                        else:
                            st.write(f"Line '{line_name}': {model_desc}, R² = {r2:.4f}, Outliers removed: {len(outlier_indices)}")
                            fig = plot_cleaned_data(x, y, mask, coeffs, method, params)
                            st.pyplot(fig)
                            cleaned_results.append((line_name, x_clean, y_clean, None))

                    if any(r[3] is None for r in cleaned_results):
                        output = create_cleaned_excel(cleaned_results)
                        st.download_button("Download Cleaned Data", output, file_name="cleaned_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
