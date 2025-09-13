# app.py - Main Streamlit app for curve fitting
# Run with: streamlit run app.py
# Requirements: streamlit, pandas, numpy, scikit-learn, scipy, xlsxwriter, matplotlib, openpyxl
# Install via: pip install -r requirements.txt

import streamlit as st
import pandas as pd
import io
from utils import parse_excel, suggest_best_method, fit_polynomial, fit_exponential, fit_logarithmic, fit_compound_poly_log, fit_spline
from plotting import plot_fit

st.title("Curve Fitting App")

st.markdown("""
### Instructions
1. Upload an Excel file with line data: Line name in column A (B empty), then x in A, y in B below it.
2. Choose a fitting method (and parameters like degree).
3. View suggestions for the best overall method based on Adjusted R².
4. Fit curves, view graphs, and download the output Excel.

**Output Excel Guide:**
- **Columns**: 'Line Name', then coefficients/parameters, followed by R².
- **Polynomial**: Coefficients from highest degree to constant (e.g., degree 2: a_2, a_1, a_0 for y = a_2*x^2 + a_1*x + a_0).
- **Exponential**: a, b, c for y = a * exp(b*x) + c.
- **Logarithmic**: a, b for y = a * log(x) + b.
- **Compound Poly+Log**: a, b, c for y = a*x^2 + b*log(x) + c.
- **Spline**: Spline coefficients followed by knots (variable length; interpret with scipy.interpolate.UnivariateSpline). Note: Requires strictly increasing x values.
""")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file:
    try:
        lines, skipped_lines = parse_excel(uploaded_file)
        if not lines:
            st.error("No valid lines found in the file.")
        else:
            st.success(f"Found {len(lines)} valid lines.")
            if skipped_lines:
                st.warning(f"Skipped {len(skipped_lines)} lines due to duplicate x values (not suitable for splines): {', '.join(skipped_lines)}")

            # Suggest best method
            st.subheader("Suggested Best Method (Based on Adjusted R²)")
            suggestions = []
            for line_name, x, y, has_duplicates in lines:
                if len(x) > 1:
                    try:
                        best_method, coeffs, r2, adj_r2, desc, _ = suggest_best_method(x, y, has_duplicates)
                        suggestions.append((line_name, best_method, r2, adj_r2, desc))
                        st.write(f"Line '{line_name}': Best method = {best_method} ({desc}), R² = {r2:.4f}, Adjusted R² = {adj_r2:.4f}")
                    except ValueError as e:
                        st.warning(f"Line '{line_name}': Skipped in suggestion due to error: {str(e)}")

            # Choose method
            method = st.selectbox("Choose Fitting Method", ["Polynomial", "Exponential", "Logarithmic", "Compound Poly+Log", "Spline"])

            params = {}
            if method == "Polynomial":
                params['degree'] = st.number_input("Polynomial Degree", min_value=1, max_value=10, value=2)
                min_points = params['degree'] + 1
            elif method == "Spline":
                params['degree'] = st.number_input("Spline Degree (1-5, 3=cubic)", min_value=1, max_value=5, value=3)
                min_points = params['degree'] + 1
            else:
                min_points = 3

            # Filter lines with enough points
            filtered_lines = [(name, x, y, has_duplicates) for name, x, y, has_duplicates in lines if len(x) >= min_points]
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
                    "Spline": fit_spline
                }
                for line_name, x, y, has_duplicates in filtered_lines:
                    try:
                        if method == "Spline" and has_duplicates:
                            st.warning(f"Line '{line_name}': Skipped for Spline due to duplicate x values.")
                            continue
                        fit_func = fit_funcs[method]
                        coeffs, r2, model_desc = fit_func(x, y, **params)
                        if coeffs is not None:
                            result_row = [line_name] + coeffs + [r2]
                            results.append(result_row)
                            st.write(f"Line '{line_name}': {model_desc}, R² = {r2:.4f}")
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
    except ValueError as e:
        st.error(f"Failed to read Excel file: {str(e)}")
