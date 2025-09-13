# app.py - Main Streamlit app for curve fitting
# Run with: streamlit run app.py
# Requirements: streamlit, pandas, numpy, scikit-learn, scipy, xlsxwriter, matplotlib
# Install via: pip install streamlit pandas numpy scikit-learn scipy xlsxwriter matplotlib

import streamlit as st
import pandas as pd
import io
from utils import parse_excel, suggest_best_poly_degree, fit_polynomial, fit_exponential, fit_logarithmic, fit_compound_poly_log, fit_spline
from plotting import plot_fit

st.title("Curve Fitting App")

st.markdown("""
### Instructions
1. Upload an Excel file with line data: Line name in column A (B empty), then x in A, y in B below it.
2. Choose a fitting method (and degree for Polynomial).
3. View suggestions for Polynomial degrees.
4. Fit curves, view graphs, and download the output Excel.

**Output Excel Guide:**
- **Columns**: 'Line Name', then coefficients/parameters, followed by R².
- **Polynomial**: Coefficients from highest degree to constant (e.g., degree 2: a_2, a_1, a_0 for y = a_2*x^2 + a_1*x + a_0).
- **Exponential**: a, b, c for y = a * exp(b*x) + c.
- **Logarithmic**: a, b for y = a * log(x) + b.
- **Compound Poly+Log**: a, b, c for y = a*x^2 + b*log(x) + c.
- **Spline**: Spline coefficients followed by knots (variable length; interpret with scipy.interpolate.UnivariateSpline).
""")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file:
    lines = parse_excel(uploaded_file)
    if not lines:
        st.error("No valid lines found in the file.")
    else:
        st.success(f"Found {len(lines)} valid lines.")

        # Suggest best polynomial degrees
        st.subheader("Suggested Polynomial Degrees")
        suggestions = []
        for line_name, x, y in lines:
            if len(x) > 1:
                best_degree, best_r2 = suggest_best_poly_degree(x, y)
                suggestions.append((line_name, best_degree, best_r2))
                st.write(f"Line '{line_name}': Suggested degree {best_degree} (R² = {best_r2:.4f})")

        # Choose method
        method = st.selectbox("Choose Fitting Method", ["Polynomial", "Exponential", "Logarithmic", "Compound Poly+Log", "Spline"])

        params = {}
        if method == "Polynomial":
            degree = st.number_input("Polynomial Degree", min_value=1, max_value=10, value=2)
            params['degree'] = degree
            min_points = degree + 1
        elif method == "Spline":
            min_points = 4  # For cubic spline
        else:
            min_points = 3  # For most nonlinear

        # Filter lines with enough points
        filtered_lines = [(name, x, y) for name, x, y in lines if len(x) >= min_points]
        if len(filtered_lines) < len(lines):
            st.warning(f"Some lines skipped due to insufficient points for {method} (need at least {min_points}).")

        if st.button("Fit Curves"):
            results = []
            st.subheader("Fit Results and Graphs")
            for line_name, x, y in filtered_lines:
                if method == "Polynomial":
                    coeffs, r2, model_desc = fit_polynomial(x, y, params['degree'])
                elif method == "Exponential":
                    coeffs, r2, model_desc = fit_exponential(x, y)
                elif method == "Logarithmic":
                    coeffs, r2, model_desc = fit_logarithmic(x, y)
                elif method == "Compound Poly+Log":
                    coeffs, r2, model_desc = fit_compound_poly_log(x, y)
                elif method == "Spline":
                    coeffs, r2, model_desc = fit_spline(x, y)
                
                if coeffs is not None:
                    result_row = [line_name] + coeffs + [r2]
                    results.append(result_row)
                    st.write(f"Line '{line_name}': {model_desc}, R² = {r2:.4f}")
                    # Plot the fit
                    fig = plot_fit(x, y, coeffs, method, params.get('degree', 3))
                    st.pyplot(fig)
                else:
                    st.write(f"Line '{line_name}': {model_desc}")

            if results:
                # Create DataFrame with dynamic columns
                max_coeffs = max(len(row) - 2 for row in results)  # -2 for name and r2
                columns = ['Line Name'] + [f'param_{i}' for i in range(max_coeffs)] + ['R2']
                output_df = pd.DataFrame(results, columns=columns)

                # Prepare download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    output_df.to_excel(writer, index=False, sheet_name='Fits')
                output.seek(0)

                st.download_button(
                    label="Download Output Excel",
                    data=output,
                    file_name=f"{method.lower()}_fits.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error("No fits could be performed.")
