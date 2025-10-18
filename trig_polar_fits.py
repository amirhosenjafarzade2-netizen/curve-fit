import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import sympy as sp

def trig_polar_ui():
    """
    Render UI for Trigonometric and Polar fitting mode.
    Returns: Dictionary of parameters including fit_type and specific parameters
    """
    params = {}
    fit_type_options = ["Trigonometric", "Polar"]
    params['fit_type'] = st.selectbox("Fit Type", fit_type_options)
    
    if params['fit_type'] == "Trigonometric":
        params['trig_model'] = st.selectbox(
            "Trigonometric Model",
            [
                "Cosine + Sine",
                "Tan + Cot",
                "Hyperbolic",
                "Inverse Trig",
                "Complex Trig",
                "Combined"
            ]
        )
        if params['trig_model'] in ["Cosine + Sine", "Hyperbolic", "Combined"]:
            params['n_terms'] = st.number_input("Number of terms", min_value=1, max_value=3, value=2, help="Number of terms (reduced for Combined to avoid overfitting)")
        elif params['trig_model'] == "Tan + Cot":
            params['scale'] = st.slider("Scale factor", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        elif params['trig_model'] == "Inverse Trig":
            params['scale'] = st.slider("Scale factor", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        elif params['trig_model'] == "Complex Trig":
            params['n_terms'] = st.number_input("Number of terms", min_value=1, max_value=3, value=2)
    
    elif params['fit_type'] == "Polar":
        params['polar_model'] = st.selectbox(
            "Polar Model",
            [
                "Power Spiral",
                "Rose Curve",
                "Archimedean Spiral",
                "Complex Polar",
                "Combined"
            ]
        )
        if params['polar_model'] == "Power Spiral":
            params['max_power'] = st.number_input("Maximum power", min_value=1, max_value=3, value=2)
        elif params['polar_model'] == "Rose Curve":
            params['n_petals'] = st.number_input("Number of petals", min_value=1, max_value=10, value=4)
        elif params['polar_model'] == "Archimedean Spiral":
            params['a'] = st.slider("Spiral constant", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        elif params['polar_model'] == "Complex Polar":
            params['n_terms'] = st.number_input("Number of terms", min_value=1, max_value=3, value=2)
        elif params['polar_model'] == "Combined":
            params['n_terms'] = st.number_input("Number of terms", min_value=1, max_value=2, value=2, help="Number of terms (reduced to avoid overfitting)")
    
    params['compare_modes'] = st.checkbox("Enable Visual Comparison of All Sub-Modes", value=False)
    return params

def fit_trigonometric(x, y, trig_model, params):
    """
    Fit a trigonometric model to the data.
    Returns: coefficients, R², description, sympy formula
    """
    try:
        if trig_model == "Cosine + Sine":
            def trig_model_func(x, *p):
                result = 0
                for i in range(params['n_terms']):
                    result += p[2*i] * np.cos(p[2*i+1] * x)
                    result += p[2*i + params['n_terms']*2] * np.sin(p[2*i + params['n_terms']*2 + 1] * x)
                result += p[-1]  # Constant term
                return result
            n_params = 4 * params['n_terms'] + 1
            p0 = [1.0] * n_params
            popt, _ = curve_fit(trig_model_func, x, y, p0=p0, maxfev=10000)
            y_pred = trig_model_func(x, *popt)
            x_sym = sp.Symbol('x')
            formula = 0
            for i in range(params['n_terms']):
                formula += popt[2*i] * sp.cos(popt[2*i+1] * x_sym)
                formula += popt[2*i + params['n_terms']*2] * sp.sin(popt[2*i + params['n_terms']*2 + 1] * x_sym)
            formula += popt[-1]
            desc = f"Cosine + Sine: {params['n_terms']} terms + constant"
        
        elif trig_model == "Tan + Cot":
            def trig_model_func(x, a, b, c, d):
                return a * np.tan(b * x) + c / np.tan(d * x)
            popt, _ = curve_fit(trig_model_func, x, y, p0=[params['scale'], 1.0, params['scale'], 1.0], maxfev=10000)
            y_pred = trig_model_func(x, *popt)
            x_sym = sp.Symbol('x')
            formula = popt[0] * sp.tan(popt[1] * x_sym) + popt[2] / sp.tan(popt[3] * x_sym)
            desc = "Tan + Cot: a*tan(b*x) + c/tan(d*x)"
        
        elif trig_model == "Hyperbolic":
            def trig_model_func(x, *p):
                result = 0
                for i in range(params['n_terms']):
                    result += p[2*i] * np.cosh(p[2*i+1] * x)
                    result += p[2*i + params['n_terms']*2] * np.sinh(p[2*i + params['n_terms']*2 + 1] * x)
                result += p[-1]
                return result
            n_params = 4 * params['n_terms'] + 1
            popt, _ = curve_fit(trig_model_func, x, y, p0=[1.0] * n_params, maxfev=10000)
            y_pred = trig_model_func(x, *popt)
            x_sym = sp.Symbol('x')
            formula = 0
            for i in range(params['n_terms']):
                formula += popt[2*i] * sp.cosh(popt[2*i+1] * x_sym)
                formula += popt[2*i + params['n_terms']*2] * sp.sinh(popt[2*i + params['n_terms']*2 + 1] * x_sym)
            formula += popt[-1]
            desc = f"Hyperbolic: {params['n_terms']} cosh/sinh terms + constant"
        
        elif trig_model == "Inverse Trig":
            def trig_model_func(x, a, b, c, d):
                return a * np.arctan(b * x) + c * np.arcsin(d * x / (1 + np.abs(d * x)))
            popt, _ = curve_fit(trig_model_func, x, y, p0=[params['scale'], 1.0, params['scale'], 1.0], maxfev=10000)
            y_pred = trig_model_func(x, *popt)
            x_sym = sp.Symbol('x')
            formula = popt[0] * sp.atan(popt[1] * x_sym) + popt[2] * sp.asin(popt[3] * x_sym / (1 + sp.Abs(popt[3] * x_sym)))
            desc = "Inverse Trig: a*arctan(b*x) + c*arcsin(d*x/(1+|d*x|))"
        
        elif trig_model == "Complex Trig":
            def trig_model_func(x, *p):
                result = 0
                for i in range(params['n_terms']):
                    result += p[3*i] * np.cos(p[3*i+1] * x) * np.tan(p[3*i+2] * x)
                result += p[-1]
                return result
            n_params = 3 * params['n_terms'] + 1
            popt, _ = curve_fit(trig_model_func, x, y, p0=[1.0] * n_params, maxfev=10000)
            y_pred = trig_model_func(x, *popt)
            x_sym = sp.Symbol('x')
            formula = 0
            for i in range(params['n_terms']):
                formula += popt[3*i] * sp.cos(popt[3*i+1] * x_sym) * sp.tan(popt[3*i+2] * x_sym)
            formula += popt[-1]
            desc = f"Complex Trig: {params['n_terms']} cos*tan terms + constant"
        
        elif trig_model == "Combined":
            def trig_model_func(x, *p):
                result = 0
                n_terms = params['n_terms']
                # Cosine + Sine terms
                for i in range(n_terms):
                    result += p[2*i] * np.cos(p[2*i+1] * x)
                    result += p[2*i + n_terms*2] * np.sin(p[2*i + n_terms*2 + 1] * x)
                # Hyperbolic terms
                for i in range(n_terms):
                    result += p[2*i + n_terms*4] * np.cosh(p[2*i + n_terms*4 + 1] * x)
                    result += p[2*i + n_terms*6] * np.sinh(p[2*i + n_terms*6 + 1] * x)
                result += p[-1]  # Constant term
                return result
            n_params = 8 * params['n_terms'] + 1
            popt, _ = curve_fit(trig_model_func, x, y, p0=[0.5] * n_params, maxfev=15000)
            y_pred = trig_model_func(x, *popt)
            x_sym = sp.Symbol('x')
            formula = 0
            for i in range(params['n_terms']):
                formula += popt[2*i] * sp.cos(popt[2*i+1] * x_sym)
                formula += popt[2*i + n_terms*2] * sp.sin(popt[2*i + n_terms*2 + 1] * x_sym)
                formula += popt[2*i + n_terms*4] * sp.cosh(popt[2*i + n_terms*4 + 1] * x_sym)
                formula += popt[2*i + n_terms*6] * sp.sinh(popt[2*i + n_terms*6 + 1] * x_sym)
            formula += popt[-1]
            desc = f"Combined Trig: {params['n_terms']} cos/sin/cosh/sinh terms + constant"
        
        r2 = r2_score(y, y_pred)
        formula_str = sp.latex(formula)
        return popt.tolist(), r2, desc, formula_str
    
    except Exception as e:
        return None, None, f"{trig_model} fit failed: {str(e)}", None

def fit_polar(x, y, polar_model, params):
    """
    Fit a polar model to the data (x, y treated as Cartesian, converted to polar).
    Returns: coefficients, R², description, sympy formula
    """
    try:
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        if polar_model == "Power Spiral":
            def polar_func(theta, *p):
                result = 0
                for i in range(params['max_power'] + 1):
                    result += p[i] * theta**i
                return result
            n_params = params['max_power'] + 1
            popt, _ = curve_fit(polar_func, theta, r, p0=[1.0] * n_params, maxfev=10000)
            r_pred = polar_func(theta, *popt)
            x_pred = r_pred * np.cos(theta)
            y_pred = r_pred * np.sin(theta)
            residuals = np.sqrt((x - x_pred)**2 + (y - y_pred)**2)
            r2 = 1 - np.sum(residuals**2) / np.sum((np.sqrt(x**2 + y**2) - np.mean(np.sqrt(x**2 + y**2)))**2)
            theta_sym = sp.Symbol('theta')
            formula = sum(popt[i] * theta_sym**i for i in range(n_params))
            desc = f"Power Spiral: r = {' + '.join(f'{popt[i]:.4f}*theta^{i}' for i in range(n_params))}"
            formula_str = sp.latex(formula)
        
        elif polar_model == "Rose Curve":
            def polar_func(theta, a, n):
                return a * np.cos(n * theta)
            popt, _ = curve_fit(polar_func, theta, r, p0=[np.max(r), params['n_petals']], maxfev=10000)
            r_pred = polar_func(theta, *popt)
            x_pred = r_pred * np.cos(theta)
            y_pred = r_pred * np.sin(theta)
            residuals = np.sqrt((x - x_pred)**2 + (y - y_pred)**2)
            r2 = 1 - np.sum(residuals**2) / np.sum((np.sqrt(x**2 + y**2) - np.mean(np.sqrt(x**2 + y**2)))**2)
            theta_sym = sp.Symbol('theta')
            formula = popt[0] * sp.cos(popt[1] * theta_sym)
            desc = f"Rose Curve: r = {popt[0]:.4f}*cos({popt[1]:.4f}*theta)"
            formula_str = sp.latex(formula)
        
        elif polar_model == "Archimedean Spiral":
            def polar_func(theta, a, b):
                return a + b * theta
            popt, _ = curve_fit(polar_func, theta, r, p0=[params['a'], 1.0], maxfev=10000)
            r_pred = polar_func(theta, *popt)
            x_pred = r_pred * np.cos(theta)
            y_pred = r_pred * np.sin(theta)
            residuals = np.sqrt((x - x_pred)**2 + (y - y_pred)**2)
            r2 = 1 - np.sum(residuals**2) / np.sum((np.sqrt(x**2 + y**2) - np.mean(np.sqrt(x**2 + y**2)))**2)
            theta_sym = sp.Symbol('theta')
            formula = popt[0] + popt[1] * theta_sym
            desc = f"Archimedean Spiral: r = {popt[0]:.4f} + {popt[1]:.4f}*theta"
            formula_str = sp.latex(formula)
        
        elif polar_model == "Complex Polar":
            def polar_func(theta, *p):
                result = 0
                for i in range(params['n_terms']):
                    result += p[2*i] * np.cos(p[2*i+1] * theta) * theta**i
                result += p[-1]
                return result
            n_params = 2 * params['n_terms'] + 1
            popt, _ = curve_fit(polar_func, theta, r, p0=[1.0] * n_params, maxfev=10000)
            r_pred = polar_func(theta, *popt)
            x_pred = r_pred * np.cos(theta)
            y_pred = r_pred * np.sin(theta)
            residuals = np.sqrt((x - x_pred)**2 + (y - y_pred)**2)
            r2 = 1 - np.sum(residuals**2) / np.sum((np.sqrt(x**2 + y**2) - np.mean(np.sqrt(x**2 + y**2))**2)
            theta_sym = sp.Symbol('theta')
            formula = sum(popt[2*i] * sp.cos(popt[2*i+1] * theta_sym) * theta_sym**i for i in range(params['n_terms'])) + popt[-1]
            desc = f"Complex Polar: r = {' + '.join(f'{popt[2*i]:.4f}*cos({popt[2*i+1]:.4f}*theta)*theta^{i}' for i in range(params['n_terms']))} + {popt[-1]:.4f}"
            formula_str = sp.latex(formula)
        
        elif polar_model == "Combined":
            def polar_func(theta, *p):
                result = 0
                n_terms = params['n_terms']
                # Power terms
                for i in range(n_terms + 1):
                    result += p[i] * theta**i
                # Rose curve term
                result += p[n_terms + 1] * np.cos(p[n_terms + 2] * theta)
                # Archimedean term
                result += p[n_terms + 3] * theta
                result += p[-1]  # Constant
                return result
            n_params = n_terms + 4  # power terms + rose (a, n) + spiral + constant
            popt, _ = curve_fit(polar_func, theta, r, p0=[0.5] * n_params, maxfev=15000)
            r_pred = polar_func(theta, *popt)
            x_pred = r_pred * np.cos(theta)
            y_pred = r_pred * np.sin(theta)
            residuals = np.sqrt((x - x_pred)**2 + (y - y_pred)**2)
            r2 = 1 - np.sum(residuals**2) / np.sum((np.sqrt(x**2 + y**2) - np.mean(np.sqrt(x**2 + y**2))**2)
            theta_sym = sp.Symbol('theta')
            formula = sum(popt[i] * theta_sym**i for i in range(n_terms + 1))
            formula += popt[n_terms + 1] * sp.cos(popt[n_terms + 2] * theta_sym)
            formula += popt[n_terms + 3] * theta_sym
            formula += popt[-1]
            desc = f"Combined Polar: power terms up to theta^{n_terms} + cos term + spiral + constant"
            formula_str = sp.latex(formula)
        
        return popt.tolist(), r2, desc, formula_str
    
    except Exception as e:
        return None, None, f"{polar_model} fit failed: {str(e)}", None

def plot_trig_polar(x, y, coeffs, method, params, formula_str):
    """
    Plot the trigonometric or polar fit.
    Returns: Matplotlib figure
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data Points', s=50)
    
    try:
        x_smooth = np.linspace(min(x), max(x), 200)
        if params['fit_type'] == "Trigonometric":
            trig_model = params['trig_model']
            if trig_model == "Cosine + Sine":
                y_smooth = sum(coeffs[2*i] * np.cos(coeffs[2*i+1] * x_smooth) +
                              coeffs[2*i + params['n_terms']*2] * np.sin(coeffs[2*i + params['n_terms']*2 + 1] * x_smooth)
                              for i in range(params['n_terms'])) + coeffs[-1]
            elif trig_model == "Tan + Cot":
                y_smooth = coeffs[0] * np.tan(coeffs[1] * x_smooth) + coeffs[2] / np.tan(coeffs[3] * x_smooth)
            elif trig_model == "Hyperbolic":
                y_smooth = sum(coeffs[2*i] * np.cosh(coeffs[2*i+1] * x_smooth) +
                              coeffs[2*i + params['n_terms']*2] * np.sinh(coeffs[2*i + params['n_terms']*2 + 1] * x_smooth)
                              for i in range(params['n_terms'])) + coeffs[-1]
            elif trig_model == "Inverse Trig":
                y_smooth = coeffs[0] * np.arctan(coeffs[1] * x_smooth) + coeffs[2] * np.arcsin(coeffs[3] * x_smooth / (1 + np.abs(coeffs[3] * x_smooth)))
            elif trig_model == "Complex Trig":
                y_smooth = sum(coeffs[3*i] * np.cos(coeffs[3*i+1] * x_smooth) * np.tan(coeffs[3*i+2] * x_smooth)
                              for i in range(params['n_terms'])) + coeffs[-1]
            elif trig_model == "Combined":
                n_terms = params['n_terms']
                y_smooth = sum(coeffs[2*i] * np.cos(coeffs[2*i+1] * x_smooth) +
                              coeffs[2*i + n_terms*2] * np.sin(coeffs[2*i + n_terms*2 + 1] * x_smooth) +
                              coeffs[2*i + n_terms*4] * np.cosh(coeffs[2*i + n_terms*4 + 1] * x_smooth) +
                              coeffs[2*i + n_terms*6] * np.sinh(coeffs[2*i + n_terms*6 + 1] * x_smooth)
                              for i in range(n_terms)) + coeffs[-1]
            ax.plot(x_smooth, y_smooth, color='red', label=f'{trig_model} Fit')
        
        elif params['fit_type'] == "Polar":
            polar_model = params['polar_model']
            theta_smooth = np.linspace(min(np.arctan2(y, x)), max(np.arctan2(y, x)), 200)
            if polar_model == "Power Spiral":
                r_smooth = sum(coeffs[i] * theta_smooth**i for i in range(params['max_power'] + 1))
            elif polar_model == "Rose Curve":
                r_smooth = coeffs[0] * np.cos(coeffs[1] * theta_smooth)
            elif polar_model == "Archimedean Spiral":
                r_smooth = coeffs[0] + coeffs[1] * theta_smooth
            elif polar_model == "Complex Polar":
                r_smooth = sum(coeffs[2*i] * np.cos(coeffs[2*i+1] * theta_smooth) * theta_smooth**i
                              for i in range(params['n_terms'])) + coeffs[-1]
            elif polar_model == "Combined":
                n_terms = params['n_terms']
                r_smooth = sum(coeffs[i] * theta_smooth**i for i in range(n_terms + 1))
                r_smooth += coeffs[n_terms + 1] * np.cos(coeffs[n_terms + 2] * theta_smooth)
                r_smooth += coeffs[n_terms + 3] * theta_smooth
                r_smooth += coeffs[-1]
            x_smooth = r_smooth * np.cos(theta_smooth)
            y_smooth = r_smooth * np.sin(theta_smooth)
            ax.plot(x_smooth, y_smooth, color='red', label=f'{polar_model} Fit')
        
        ax.text(0.02, 0.98, f"Formula: ${formula_str}$", transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True)
        return fig
    
    except Exception as e:
        ax.text(0.5, 0.5, f'Plot failed: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes, color='red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        return fig

def compare_trig_polar_modes(lines, params):
    """
    Compare all Trigonometric and Polar sub-modes for each line.
    Args:
        lines: List of tuples (line_name, x, y, has_duplicates, has_invalid_x)
        params: Parameters including n_points for plotting
    """
    st.subheader("Trigonometric and Polar Visual Comparison")
    st.markdown("This mode compares all trigonometric and polar sub-modes with default parameters for visual inspection.")
    
    trig_methods = [
        ("Cosine + Sine", {"n_terms": 2}),
        ("Tan + Cot", {"scale": 1.0}),
        ("Hyperbolic", {"n_terms": 2}),
        ("Inverse Trig", {"scale": 1.0}),
        ("Complex Trig", {"n_terms": 2}),
        ("Combined", {"n_terms": 1})
    ]
    
    polar_methods = [
        ("Power Spiral", {"max_power": 2}),
        ("Rose Curve", {"n_petals": 4}),
        ("Archimedean Spiral", {"a": 1.0}),
        ("Complex Polar", {"n_terms": 2}),
        ("Combined", {"n_terms": 1})
    ]
    
    results = {}
    for method, method_params in trig_methods:
        method_params['fit_type'] = "Trigonometric"
        method_params['trig_model'] = method
        results[f"Trig: {method}"] = [(line_name, *fit_trigonometric(x, y, method, method_params), has_duplicates, has_invalid_x)
                                     for line_name, x, y, has_duplicates, has_invalid_x in lines]
    
    for method, method_params in polar_methods:
        method_params['fit_type'] = "Polar"
        method_params['polar_model'] = method
        results[f"Polar: {method}"] = [(line_name, *fit_polar(x, y, method, method_params), has_duplicates, has_invalid_x)
                                      for line_name, x, y, has_duplicates, has_invalid_x in lines]
    
    comparison_data = []
    for method_key in results:
        fit_type, sub_model = method_key.split(": ")
        for line_name, coeffs, r2, desc, formula, has_duplicates, has_invalid_x in results[method_key]:
            if coeffs is not None and formula is not None:
                comparison_data.append({
                    'Line': line_name,
                    'Fit Type': fit_type,
                    'Sub-Model': sub_model,
                    'R²': f"{r2:.4f}",
                    'Description': desc
                })
    
    if comparison_data:
        st.subheader("Fit Quality Comparison Table")
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    for line_name, x, y, has_duplicates, has_invalid_x in lines:
        st.markdown(f"### Line: {line_name}")
        if len(x) < 3:
            st.warning(f"Line '{line_name}': Skipped due to insufficient points (need at least 3).")
            continue
        for method_key in results:
            fit_type, sub_model = method_key.split(": ")
            result = next((r for r in results[method_key] if r[0] == line_name), None)
            if result and result[3] is not None:
                coeffs, r2, desc, formula, _, _ = result[1:]
                st.write(f"{method_key}: {desc}, R² = {r2:.4f}")
                st.markdown(f"Formula: ${formula}$")
                st.markdown(f"```latex\n{formula}\n```")
                params_copy = {'fit_type': fit_type, f"{fit_type.lower()}_model": sub_model}
                if sub_model in ["Cosine + Sine", "Hyperbolic", "Complex Trig", "Combined"]:
                    params_copy['n_terms'] = 2 if sub_model != "Combined" else 1
                elif sub_model in ["Tan + Cot", "Inverse Trig"]:
                    params_copy['scale'] = 1.0
                elif sub_model == "Power Spiral":
                    params_copy['max_power'] = 2
                elif sub_model == "Rose Curve":
                    params_copy['n_petals'] = 4
                elif sub_model == "Archimedean Spiral":
                    params_copy['a'] = 1.0
                fig = plot_trig_polar(x, y, coeffs, fit_type, params_copy, formula)
                st.pyplot(fig)
            else:
                st.warning(f"Line '{line_name}' - {method_key}: {result[3] if result else 'No result'}")
