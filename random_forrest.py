```python
# random_forest.py - Standalone module for Random Forest fitting, plotting, and UI

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import streamlit as st

def fit_random_forest(x, y, n_estimators=100):
    """
    Fit a Random Forest model to the data.
    Returns: coefficients ([n_estimators]), R², description
    """
    try:
        if len(x) < 3:
            raise ValueError("Insufficient points for Random Forest")
        model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=42)
        model.fit(x.reshape(-1, 1), y)
        y_pred = model.predict(x.reshape(-1, 1))
        r2 = r2_score(y, y_pred)
        return [n_estimators], r2, f"Random Forest (n_estimators {int(n_estimators)})"
    except Exception as e:
        return None, None, f"Random Forest fit failed: {str(e)}"

def plot_random_forest(x, y, coeffs, method, params):
    """
    Plot the Random Forest fit.
    Args:
        x, y: Input data
        coeffs: [n_estimators]
        method: Method name (for compatibility)
        params: Parameters (for compatibility)
    Returns: Matplotlib figure
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data Points', s=50)

    try:
        n_estimators = coeffs[0]
        model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=42)
        model.fit(x.reshape(-1, 1), y)
        x_smooth = np.linspace(min(x), max(x), 200)
        y_smooth = model.predict(x_smooth.reshape(-1, 1))
        ax.plot(x_smooth, y_smooth, color='red', label=f'Random Forest (n_estimators {int(n_estimators)})')
    except ValueError:
        ax.text(0.5, 0.5, 'Random Forest plot failed', 
                ha='center', va='center', transform=ax.transAxes, color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    return fig

def random_forest_ui():
    """
    Render UI for Random Forest parameters in Standard mode.
    Returns: Dictionary of parameters
    """
    params = {}
    params['n_estimators'] = st.number_input("Number of trees", min_value=10, max_value=500, value=100, step=10)
    return params
```
