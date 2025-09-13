# plotting.py - Functions for generating plots

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def plot_fit(x, y, coeffs, method, params):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data Points', s=50)

    x_smooth = np.linspace(min(x), max(x), 200)

    if method == "Polynomial":
        poly = np.poly1d(coeffs)
        y_smooth = poly(x_smooth)
        ax.plot(x_smooth, y_smooth, color='red', label=f'Polynomial (deg {params.get("degree", "?")})')
    elif method == "Exponential":
        y_smooth = coeffs[0] * np.exp(coeffs[1] * x_smooth) + coeffs[2]
        ax.plot(x_smooth, y_smooth, color='red', label='Exponential')
    elif method == "Logarithmic":
        y_smooth = coeffs[0] * np.log(x_smooth + 1e-10) + coeffs[1]
        ax.plot(x_smooth, y_smooth, color='red', label='Logarithmic')
    elif method == "Compound Poly+Log":
        y_smooth = coeffs[0] * x_smooth**2 + coeffs[1] * np.log(x_smooth + 1e-10) + coeffs[2]
        ax.plot(x_smooth, y_smooth, color='red', label='Poly+Log')
    elif method == "Spline":
        degree = params.get('degree', 3)
        try:
            spline = UnivariateSpline(x, y, k=degree, s=0)
            y_smooth = spline(x_smooth)
            ax.plot(x_smooth, y_smooth, color='red', label=f'Spline (deg {degree})')
        except ValueError:
            ax.text(0.5, 0.5, 'Spline plot failed: x must be strictly increasing', 
                    ha='center', va='center', transform=ax.transAxes, color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    return fig
