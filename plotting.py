# plotting.py - Functions for generating plots

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import pywt
import pandas as pd

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
    elif method == "Savitzky-Golay":
        try:
            window, polyorder = coeffs
            smoothed_y = savgol_filter(y, window, polyorder)
            y_smooth = np.interp(x_smooth, x, smoothed_y)
            ax.plot(x_smooth, y_smooth, color='red', label=f'Savitzky-Golay (win {window}, ord {polyorder})')
        except ValueError:
            ax.text(0.5, 0.5, 'Savitzky-Golay plot failed', 
                    ha='center', va='center', transform=ax.transAxes, color='red')
    elif method == "LOWESS":
        try:
            frac = coeffs[0]
            smoothed_y = lowess(y, x, frac=frac, return_sorted=False)
            y_smooth = np.interp(x_smooth, x, smoothed_y)
            ax.plot(x_smooth, y_smooth, color='red', label=f'LOWESS (frac {frac})')
        except ValueError:
            ax.text(0.5, 0.5, 'LOWESS plot failed', 
                    ha='center', va='center', transform=ax.transAxes, color='red')
    elif method == "Exponential Smoothing":
        try:
            alpha = coeffs[0]
            smoothed_y = pd.Series(y).ewm(alpha=alpha).mean().values
            y_smooth = np.interp(x_smooth, x, smoothed_y)
            ax.plot(x_smooth, y_smooth, color='red', label=f'Exponential Smoothing (alpha {alpha})')
        except ValueError:
            ax.text(0.5, 0.5, 'Exponential Smoothing plot failed', 
                    ha='center', va='center', transform=ax.transAxes, color='red')
    elif method == "Gaussian Smoothing":
        try:
            sigma = coeffs[0]
            smoothed_y = gaussian_filter1d(y, sigma)
            y_smooth = np.interp(x_smooth, x, smoothed_y)
            ax.plot(x_smooth, y_smooth, color='red', label=f'Gaussian Smoothing (sigma {sigma})')
        except ValueError:
            ax.text(0.5, 0.5, 'Gaussian Smoothing plot failed', 
                    ha='center', va='center', transform=ax.transAxes, color='red')
    elif method == "Wavelet Denoising":
        try:
            wavelet, level, threshold = coeffs
            coeffs_wave = pywt.wavedec(y, wavelet, level=level)
            for i in range(1, len(coeffs_wave)):
                coeffs_wave[i] = pywt.threshold(coeffs_wave[i], threshold * np.max(coeffs_wave[i]), mode='soft')
            smoothed_y = pywt.waverec(coeffs_wave, wavelet)
            if len(smoothed_y) > len(y):
                smoothed_y = smoothed_y[:len(y)]
            y_smooth = np.interp(x_smooth, x, smoothed_y)
            ax.plot(x_smooth, y_smooth, color='red', label=f'Wavelet Denoising ({wavelet}, lev {level}, thr {threshold})')
        except ValueError:
            ax.text(0.5, 0.5, 'Wavelet Denoising plot failed', 
                    ha='center', va='center', transform=ax.transAxes, color='red')
    elif method == "Sine":
        a, b, c, d = coeffs
        y_smooth = a * np.sin(b * x_smooth + c) + d
        ax.plot(x_smooth, y_smooth, color='red', label=f'Sine (freq_guess {params.get("frequency_guess", "?")})')
    elif method == "Cosine":
        a, b, c, d = coeffs
        y_smooth = a * np.cos(b * x_smooth + c) + d
        ax.plot(x_smooth, y_smooth, color='red', label=f'Cosine (freq_guess {params.get("frequency_guess", "?")})')
    elif method == "Tangent":
        a, b, c, d = coeffs
        y_smooth = a * np.tan(b * x_smooth + c) + d
        ax.plot(x_smooth, y_smooth, color='red', label=f'Tangent (freq_guess {params.get("frequency_guess", "?")})')
    elif method == "Cotangent":
        a, b, c, d = coeffs
        y_smooth = a / np.tan(b * x_smooth + c) + d
        ax.plot(x_smooth, y_smooth, color='red', label=f'Cotangent (freq_guess {params.get("frequency_guess", "?")})')
    elif method == "Sinh":
        a, b, c, d = coeffs
        y_smooth = a * np.sinh(b * x_smooth + c) + d
        ax.plot(x_smooth, y_smooth, color='red', label=f'Sinh (scaling_guess {params.get("scaling_guess", "?")})')
    elif method == "Cosh":
        a, b, c, d = coeffs
        y_smooth = a * np.cosh(b * x_smooth + c) + d
        ax.plot(x_smooth, y_smooth, color='red', label=f'Cosh (scaling_guess {params.get("scaling_guess", "?")})')
    elif method == "Tanh":
        a, b, c, d = coeffs
        y_smooth = a * np.tanh(b * x_smooth + c) + d
        ax.plot(x_smooth, y_smooth, color='red', label=f'Tanh (scaling_guess {params.get("scaling_guess", "?")})')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    return fig
