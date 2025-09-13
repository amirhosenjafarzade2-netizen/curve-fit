# plotting.py - Functions for generating interactive plots with Plotly

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import UnivariateSpline

def plot_fit(x, y, coeffs, method, params, r2, line_name):
    # Create figure
    fig = go.Figure()

    # Add scatter for data points
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Data Points',
        marker=dict(color='blue', size=8),
        hovertemplate='x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
    ))

    # Generate smooth curve
    x_smooth = np.linspace(min(x), max(x), 200)
    if method in ["Logarithmic", "Compound Poly+Log"]:
        # Ensure x_smooth is positive for log models
        x_smooth = np.linspace(max(min(x), 1e-10), max(x), 200)

    try:
        if method == "Polynomial":
            poly = np.poly1d(coeffs)
            y_smooth = poly(x_smooth)
            label = f'Polynomial (deg {params.get("degree", "?")})'
        elif method == "Exponential":
            y_smooth = coeffs[0] * np.exp(coeffs[1] * x_smooth) + coeffs[2]
            label = 'Exponential'
        elif method == "Logarithmic":
            y_smooth = coeffs[0] * np.log(x_smooth) + coeffs[1]
            label = 'Logarithmic'
        elif method == "Compound Poly+Log":
            y_smooth = coeffs[0] * x_smooth**2 + coeffs[1] * np.log(x_smooth) + coeffs[2]
            label = 'Poly+Log'
        elif method == "Spline":
            degree = params.get('degree', 3)
            try:
                spline = UnivariateSpline(x, y, k=degree, s=0)
                y_smooth = spline(x_smooth)
                label = f'Spline (deg {degree})'
            except ValueError:
                return go.Figure(data=[
                    go.Scatter(x=x, y=y, mode='markers', name='Data Points', marker=dict(color='blue', size=8)),
                    go.Scatter(x=[None], y=[None], mode='text', text=['Spline plot failed: x must be strictly increasing'], textposition='middle center')
                ], layout=dict(title=f"{line_name}: Spline Failed", showlegend=False))

        # Add fitted curve
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode='lines',
            name=label,
            line=dict(color='red', width=2),
            hovertemplate='x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
        ))
    except Exception as e:
        print(f"Plotting failed for {method}: {str(e)}")
        return go.Figure(data=[
            go.Scatter(x=x, y=y, mode='markers', name='Data Points', marker=dict(color='blue', size=8)),
            go.Scatter(x=[None], y=[None], mode='text', text=[f'{method} plot failed: {str(e)}'], textposition='middle center')
        ], layout=dict(title=f"{line_name}: {method} Failed", showlegend=False))

    # Update layout
    fig.update_layout(
        title=f"{line_name}: {label} (R² = {r2:.4f})",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        font=dict(family="Arial", size=14),
        xaxis=dict(
            gridcolor='lightgrey',
            zerolinecolor='lightgrey',
            type='log' if method in ["Logarithmic", "Compound Poly+Log"] else 'linear'
        ),
        yaxis=dict(
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig
