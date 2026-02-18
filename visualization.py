# visualization.py
# Advanced plot customization module for the Curve Fitting Streamlit App
# Place this file in the same directory as app.py

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

def visualization_ui(default_title="Curve Fit Plot"):
    """
    Renders UI controls for customizing matplotlib plots.
    Returns dictionary of parameters to be passed to apply_customizations().
    """
    st.subheader("Plot Appearance Settings")

    col1, col2 = st.columns([3, 2])

    with col1:
        params = {}

        # ── Title & Labels ───────────────────────────────────────────────
        params['title']       = st.text_input("Plot Title", value=default_title)
        params['xlabel']      = st.text_input("X-axis Label", value="X")
        params['ylabel']      = st.text_input("Y-axis Label", value="Y")
        params['font_family'] = st.selectbox("Font Family", ["sans-serif", "serif", "monospace", "Arial", "Helvetica", "DejaVu Sans"], index=0)
        params['font_size']   = st.slider("Base Font Size", 8, 24, 12)

        # ── Axis Limits & Direction ──────────────────────────────────────
        col_lim1, col_lim2 = st.columns(2)
        with col_lim1:
            params['x_min'] = st.number_input("X min", value=None, format="%.4f")
            params['x_max'] = st.number_input("X max", value=None, format="%.4f")
            params['invert_x'] = st.checkbox("Invert X axis", False)
        with col_lim2:
            params['y_min'] = st.number_input("Y min", value=None, format="%.4f")
            params['y_max'] = st.number_input("Y max", value=None, format="%.4f")
            params['invert_y'] = st.checkbox("Invert Y axis", False)

        # ── Grid & Ticks ─────────────────────────────────────────────────
        params['show_grid']       = st.checkbox("Show major grid", True)
        params['show_minor_grid'] = st.checkbox("Show minor grid", False) if params['show_grid'] else False

        if params['show_grid'] or params['show_minor_grid']:
            col_tick1, col_tick2 = st.columns(2)
            with col_tick1:
                params['major_x_step'] = st.number_input("Major X tick step", value=None, min_value=0.001, format="%.4f")
                params['minor_x_step'] = st.number_input("Minor X tick step", value=None, min_value=0.001, format="%.4f") if params['show_minor_grid'] else None
            with col_tick2:
                params['major_y_step'] = st.number_input("Major Y tick step", value=None, min_value=0.001, format="%.4f")
                params['minor_y_step'] = st.number_input("Minor Y tick step", value=None, min_value=0.001, format="%.4f") if params['show_minor_grid'] else None

    with col2:
        # ── Colors & Style ───────────────────────────────────────────────
        params['bg_color']       = st.color_picker("Background color", "#ffffff")
        params['grid_color']     = st.color_picker("Grid color", "#cccccc")
        params['axis_color']     = st.color_picker("Axis & tick color", "#000000")
        params['line_color']     = st.color_picker("Main fitted line color", "#d32f2f")
        params['data_point_color']= st.color_picker("Original data points color", "#1976d2")

        # ── Lines & Markers ──────────────────────────────────────────────
        params['line_width']     = st.slider("Line width", 0.5, 6.0, 1.8, step=0.2)
        params['line_style']     = st.selectbox("Line style", ["-", "--", ":", "-."], index=0)

        params['show_markers']   = st.selectbox("Show markers", ["None", "All points", "Every N points", "Only original points"], index=0)
        if params['show_markers'] != "None":
            params['marker']     = st.selectbox("Marker shape", ['o','^','s','D','*','+','x','p','h','v'], index=0)
            params['marker_size']= st.slider("Marker size", 3, 18, 6)
            if params['show_markers'] == "Every N points":
                params['marker_every'] = st.number_input("Show marker every", 1, 50, 10)

        # ── Legend ───────────────────────────────────────────────────────
        params['show_legend']    = st.checkbox("Show legend", True)
        if params['show_legend']:
            params['legend_loc'] = st.selectbox("Legend location",
                ["best", "upper right", "upper left", "lower right", "lower left",
                 "center right", "center left", "outside right"],
                index=0)

        # ── Resolution ───────────────────────────────────────────────────
        params['dpi'] = st.slider("Export / Display DPI", 100, 600, 300)

    return params


def apply_plot_customizations(fig, ax, params, original_data=False):
    """
    Applies all custom settings to an existing figure & axis.
    Call this after you have already plotted your data.

    Args:
        fig: matplotlib.figure.Figure
        ax:  matplotlib.axes.Axes
        params: dict from visualization_ui()
        original_data: bool - whether original scatter points are already plotted
    """
    # Title & labels
    if params.get('title'):
        fig.suptitle(params['title'], fontsize=params['font_size']+4, family=params['font_family'])
    ax.set_xlabel(params.get('xlabel', 'X'), fontsize=params['font_size']+1)
    ax.set_ylabel(params.get('ylabel', 'Y'), fontsize=params['font_size']+1)

    # Axis limits & inversion
    if params.get('x_min') is not None and params.get('x_max') is not None:
        ax.set_xlim(params['x_min'], params['x_max'])
    if params.get('y_min') is not None and params.get('y_max') is not None:
        ax.set_ylim(params['y_min'], params['y_max'])
    if params.get('invert_x'): ax.invert_xaxis()
    if params.get('invert_y'): ax.invert_yaxis()

    # Grid
    if params.get('show_grid', False):
        ax.grid(True, color=params['grid_color'], linestyle='-', linewidth=0.8, alpha=0.7)
    if params.get('show_minor_grid', False):
        ax.minorticks_on()
        ax.grid(True, which='minor', color=params['grid_color'], linestyle=':', linewidth=0.4, alpha=0.5)

    # Custom tick locators
    if params.get('major_x_step') is not None:
        ax.xaxis.set_major_locator(MultipleLocator(params['major_x_step']))
    if params.get('major_y_step') is not None:
        ax.yaxis.set_major_locator(MultipleLocator(params['major_y_step']))
    if params.get('minor_x_step') is not None:
        ax.xaxis.set_minor_locator(MultipleLocator(params['minor_x_step']))
    if params.get('minor_y_step') is not None:
        ax.yaxis.set_minor_locator(MultipleLocator(params['minor_y_step']))

    # Colors & spines
    ax.set_facecolor(params['bg_color'])
    fig.patch.set_facecolor(params['bg_color'])
    for spine in ax.spines.values():
        spine.set_edgecolor(params['axis_color'])
    ax.tick_params(colors=params['axis_color'], labelsize=params['font_size']-1)

    # Legend
    if params.get('show_legend', True) and ax.get_legend_handles_labels()[0]:
        loc = params['legend_loc']
        if loc == "outside right":
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=params['font_size']-1)
        else:
            ax.legend(loc=loc, fontsize=params['font_size']-1)

    # You should apply line/marker properties when plotting, e.g.:
    # ax.plot(..., color=params['line_color'], lw=params['line_width'], ls=...)

    fig.set_dpi(params['dpi'])
    fig.tight_layout()
