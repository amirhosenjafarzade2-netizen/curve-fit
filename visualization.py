# visualization.py
# Advanced plot customization module for the Curve Fitting Streamlit App
# Place this file in the same directory as app.py

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, LogLocator
import numpy as np
import io


# ── Preset themes ────────────────────────────────────────────────────────────

THEMES = {
    "Default (white)": {
        "bg_color": "#ffffff", "grid_color": "#cccccc", "axis_color": "#000000",
        "line_color": "#d32f2f", "data_point_color": "#1976d2",
        "title_color": "#000000", "label_color": "#000000",
        "fig_bg_color": "#ffffff",
    },
    "Dark": {
        "bg_color": "#1e1e1e", "grid_color": "#444444", "axis_color": "#dddddd",
        "line_color": "#ff6b6b", "data_point_color": "#64b5f6",
        "title_color": "#ffffff", "label_color": "#cccccc",
        "fig_bg_color": "#121212",
    },
    "Seaborn": {
        "bg_color": "#eaeaf2", "grid_color": "#ffffff", "axis_color": "#555555",
        "line_color": "#4c72b0", "data_point_color": "#dd8452",
        "title_color": "#333333", "label_color": "#555555",
        "fig_bg_color": "#eaeaf2",
    },
    "Minimal / Clean": {
        "bg_color": "#ffffff", "grid_color": "#e0e0e0", "axis_color": "#888888",
        "line_color": "#222222", "data_point_color": "#888888",
        "title_color": "#222222", "label_color": "#666666",
        "fig_bg_color": "#ffffff",
    },
    "Blueprint": {
        "bg_color": "#0a1628", "grid_color": "#1e3a5f", "axis_color": "#4fc3f7",
        "line_color": "#00e5ff", "data_point_color": "#80cbc4",
        "title_color": "#e1f5fe", "label_color": "#b3e5fc",
        "fig_bg_color": "#060e1a",
    },
    "Warm Paper": {
        "bg_color": "#fdf6e3", "grid_color": "#e8dcc8", "axis_color": "#5c4a2a",
        "line_color": "#a0522d", "data_point_color": "#6b8e23",
        "title_color": "#3b2f1e", "label_color": "#5c4a2a",
        "fig_bg_color": "#fdf6e3",
    },
    "High Contrast": {
        "bg_color": "#000000", "grid_color": "#333333", "axis_color": "#ffffff",
        "line_color": "#ffff00", "data_point_color": "#00ff00",
        "title_color": "#ffffff", "label_color": "#ffffff",
        "fig_bg_color": "#000000",
    },
    "Custom": {},
}

FIGURE_SIZES = {
    "Standard (10x6)": (10, 6),
    "Square (8x8)": (8, 8),
    "Wide (14x6)": (14, 6),
    "Tall (8x10)": (8, 10),
    "Presentation (16x9)": (16, 9),
    "A4 landscape (11.7x8.3)": (11.7, 8.3),
    "Custom": None,
}

EXPORT_FORMATS = ["png", "pdf", "svg", "eps"]

MARKER_DISPLAY = {
    "Circle (o)": "o", "Triangle up (^)": "^", "Triangle down (v)": "v",
    "Square (s)": "s", "Diamond (D)": "D", "Thin diamond (d)": "d",
    "Star (*)": "*", "Plus (+)": "+", "Cross (x)": "x",
    "Pentagon (p)": "p", "Hexagon (h)": "h", "Octagon (8)": "8",
}

LINE_STYLES = {
    "Solid": "-",
    "Dashed": "--",
    "Dotted": ":",
    "Dash-dot": "-.",
}

SCALE_OPTIONS = ["linear", "log", "symlog", "logit"]


# ── Main UI ──────────────────────────────────────────────────────────────────

def visualization_ui(default_title="Curve Fit Plot"):
    """
    Renders UI controls for customizing matplotlib plots.
    Returns dict of parameters to be passed to apply_plot_customizations().
    """
    st.subheader("Plot Appearance Settings")
    params = {}

    # ── Theme ────────────────────────────────────────────────────────────────
    with st.expander("Theme Preset", expanded=True):
        theme_name = st.selectbox(
            "Apply a theme preset",
            list(THEMES.keys()), index=0,
            help="Quickly apply a coordinated color scheme. Choose 'Custom' to pick every color manually."
        )
        theme = THEMES[theme_name]

    def t(key, default):
        """Pull value from theme, falling back to default."""
        return theme.get(key, default) if theme else default

    # ── Figure size & layout ─────────────────────────────────────────────────
    with st.expander("Figure Size & Layout", expanded=False):
        size_label = st.selectbox("Figure size", list(FIGURE_SIZES.keys()), index=0)
        if size_label == "Custom":
            c1, c2 = st.columns(2)
            with c1:
                fw = st.number_input("Width (inches)", 4.0, 30.0, 10.0, 0.5)
            with c2:
                fh = st.number_input("Height (inches)", 2.0, 20.0, 6.0, 0.5)
            params['fig_size'] = (fw, fh)
        else:
            params['fig_size'] = FIGURE_SIZES[size_label]

        params['dpi'] = st.slider("DPI (display & export)", 72, 600, 150, step=1)
        params['tight_layout'] = st.checkbox("Auto tight layout", True)
        params['constrained_layout'] = st.checkbox("Constrained layout", False,
            help="Alternative to tight layout — avoids label clipping in complex figures.")

        c1, c2, c3, c4 = st.columns(4)
        with c1: params['subplot_left']   = st.slider("Left margin",   0.0, 0.5, 0.08, 0.01)
        with c2: params['subplot_right']  = st.slider("Right margin",  0.5, 1.0, 0.95, 0.01)
        with c3: params['subplot_top']    = st.slider("Top margin",    0.5, 1.0, 0.92, 0.01)
        with c4: params['subplot_bottom'] = st.slider("Bottom margin", 0.0, 0.5, 0.10, 0.01)

        params['fig_bg_color'] = st.color_picker("Figure background", t("fig_bg_color", "#ffffff"))

    # ── Title & labels ────────────────────────────────────────────────────────
    with st.expander("Title & Labels", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            params['title']       = st.text_input("Plot title", value=default_title)
            params['title_size']  = st.slider("Title font size", 8, 36, 16)
            params['title_color'] = st.color_picker("Title color", t("title_color", "#000000"))
            params['title_style'] = st.selectbox("Title weight/style",
                                                  ["normal", "bold", "italic", "bold italic"])
            params['title_pad']   = st.slider("Title padding (pts)", 0, 30, 10)
        with col2:
            params['xlabel']      = st.text_input("X-axis label", value="X")
            params['ylabel']      = st.text_input("Y-axis label", value="Y")
            params['label_size']  = st.slider("Axis label font size", 6, 28, 13)
            params['label_color'] = st.color_picker("Axis label color", t("label_color", "#000000"))
            params['font_family'] = st.selectbox("Font family",
                ["sans-serif", "serif", "monospace", "Arial", "Helvetica",
                 "DejaVu Sans", "Times New Roman", "Courier New", "Palatino"], index=0)
            params['font_size']   = st.slider("Tick label font size", 6, 24, 11)

    # ── Axis limits, scale & direction ───────────────────────────────────────
    with st.expander("Axis Limits, Scale & Direction", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**X Axis**")
            params['x_scale']  = st.selectbox("X scale", SCALE_OPTIONS, index=0, key="xs")
            params['x_min']    = st.number_input("X min (0 = auto)", value=0.0, format="%.6f", key="xmn")
            params['x_max']    = st.number_input("X max (0 = auto)", value=0.0, format="%.6f", key="xmx")
            params['invert_x'] = st.checkbox("Invert X axis", False)
            if params['x_scale'] == "symlog":
                params['x_sym_linthresh'] = st.number_input("symlog X threshold", value=1.0, format="%.4f")
            else:
                params['x_sym_linthresh'] = 1.0
        with col2:
            st.markdown("**Y Axis**")
            params['y_scale']  = st.selectbox("Y scale", SCALE_OPTIONS, index=0, key="ys")
            params['y_min']    = st.number_input("Y min (0 = auto)", value=0.0, format="%.6f", key="ymn")
            params['y_max']    = st.number_input("Y max (0 = auto)", value=0.0, format="%.6f", key="ymx")
            params['invert_y'] = st.checkbox("Invert Y axis", False)
            if params['y_scale'] == "symlog":
                params['y_sym_linthresh'] = st.number_input("symlog Y threshold", value=1.0, format="%.4f")
            else:
                params['y_sym_linthresh'] = 1.0

        params['aspect_ratio'] = st.selectbox("Aspect ratio", ["auto", "equal"], index=0,
            help="'equal' forces identical unit lengths on both axes.")

    # ── Grid & ticks ─────────────────────────────────────────────────────────
    with st.expander("Grid & Ticks", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            params['show_grid']       = st.checkbox("Show major grid", True)
            if params['show_grid']:
                params['grid_linestyle'] = st.selectbox("Major grid style",
                                                         list(LINE_STYLES.keys()), index=0, key="gls")
                params['grid_linewidth'] = st.slider("Major grid width", 0.2, 3.0, 0.8, 0.1)
                params['grid_alpha']     = st.slider("Major grid alpha", 0.0, 1.0, 0.7, 0.05)
            else:
                params['grid_linestyle'] = "Solid"
                params['grid_linewidth'] = 0.8
                params['grid_alpha'] = 0.7

            params['grid_color']      = st.color_picker("Grid color", t("grid_color", "#cccccc"))
            params['show_minor_grid'] = st.checkbox("Show minor grid", False) if params['show_grid'] else False
            if params['show_minor_grid']:
                params['minor_grid_alpha'] = st.slider("Minor grid alpha", 0.0, 1.0, 0.35, 0.05)
            else:
                params['minor_grid_alpha'] = 0.35

        with col2:
            st.markdown("**Tick spacing** (0 = automatic)")
            params['major_x_step'] = st.number_input("Major X step", value=0.0, min_value=0.0, format="%.4f")
            params['major_y_step'] = st.number_input("Major Y step", value=0.0, min_value=0.0, format="%.4f")
            params['minor_x_step'] = st.number_input("Minor X step", value=0.0, min_value=0.0, format="%.4f")
            params['minor_y_step'] = st.number_input("Minor Y step", value=0.0, min_value=0.0, format="%.4f")

            params['axis_color']      = st.color_picker("Axis & tick color", t("axis_color", "#000000"))
            params['tick_direction']  = st.selectbox("Tick direction", ["in", "out", "inout"], index=1)
            params['tick_length']     = st.slider("Major tick length", 2, 20, 6)
            params['tick_width']      = st.slider("Tick width", 0.5, 3.0, 1.0, 0.1)
            params['x_tick_rotation'] = st.slider("X tick label rotation (°)", 0, 90, 0, step=5)
            params['y_tick_rotation'] = st.slider("Y tick label rotation (°)", 0, 90, 0, step=5)

        params['show_top_spine']   = st.checkbox("Show top spine", True)
        params['show_right_spine'] = st.checkbox("Show right spine", True)

    # ── Fitted line ───────────────────────────────────────────────────────────
    with st.expander("Fitted Line Appearance", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            params['line_color']  = st.color_picker("Line color", t("line_color", "#d32f2f"))
            params['line_width']  = st.slider("Line width", 0.5, 8.0, 2.0, step=0.25)
            params['line_style']  = LINE_STYLES[st.selectbox("Line style", list(LINE_STYLES.keys()), index=0, key="lls")]
            params['line_alpha']  = st.slider("Line opacity", 0.1, 1.0, 1.0, 0.05)
            params['line_zorder'] = st.number_input("Draw order (zorder)", value=5, min_value=1, max_value=20)
        with col2:
            params['line_shadow'] = st.checkbox("Drop shadow on line", False,
                help="Adds a glow/shadow effect under the fitted line.")
            if params['line_shadow']:
                params['shadow_color'] = st.color_picker("Shadow color", "#888888")
                params['shadow_width'] = st.slider("Shadow width (pts)", 1, 12, 4)
            else:
                params['shadow_color'] = "#888888"
                params['shadow_width'] = 4

            params['fill_under_line'] = st.checkbox("Fill area under fitted line", False)
            if params['fill_under_line']:
                params['fill_color'] = st.color_picker("Fill color", "#ffcccc")
                params['fill_alpha'] = st.slider("Fill opacity", 0.05, 0.8, 0.2, 0.05)
            else:
                params['fill_color'] = "#ffcccc"
                params['fill_alpha'] = 0.2

    # ── Data points ──────────────────────────────────────────────────────────
    with st.expander("Data Points", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            params['data_point_color']  = st.color_picker("Marker fill color", t("data_point_color", "#1976d2"))
            params['data_point_size']   = st.slider("Marker area (pts²)", 4, 300, 45, step=1)
            params['data_point_alpha']  = st.slider("Marker opacity", 0.1, 1.0, 0.85, 0.05)
            params['data_point_zorder'] = st.number_input("Marker draw order", value=3, min_value=1, max_value=20)
            mk_choice = st.selectbox("Marker shape", list(MARKER_DISPLAY.keys()), index=0)
            params['data_point_marker'] = MARKER_DISPLAY[mk_choice]
        with col2:
            params['data_edge_color']  = st.color_picker("Marker edge color", "#ffffff")
            params['data_edge_width']  = st.slider("Marker edge width", 0.0, 3.0, 0.5, 0.1)
            params['show_errorbars']   = st.checkbox("Show error bars", False,
                help="Requires xerr/yerr arrays passed into apply_plot_customizations().")
            if params['show_errorbars']:
                params['errorbar_capsize']   = st.slider("Cap size", 0, 15, 4)
                params['errorbar_color']     = st.color_picker("Error bar color", "#555555")
                params['errorbar_linewidth'] = st.slider("Error bar width", 0.5, 3.0, 1.0, 0.1)
            else:
                params['errorbar_capsize'] = 4
                params['errorbar_color'] = "#555555"
                params['errorbar_linewidth'] = 1.0

    # ── Markers on fitted line ────────────────────────────────────────────────
    with st.expander("Markers on Fitted Line", expanded=False):
        params['show_markers'] = st.selectbox("Show markers on fitted line",
                                               ["None", "All points", "Every N points"], index=0)
        if params['show_markers'] != "None":
            col1, col2 = st.columns(2)
            with col1:
                mk2 = st.selectbox("Marker shape", list(MARKER_DISPLAY.keys()), index=0, key="fmk")
                params['marker']           = MARKER_DISPLAY[mk2]
                params['marker_size']      = st.slider("Marker size (fitted)", 3, 20, 6)
                params['marker_color']     = st.color_picker("Marker fill", "#d32f2f", key="fmc")
            with col2:
                params['marker_edge_color']= st.color_picker("Marker edge", "#ffffff", key="fmec")
                params['marker_edge_width']= st.slider("Marker edge width", 0.0, 3.0, 1.0, 0.1, key="fmew")
                if params['show_markers'] == "Every N points":
                    params['marker_every'] = st.number_input("Every N points", 1, 200, 10)
        else:
            params['marker'] = 'o'
            params['marker_size'] = 6
            params['marker_color'] = "#d32f2f"
            params['marker_edge_color'] = "#ffffff"
            params['marker_edge_width'] = 1.0
            params['marker_every'] = 10

    # ── Legend ────────────────────────────────────────────────────────────────
    with st.expander("Legend", expanded=False):
        params['show_legend'] = st.checkbox("Show legend", True)
        if params['show_legend']:
            col1, col2 = st.columns(2)
            with col1:
                params['legend_loc'] = st.selectbox("Location",
                    ["best", "upper right", "upper left", "lower right", "lower left",
                     "center right", "center left", "center", "upper center", "lower center",
                     "outside right", "outside bottom"], index=0)
                params['legend_fontsize']  = st.slider("Legend font size", 6, 24, 10)
                params['legend_title']     = st.text_input("Legend title", value="")
                params['legend_ncol']      = st.number_input("Columns", 1, 6, 1)
            with col2:
                params['legend_frameon']   = st.checkbox("Show frame", True)
                params['legend_alpha']     = st.slider("Background alpha", 0.0, 1.0, 0.8, 0.05)
                params['legend_edgecolor'] = st.color_picker("Frame color", "#cccccc")
                params['legend_facecolor'] = st.color_picker("Background color", "#ffffff", key="lgbg")
                params['legend_shadow']    = st.checkbox("Drop shadow", False)
        else:
            params['legend_loc'] = 'best'
            params['legend_fontsize'] = 10
            params['legend_title'] = ''
            params['legend_ncol'] = 1
            params['legend_frameon'] = True
            params['legend_alpha'] = 0.8
            params['legend_edgecolor'] = '#cccccc'
            params['legend_facecolor'] = '#ffffff'
            params['legend_shadow'] = False

    # ── Annotations & reference lines ────────────────────────────────────────
    with st.expander("Annotations & Reference Lines", expanded=False):
        params['show_r2_box'] = st.checkbox("Show metrics box on plot", True,
            help="Displays R² and fit quality as a text box inside the axes.")
        if params['show_r2_box']:
            params['r2_box_loc']      = st.selectbox("Metrics box corner",
                                                       ["upper left", "upper right",
                                                        "lower left", "lower right"], index=0)
            params['r2_box_fontsize'] = st.slider("Metrics box font size", 6, 18, 9)
            params['r2_box_bg']       = st.color_picker("Metrics box background", "#fffff0")
        else:
            params['r2_box_loc'] = "upper left"
            params['r2_box_fontsize'] = 9
            params['r2_box_bg'] = "#fffff0"

        st.markdown("**Horizontal reference line**")
        col1, col2 = st.columns(2)
        with col1:
            params['hline_enabled'] = st.checkbox("Add H-line", False)
            params['hline_y']       = st.number_input("Y position", value=0.0, format="%.4f",
                                                       key="hly") if params['hline_enabled'] else 0.0
        with col2:
            if params['hline_enabled']:
                params['hline_color'] = st.color_picker("H-line color", "#888888", key="hlc")
                params['hline_style'] = LINE_STYLES[st.selectbox("H-line style",
                                                     list(LINE_STYLES.keys()), index=1, key="hlls")]
                params['hline_width'] = st.slider("H-line width", 0.5, 4.0, 1.0, 0.25, key="hlw")
                params['hline_label'] = st.text_input("H-line label", value="", key="hll")
            else:
                params['hline_color'] = "#888888"
                params['hline_style'] = "--"
                params['hline_width'] = 1.0
                params['hline_label'] = ""

        st.markdown("**Vertical reference line**")
        col1, col2 = st.columns(2)
        with col1:
            params['vline_enabled'] = st.checkbox("Add V-line", False)
            params['vline_x']       = st.number_input("X position", value=0.0, format="%.4f",
                                                       key="vlx") if params['vline_enabled'] else 0.0
        with col2:
            if params['vline_enabled']:
                params['vline_color'] = st.color_picker("V-line color", "#888888", key="vlc")
                params['vline_style'] = LINE_STYLES[st.selectbox("V-line style",
                                                     list(LINE_STYLES.keys()), index=1, key="vlls")]
                params['vline_width'] = st.slider("V-line width", 0.5, 4.0, 1.0, 0.25, key="vlw")
                params['vline_label'] = st.text_input("V-line label", value="", key="vll")
            else:
                params['vline_color'] = "#888888"
                params['vline_style'] = "--"
                params['vline_width'] = 1.0
                params['vline_label'] = ""

    # ── Curve Smoothing ───────────────────────────────────────────────────────
    with st.expander("Curve Smoothing (fitted line rendering)", expanded=False):
        st.markdown(
            "Controls how the fitted/smoothed line is rendered — **independent of the fitting method itself**. "
            "Increasing resolution makes curves look crisper. Post-fit smoothing can remove jaggedness "
            "in noisy fits (e.g. Random Forest, Wavelet) without changing the underlying model."
        )
        col1, col2 = st.columns(2)
        with col1:
            params['curve_n_points'] = st.slider(
                "Curve resolution (points)", 50, 2000, 500, step=50,
                help="Number of X points used to draw the fitted line. Higher = smoother curves on screen."
            )
            params['post_smooth_method'] = st.selectbox(
                "Post-fit line smoothing",
                ["None", "Moving Average", "Gaussian", "Spline Resample"],
                index=0,
                help=(
                    "Applies a visual smoothing pass on top of the already-fitted curve.\n"
                    "• Moving Average – simple rolling mean, good for step-like artifacts.\n"
                    "• Gaussian – bell-curve weighted kernel, smooth and natural.\n"
                    "• Spline Resample – fits a cubic spline through the rendered points."
                )
            )
        with col2:
            if params['post_smooth_method'] != "None":
                if params['post_smooth_method'] == "Moving Average":
                    params['post_smooth_window'] = st.slider(
                        "Window size (pts)", 3, 201, 11, step=2,
                        help="Number of consecutive points averaged. Must be odd."
                    )
                    # enforce odd
                    if params['post_smooth_window'] % 2 == 0:
                        params['post_smooth_window'] += 1
                    params['post_smooth_sigma'] = 1.0   # unused
                    params['post_smooth_spline_s'] = 0.0
                elif params['post_smooth_method'] == "Gaussian":
                    params['post_smooth_sigma'] = st.slider(
                        "Gaussian sigma (pts)", 1.0, 100.0, 10.0, step=1.0,
                        help="Standard deviation of the Gaussian kernel in curve-point units. "
                             "Larger = smoother."
                    )
                    params['post_smooth_window'] = 5
                    params['post_smooth_spline_s'] = 0.0
                elif params['post_smooth_method'] == "Spline Resample":
                    params['post_smooth_spline_s'] = st.slider(
                        "Spline smoothing factor", 0.0, 1.0, 0.1, step=0.01,
                        help="0 = interpolate exactly, 1 = very smooth. Internally scaled to data range."
                    )
                    params['post_smooth_window'] = 5
                    params['post_smooth_sigma'] = 1.0
                params['post_smooth_preserve_endpoints'] = st.checkbox(
                    "Preserve endpoints", True,
                    help="Pin the first and last point of the smoothed line to the original values."
                )
            else:
                params['post_smooth_window'] = 5
                params['post_smooth_sigma'] = 1.0
                params['post_smooth_spline_s'] = 0.0
                params['post_smooth_preserve_endpoints'] = True

    # ── Export ────────────────────────────────────────────────────────────────
    with st.expander("Export Settings", expanded=False):
        params['export_format']      = st.selectbox("File format", EXPORT_FORMATS, index=0)
        params['export_transparent'] = st.checkbox("Transparent background (PNG/SVG)", False)
        params['export_bbox_inches'] = st.selectbox("Bounding box on save", ["tight", "standard"], index=0)

    return params


# ── Apply function ────────────────────────────────────────────────────────────

def apply_plot_customizations(fig, ax, params, r2=None, metrics_text=None):
    """
    Applies all visual customizations to an existing (fig, ax) pair.

    Args:
        fig:          matplotlib.figure.Figure
        ax:           matplotlib.axes.Axes
        params:       dict returned by visualization_ui()
        r2:           float | None  — shown in metrics box if metrics_text is None
        metrics_text: str | None    — full multi-line string for the metrics box;
                      overrides auto R² text if supplied
    """
    ff = params.get('font_family', 'sans-serif')
    plt.rcParams['font.family'] = ff

    # ── Figure background & size ──────────────────────────────────────────────
    fig.patch.set_facecolor(params.get('fig_bg_color', params.get('bg_color', '#ffffff')))
    fs = params.get('fig_size')
    if fs:
        fig.set_size_inches(fs)
    fig.set_dpi(params.get('dpi', 150))

    # ── Axes background ───────────────────────────────────────────────────────
    ax.set_facecolor(params.get('bg_color', '#ffffff'))

    # ── Axis scales ───────────────────────────────────────────────────────────
    x_scale = params.get('x_scale', 'linear')
    y_scale = params.get('y_scale', 'linear')
    try:
        if x_scale == 'symlog':
            ax.set_xscale('symlog', linthresh=params.get('x_sym_linthresh', 1.0))
        else:
            ax.set_xscale(x_scale)
    except Exception:
        ax.set_xscale('linear')
    try:
        if y_scale == 'symlog':
            ax.set_yscale('symlog', linthresh=params.get('y_sym_linthresh', 1.0))
        else:
            ax.set_yscale(y_scale)
    except Exception:
        ax.set_yscale('linear')

    # ── Axis limits ───────────────────────────────────────────────────────────
    x_min, x_max = params.get('x_min', 0.0), params.get('x_max', 0.0)
    y_min, y_max = params.get('y_min', 0.0), params.get('y_max', 0.0)
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    new_xmin = x_min if x_min != 0.0 else cur_xlim[0]
    new_xmax = x_max if x_max != 0.0 else cur_xlim[1]
    new_ymin = y_min if y_min != 0.0 else cur_ylim[0]
    new_ymax = y_max if y_max != 0.0 else cur_ylim[1]
    if new_xmin != cur_xlim[0] or new_xmax != cur_xlim[1]:
        ax.set_xlim(new_xmin, new_xmax)
    if new_ymin != cur_ylim[0] or new_ymax != cur_ylim[1]:
        ax.set_ylim(new_ymin, new_ymax)
    if params.get('invert_x'):
        ax.invert_xaxis()
    if params.get('invert_y'):
        ax.invert_yaxis()
    if params.get('aspect_ratio', 'auto') != 'auto':
        ax.set_aspect(params['aspect_ratio'])

    # ── Spines ───────────────────────────────────────────────────────────────
    axis_color = params.get('axis_color', '#000000')
    tick_width  = params.get('tick_width', 1.0)
    for side, spine in ax.spines.items():
        if side == 'top' and not params.get('show_top_spine', True):
            spine.set_visible(False)
        elif side == 'right' and not params.get('show_right_spine', True):
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_edgecolor(axis_color)
            spine.set_linewidth(tick_width)

    # ── Ticks ─────────────────────────────────────────────────────────────────
    tick_dir = params.get('tick_direction', 'out')
    tick_len = params.get('tick_length', 6)
    ax.tick_params(which='major', direction=tick_dir, length=tick_len,
                   width=tick_width, colors=axis_color,
                   labelsize=params.get('font_size', 11), labelcolor=axis_color)
    ax.tick_params(which='minor', direction=tick_dir,
                   length=max(2, tick_len // 2),
                   width=max(0.4, tick_width * 0.6), colors=axis_color)

    x_rot = params.get('x_tick_rotation', 0)
    y_rot = params.get('y_tick_rotation', 0)
    if x_rot:
        plt.setp(ax.get_xticklabels(), rotation=x_rot, ha='right')
    if y_rot:
        plt.setp(ax.get_yticklabels(), rotation=y_rot, ha='right')

    # ── Tick locators ─────────────────────────────────────────────────────────
    maj_x = params.get('major_x_step', 0.0)
    maj_y = params.get('major_y_step', 0.0)
    min_x = params.get('minor_x_step', 0.0)
    min_y = params.get('minor_y_step', 0.0)
    show_minor = params.get('show_minor_grid', False)

    if x_scale == 'linear':
        if maj_x and maj_x > 0:
            ax.xaxis.set_major_locator(MultipleLocator(maj_x))
        if min_x and min_x > 0:
            ax.xaxis.set_minor_locator(MultipleLocator(min_x))
        elif show_minor:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
    elif x_scale == 'log':
        ax.xaxis.set_major_locator(LogLocator())

    if y_scale == 'linear':
        if maj_y and maj_y > 0:
            ax.yaxis.set_major_locator(MultipleLocator(maj_y))
        if min_y and min_y > 0:
            ax.yaxis.set_minor_locator(MultipleLocator(min_y))
        elif show_minor:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
    elif y_scale == 'log':
        ax.yaxis.set_major_locator(LogLocator())

    # ── Grid ─────────────────────────────────────────────────────────────────
    grid_color = params.get('grid_color', '#cccccc')
    grid_ls    = LINE_STYLES.get(params.get('grid_linestyle', 'Solid'), '-')
    grid_lw    = params.get('grid_linewidth', 0.8)
    grid_alpha = params.get('grid_alpha', 0.7)

    ax.set_axisbelow(True)  # grid behind data
    if params.get('show_grid', True):
        ax.grid(True, which='major', color=grid_color, linestyle=grid_ls,
                linewidth=grid_lw, alpha=grid_alpha, zorder=0)
    else:
        ax.grid(False)

    if show_minor:
        ax.minorticks_on()
        ax.grid(True, which='minor', color=grid_color, linestyle=':',
                linewidth=max(0.2, grid_lw * 0.45),
                alpha=params.get('minor_grid_alpha', 0.35), zorder=0)

    # ── Title ─────────────────────────────────────────────────────────────────
    title = params.get('title', '')
    if title:
        ts = params.get('title_style', 'normal')
        ax.set_title(
            title,
            fontsize=params.get('title_size', 16),
            color=params.get('title_color', '#000000'),
            pad=params.get('title_pad', 10),
            fontfamily=ff,
            fontweight='bold' if 'bold' in ts else 'normal',
            fontstyle='italic' if 'italic' in ts else 'normal',
        )

    # ── Axis labels ───────────────────────────────────────────────────────────
    lkw = dict(fontsize=params.get('label_size', 13),
               color=params.get('label_color', '#000000'),
               fontfamily=ff)
    ax.set_xlabel(params.get('xlabel', 'X'), **lkw)
    ax.set_ylabel(params.get('ylabel', 'Y'), **lkw)

    # ── Line shadow ───────────────────────────────────────────────────────────
    if params.get('line_shadow', False):
        sw = params.get('shadow_width', 4)
        sc = params.get('shadow_color', '#888888')
        for artist in ax.get_lines():
            artist.set_path_effects([
                pe.withStroke(linewidth=artist.get_linewidth() + sw,
                              foreground=sc, alpha=0.45),
                pe.Normal()
            ])

    # ── Fill under fitted line ────────────────────────────────────────────────
    if params.get('fill_under_line', False):
        for artist in ax.get_lines():
            xd, yd = artist.get_xdata(), artist.get_ydata()
            if len(xd) > 1:
                ax.fill_between(
                    xd, yd,
                    alpha=params.get('fill_alpha', 0.2),
                    color=params.get('fill_color', '#ffcccc'),
                    zorder=int(params.get('line_zorder', 5)) - 1
                )
                break

    # ── Reference lines ───────────────────────────────────────────────────────
    if params.get('hline_enabled', False):
        lbl = params.get('hline_label', '') or None
        ax.axhline(y=params.get('hline_y', 0.0),
                   color=params.get('hline_color', '#888888'),
                   linestyle=params.get('hline_style', '--'),
                   linewidth=params.get('hline_width', 1.0),
                   label=lbl, zorder=2)
    if params.get('vline_enabled', False):
        lbl = params.get('vline_label', '') or None
        ax.axvline(x=params.get('vline_x', 0.0),
                   color=params.get('vline_color', '#888888'),
                   linestyle=params.get('vline_style', '--'),
                   linewidth=params.get('vline_width', 1.0),
                   label=lbl, zorder=2)

    # ── Metrics / R² box ─────────────────────────────────────────────────────
    if params.get('show_r2_box', True) and (r2 is not None or metrics_text is not None):
        loc_coords = {
            "upper left":  (0.03, 0.97, 'top',    'left'),
            "upper right": (0.97, 0.97, 'top',    'right'),
            "lower left":  (0.03, 0.03, 'bottom', 'left'),
            "lower right": (0.97, 0.03, 'bottom', 'right'),
        }
        bx, by, bva, bha = loc_coords.get(
            params.get('r2_box_loc', 'upper left'), (0.03, 0.97, 'top', 'left'))

        text = metrics_text if metrics_text is not None else f"R² = {r2:.5f}"
        ax.text(
            bx, by, text,
            transform=ax.transAxes,
            fontsize=params.get('r2_box_fontsize', 9),
            verticalalignment=bva,
            horizontalalignment=bha,
            fontfamily=ff,
            color=params.get('axis_color', '#000000'),
            bbox=dict(boxstyle='round,pad=0.45',
                      facecolor=params.get('r2_box_bg', '#fffff0'),
                      edgecolor=params.get('axis_color', '#000000'),
                      alpha=0.85),
            zorder=10,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    if params.get('show_legend', True):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            loc = params.get('legend_loc', 'best')
            leg_kw = dict(
                fontsize=params.get('legend_fontsize', 10),
                ncols=params.get('legend_ncol', 1),
                frameon=params.get('legend_frameon', True),
                framealpha=params.get('legend_alpha', 0.8),
                edgecolor=params.get('legend_edgecolor', '#cccccc'),
                facecolor=params.get('legend_facecolor', '#ffffff'),
                shadow=params.get('legend_shadow', False),
            )
            legend_title = params.get('legend_title', '')
            if legend_title:
                leg_kw['title'] = legend_title
            if loc == "outside right":
                ax.legend(handles, labels, bbox_to_anchor=(1.02, 1),
                          loc='upper left', **leg_kw)
            elif loc == "outside bottom":
                ax.legend(handles, labels, bbox_to_anchor=(0.5, -0.18),
                          loc='upper center', **leg_kw)
            else:
                ax.legend(handles, labels, loc=loc, **leg_kw)
    else:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    # ── Layout ────────────────────────────────────────────────────────────────
    if params.get('constrained_layout', False):
        fig.set_constrained_layout(True)
    elif params.get('tight_layout', True):
        try:
            fig.tight_layout()
        except Exception:
            pass
    else:
        fig.subplots_adjust(
            left=params.get('subplot_left', 0.08),
            right=params.get('subplot_right', 0.95),
            top=params.get('subplot_top', 0.92),
            bottom=params.get('subplot_bottom', 0.10),
        )



# ── Curve smoothing helper ────────────────────────────────────────────────────

def smooth_rendered_line(x, y, params):
    """
    Applies a post-fit visual smoothing pass to already-computed (x, y) curve arrays.
    This does NOT change the underlying model — it only affects rendering.

    Args:
        x, y:   numpy arrays of the rendered fitted line points
        params: dict from visualization_ui()

    Returns:
        x_out, y_out — smoothed arrays (x unchanged for Moving Average / Gaussian;
                        resampled for Spline Resample)
    """
    method = params.get('post_smooth_method', 'None')
    if method == 'None' or len(x) < 5:
        return x, y

    preserve = params.get('post_smooth_preserve_endpoints', True)
    y_first, y_last = y[0], y[-1]

    if method == 'Moving Average':
        w = int(params.get('post_smooth_window', 11))
        w = max(3, w if w % 2 == 1 else w + 1)
        pad = w // 2
        # reflect-pad to avoid edge shrinkage
        y_pad = np.pad(y, pad, mode='reflect')
        kernel = np.ones(w) / w
        y_out = np.convolve(y_pad, kernel, mode='valid')
        # trim back to original length
        y_out = y_out[:len(y)]
        x_out = x

    elif method == 'Gaussian':
        from scipy.ndimage import gaussian_filter1d
        sigma = float(params.get('post_smooth_sigma', 10.0))
        y_out = gaussian_filter1d(y, sigma=sigma, mode='reflect')
        x_out = x

    elif method == 'Spline Resample':
        from scipy.interpolate import UnivariateSpline
        s_factor = float(params.get('post_smooth_spline_s', 0.1))
        # scale s by data variance so 0–1 maps meaningfully
        s_scaled = s_factor * len(x) * float(np.var(y))
        try:
            spl = UnivariateSpline(x, y, s=s_scaled, k=3)
            x_out = np.linspace(x[0], x[-1], len(x))
            y_out = spl(x_out)
        except Exception:
            # fall back to no smoothing if spline fails (e.g. non-monotonic x)
            x_out, y_out = x, y

    else:
        return x, y

    if preserve:
        y_out[0]  = y_first
        y_out[-1] = y_last

    return x_out, y_out




def export_figure(fig, params):
    """
    Serialize the figure to a BytesIO buffer in the chosen format.

    Returns:
        (buf, mime_type, file_extension)
    """
    fmt         = params.get('export_format', 'png')
    transparent = params.get('export_transparent', False)
    bbox        = params.get('export_bbox_inches', 'tight')
    dpi         = params.get('dpi', 150)

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, transparent=transparent,
                bbox_inches=bbox if bbox == 'tight' else None)
    buf.seek(0)

    mime_map = {
        'png': 'image/png',
        'pdf': 'application/pdf',
        'svg': 'image/svg+xml',
        'eps': 'application/postscript',
    }
    return buf, mime_map.get(fmt, 'application/octet-stream'), fmt


def show_export_button(fig, params, filename_stem="plot"):
    """
    Renders a Streamlit download button for the current figure.
    Call this after st.pyplot(fig).
    """
    buf, mime, ext = export_figure(fig, params)
    st.download_button(
        label=f"Download plot as .{ext}",
        data=buf,
        file_name=f"{filename_stem}.{ext}",
        mime=mime,
    )
