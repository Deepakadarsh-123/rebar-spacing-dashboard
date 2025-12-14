import os
import time
import csv
import base64
import threading
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

###############################################################################
# GLOBAL PARAMETERS & DIRECTORIES
###############################################################################
PLY_FOLDER = r"D:\1. Adarsh\COMPILED DATA\PLY Files"  # Folder containing .ply files
LOG_FILE = r"analysis_log.csv"                        # Log file path
if not os.path.exists(PLY_FOLDER):
    raise FileNotFoundError(f"PLY folder not found: {PLY_FOLDER}")

# For KMeans splitting of layers (Z-axis)
MIN_K = 2
MAX_K = 10
DEFAULT_K = 2

# Scale factor (e.g., converting mm to m)
SCALE_FACTOR = 0.001

# Detection parameters for histogram-based spacing detection
DEFAULT_BINS = 1000         # Number of histogram bins
DEFAULT_SIGMA = 1           # Gaussian smoothing sigma
DEFAULT_PEAK_DIST = 21      # Minimum peak separation (bins)
DEFAULT_MIN_POINTS = 50     # Minimum count for a valid peak

# Pre-defined colors for layers (RGB values in [0,1])
layer_colors = {
    "Layer 1": [1, 0, 0],
    "Layer 2": [0, 1, 0],
    "Layer 3": [0, 0, 1],
    "Layer 4": [1, 1, 0],
    "Layer 5": [1, 0, 1],
    "Layer 6": [0, 1, 1]
}

# Global variable to store last model/KMeans for Open3D viewer
last_o3d_params = None

###############################################################################
# UTILITY FUNCTIONS
###############################################################################
def list_available_ply_files():
    """Return a list of .ply files in PLY_FOLDER."""
    files = [f for f in os.listdir(PLY_FOLDER) if f.lower().endswith('.ply')]
    return [{"label": f, "value": f} for f in files]

def load_point_cloud(file_name):
    """Load a .ply file and return Nx3 numpy array."""
    file_path = os.path.join(PLY_FOLDER, file_name)
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

###############################################################################
# LAYER SPLITTING & PCA
###############################################################################
def split_layers_kmeans(points, n_layers):
    """Split the point cloud into n_layers using KMeans on Z."""
    z_vals = points[:, 2].reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_layers, random_state=42).fit(z_vals)
    labels = kmeans.labels_
    layers = {}
    for i in range(n_layers):
        layers[i] = points[labels == i]
    sorted_indices = sorted(layers.keys(), key=lambda i: layers[i][:,2].mean(), reverse=True)
    layer_dict = {}
    for idx, key in enumerate(sorted_indices, start=1):
        layer_dict[f"Layer {idx}"] = layers[key]
    return layer_dict

def align_layer_with_pca(layer_points_3d):
    pca = PCA(n_components=3)
    pca.fit(layer_points_3d)
    centroid = pca.mean_
    centered = layer_points_3d - centroid
    reoriented = centered @ pca.components_.T
    return reoriented

###############################################################################
# RANSAC-BASED 2D ALIGNMENT
###############################################################################
def align_points_ransac(points_2d):
    """Use RANSAC to rotate so the best-fit line is vertical (Y)."""
    X = points_2d[:, 0].reshape(-1, 1)
    y = points_2d[:, 1]
    ransac = RANSACRegressor(LinearRegression(), residual_threshold=0.01, random_state=42)
    ransac.fit(X, y)
    slope = ransac.estimator_.coef_[0]
    estimated_angle = np.arctan(slope)
    correction_angle = np.pi/2 - estimated_angle
    R = np.array([[np.cos(correction_angle), -np.sin(correction_angle)],
                  [np.sin(correction_angle),  np.cos(correction_angle)]])
    aligned = points_2d @ R.T
    return aligned, correction_angle

###############################################################################
# SPACING DETECTION
###############################################################################
def detect_rebars_no_pca(pts_2d, bins=DEFAULT_BINS, smooth_sigma=DEFAULT_SIGMA,
                         peak_distance=DEFAULT_PEAK_DIST, min_points=DEFAULT_MIN_POINTS,
                         projection_axis=0):
    data = pts_2d[:, projection_axis]
    proj_min = data.min()
    shifted = data - proj_min
    hist, edges = np.histogram(shifted, bins=bins)
    smooth_density = gaussian_filter1d(hist, sigma=smooth_sigma)
    threshold = np.median(smooth_density) + np.std(smooth_density)
    peaks, _ = find_peaks(smooth_density, height=threshold, distance=peak_distance)
    valid_peaks = [p for p in peaks if (min_points == 0 or hist[p] >= min_points)]
    peak_positions = np.array([(edges[i] + edges[i+1]) / 2 for i in valid_peaks])
    spacings = np.diff(peak_positions)
    return edges, smooth_density, peak_positions, spacings, proj_min

###############################################################################
# FWHM-BASED DIAMETER ESTIMATION (for Dashboard)
###############################################################################
def estimate_peak_width(bin_centers, density, peak_index):
    peak_value = density[peak_index]
    half_max = peak_value / 2.0
    left = peak_index
    while left > 0 and density[left] > half_max:
        left -= 1
    left_interp = bin_centers[0] if left == 0 else bin_centers[left] + (bin_centers[left+1]-bin_centers[left]) * ((half_max - density[left])/(density[left+1]-density[left]))
    right = peak_index
    while right < len(density)-1 and density[right] > half_max:
        right += 1
    right_interp = bin_centers[-1] if right == len(density)-1 else bin_centers[right-1] + (bin_centers[right]-bin_centers[right-1]) * ((density[right-1]-half_max)/(density[right-1]-density[right]))
    return right_interp - left_interp

def compute_diameters_from_histogram(edges, density, peak_positions, proj_min):
    bin_centers = 0.5 * (edges[:-1] + edges[1:]) + proj_min
    diameters = []
    for peak in peak_positions:
        idx = np.argmin(np.abs(bin_centers - (peak + proj_min)))
        width = estimate_peak_width(bin_centers, density, idx)
        diameters.append(width)
    return diameters

###############################################################################
# CREATE SPACING & DIAMETER FIGURES
###############################################################################
def create_spacing_figure(pts_2d, bins, smooth_sigma, peak_distance, min_points,
                          projection_axis=0, swap_view=False):
    edges, smooth_density, peak_positions, spacings, proj_min = detect_rebars_no_pca(
        pts_2d, bins, smooth_sigma, peak_distance, min_points, projection_axis=projection_axis
    )
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Plan View (Final Rotated Coordinates)", "Rebar Density Curve"),
                        vertical_spacing=0.1)
    fig.update_layout(width=1200, height=800)

    if not swap_view:
        plan_x = pts_2d[:, 0]
        plan_y = pts_2d[:, 1]
    else:
        plan_x = pts_2d[:, 1]
        plan_y = pts_2d[:, 0]

    fig.add_trace(go.Scatter(x=plan_x, y=plan_y, mode="markers",
                             marker=dict(size=2, color="gray"), name="Points"), row=1, col=1)

    other_dim = 1 - projection_axis
    data = pts_2d[:, projection_axis]
    half_bin = (data.max() - data.min()) / bins / 2.0
    peak_plan_x = []
    peak_plan_y = []
    for i, peak_val in enumerate(peak_positions, start=1):
        actual_val = peak_val + proj_min
        mask = np.abs(data - actual_val) < half_bin
        if not np.any(mask):
            continue
        local_median = np.median(pts_2d[mask, other_dim])
        if not swap_view:
            if projection_axis == 0:
                px, py = actual_val, local_median
            else:
                px, py = local_median, actual_val
        else:
            if projection_axis == 0:
                px, py = local_median, actual_val
            else:
                px, py = actual_val, local_median
        peak_plan_x.append(px)
        peak_plan_y.append(py)

    fig.add_trace(go.Scatter(x=peak_plan_x, y=peak_plan_y,
                             mode="markers+text",
                             marker=dict(size=8, color="red"),
                             text=[f"R{i}" for i in range(1, len(peak_plan_x)+1)],
                             textposition="top center", name="Detected Peaks"),
                  row=1, col=1)

    bin_centers = 0.5 * (edges[:-1] + edges[1:]) + proj_min
    fig.add_trace(go.Scatter(x=bin_centers, y=smooth_density, mode="lines", name="Density"),
                  row=2, col=1)

    offset_fraction = 0.05
    mid_edges = 0.5 * (edges[:-1] + edges[1:])
    peak_density = [smooth_density[np.argmin(np.abs(mid_edges - (p+proj_min)))] for p in peak_positions]
    label_y = [val + offset_fraction*val for val in peak_density]
    fig.add_trace(go.Scatter(x=peak_positions + proj_min, y=label_y,
                             mode="markers+text", marker=dict(size=8, color="red"),
                             text=[f"R{i}" for i in range(1, len(peak_positions)+1)],
                             textposition="top center", name="Peaks"),
                  row=2, col=1)

    if not swap_view:
        xmin, xmax = plan_x.min(), plan_x.max()
        fig.update_xaxes(title_text="X (Final Rotated)", row=2, col=1, range=[xmin, xmax])
        fig.update_yaxes(title_text="Y", row=1, col=1)
    else:
        ymin, ymax = plan_y.min(), plan_y.max()
        fig.update_xaxes(title_text="Y (Swapped View)", row=2, col=1, range=[ymin, ymax])
        fig.update_yaxes(title_text="X", row=1, col=1)

    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_layout(title_text="Combined Plan View and Density Curve", showlegend=True)

    return fig, spacings

def create_diameter_figure(pts_2d, bins, smooth_sigma, peak_distance, min_points,
                           projection_axis=0, swap_view=False):
    edges, smooth_density, peak_positions, _, proj_min = detect_rebars_no_pca(
        pts_2d, bins, smooth_sigma, peak_distance, min_points, projection_axis=projection_axis
    )
    diameter_estimates = compute_diameters_from_histogram(edges, smooth_density, peak_positions, proj_min)
    fig = go.Figure()
    bin_centers = 0.5 * (edges[:-1] + edges[1:]) + proj_min
    fig.add_trace(go.Scatter(x=bin_centers, y=smooth_density, mode="lines", name="Density"))
    offset_fraction = 0.05
    peak_abs = peak_positions + proj_min
    label_y = []
    for peak in peak_positions:
        idx = np.argmin(np.abs(bin_centers - (peak + proj_min)))
        base_val = smooth_density[idx]
        label_y.append(base_val + offset_fraction*base_val)
    fig.add_trace(go.Scatter(x=peak_abs, y=label_y,
                             mode="markers+text", marker=dict(size=8, color="red"),
                             text=[f"{round(d*1000,1)} mm" for d in diameter_estimates],
                             textposition="top center", name="Estimated Diameters"))
    fig.update_layout(title="Histogram-based Diameter Estimation",
                      xaxis_title="Projection coordinate (m)",
                      yaxis_title="Density",
                      width=800, height=500)
    diameter_table = [{"RebarID": f"R{i}", "Diameter_mm": round(d * 1000, 1)}
                      for i, d in enumerate(diameter_estimates, start=1)]
    return fig, diameter_table

###############################################################################
# OPEN3D VIEWER
###############################################################################
def launch_o3d_window_multiple(pcd_list, window_name):
    o3d.visualization.draw_geometries(pcd_list, window_name=window_name)

def start_o3d_viewer_multiple(pcd_list, window_name):
    thread = threading.Thread(target=launch_o3d_window_multiple, args=(pcd_list, window_name))
    thread.daemon = True
    thread.start()

###############################################################################
# LOGGING
###############################################################################
def log_analysis(params_dict, processing_time):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    header = ["timestamp", "model_file", "kmeans_clusters", "selected_layer",
              "manual_offset", "bins", "sigma", "peak_distance", "min_points", "swap_axis", "processing_time_sec"]
    new_row = [timestamp,
               params_dict.get("model_file", ""),
               params_dict.get("kmeans_clusters", ""),
               params_dict.get("selected_layer", ""),
               params_dict.get("manual_offset", ""),
               params_dict.get("bins", ""),
               params_dict.get("sigma", ""),
               params_dict.get("peak_distance", ""),
               params_dict.get("min_points", ""),
               params_dict.get("swap_axis", ""),
               processing_time]
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(new_row)

def fig_to_base64(fig):
    """Convert a Plotly figure to base64 PNG string."""
    img_bytes = fig.to_image(format="png")
    encoded = base64.b64encode(img_bytes).decode("ascii")
    return "data:image/png;base64," + encoded

###############################################################################
# DASH APP LAYOUT
###############################################################################
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = html.Div([
    dcc.Tabs(id="tabs", value="dashboard", children=[
        # ------------------- DASHBOARD TAB ------------------- #
        dcc.Tab(label="Dashboard", value="dashboard", children=[
            html.Div([
                html.Label("Select Model File:"),
                dcc.Dropdown(
                    id="model-dropdown",
                    options=list_available_ply_files(),
                    value=(list_available_ply_files()[0]["value"] if list_available_ply_files() else None),
                    style={"width": "300px"}
                )
            ], style={"margin": "20px", "border": "1px solid #888", "padding": "10px"}),
            html.Div([
                html.Div([
                    html.Label("Select KMeans Clusters (layers):"),
                    dcc.Slider(
                        id="kmeans-slider",
                        min=MIN_K,
                        max=MAX_K,
                        step=1,
                        value=DEFAULT_K,
                        marks={i: str(i) for i in range(MIN_K, MAX_K+1)}
                    )
                ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top", "marginRight": "40px"}),
                html.Div([
                    html.Label("Select Layer for Analysis:"),
                    dcc.Dropdown(id="layer-dropdown-analysis", style={"width": "300px"})
                ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top"})
            ], style={"margin": "20px"}),
            html.Div([
                dcc.Checklist(
                    id="swap-axis",
                    options=[{"label": "Swap Projection Axis", "value": "swap"}],
                    value=[]
                )
            ], style={"margin": "20px"}),
            html.Div(id="conversion-info", style={"margin": "20px", "fontWeight": "bold", "fontSize": "14px"}),
            html.Div([
                html.Label("Manual Rotation Offset (degrees):"),
                dcc.Slider(
                    id="rotation-slider",
                    min=-10,
                    max=10,
                    step=0.5,
                    value=0,
                    marks={i: str(i) for i in range(-10, 11, 2)}
                ),
                html.Div([
                    html.Label("Or type offset (degrees):"),
                    dcc.Input(id="rotation-input", type="number", min=-10, max=10, step=0.1, value=0, style={"width": "100px"})
                ], style={"display": "inline-block", "marginLeft": "20px"}),
                html.Div(id="rotation-display", style={"marginTop": "10px", "fontWeight": "bold"})
            ], style={"margin": "20px", "width": "500px"}),
            html.Div([
                html.H3("Spacing Detection Parameters"),
                html.Div([
                    html.Label("Histogram Bins (approx. bin width in mm will be shown):"),
                    dcc.Slider(
                        id="bins-slider",
                        min=100,
                        max=1000,
                        step=50,
                        value=DEFAULT_BINS,
                        marks={i: str(i) for i in range(100, 1100, 100)}
                    )
                ], style={"margin": "10px"}),
                html.Div([
                    html.Label("Gaussian Smoothing Sigma:"),
                    dcc.Slider(
                        id="sigma-slider",
                        min=0,
                        max=10,
                        step=0.5,
                        value=DEFAULT_SIGMA,
                        marks={i: str(i) for i in range(0, 11)}
                    )
                ], style={"margin": "10px"}),
                html.Div([
                    html.Label("Min Peak Separation (bins) (approx. separation in mm will be shown):"),
                    dcc.Slider(
                        id="peak-dist-slider",
                        min=1,
                        max=50,
                        step=1,
                        value=DEFAULT_PEAK_DIST,
                        marks={i: str(i) for i in range(1, 51, 5)}
                    )
                ], style={"margin": "10px"}),
                html.Div([
                    html.Label("Min Points per Peak:"),
                    dcc.Slider(
                        id="min-points-slider",
                        min=0,
                        max=500,
                        step=10,
                        value=DEFAULT_MIN_POINTS,
                        marks={i: str(i) for i in range(0, 501, 50)}
                    )
                ], style={"margin": "10px"}),
                html.Div([
                    dcc.Checklist(
                        id="disable-minpoints",
                        options=[{"label": "Disable Minimum Points Constraint", "value": "disable"}],
                        value=[]
                    )
                ], style={"margin": "10px"})
            ], style={"border": "1px solid #ccc", "padding": "10px", "margin": "20px"}),
            html.Div([
                html.H3("(Optional) Additional DBSCAN Parameters (Not used)"),
                html.Div([
                    html.Label("eps (m):"),
                    dcc.Input(id="dbscan-eps", type="number", min=0.005, max=0.1, step=0.005, value=0.02, style={"width": "100px"})
                ], style={"display": "inline-block", "marginRight": "40px"}),
                html.Div([
                    html.Label("min_samples:"),
                    dcc.Input(id="dbscan-min", type="number", min=5, max=100, step=1, value=20, style={"width": "100px"})
                ], style={"display": "inline-block"})
            ], style={"border": "1px solid #ccc", "padding": "10px", "margin": "20px"}),
            html.Button("Run Analysis", id="run-analysis", n_clicks=0, style={"margin": "20px"}),
            dcc.Loading(id="loading-spacing-graph", type="default", children=dcc.Graph(id="spacing-graph")),
            html.H2("Spacing Table"),
            dcc.Loading(id="loading-spacing-table", type="default", children=dash_table.DataTable(
                id="spacing-table",
                columns=[
                    {"name": "Rebar Pairs", "id": "RebarPair"},
                    {"name": "Spacing (mm)", "id": "Spacing_mm"}
                ],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center"},
                page_size=10
            )),
            dcc.Loading(id="loading-diameter-graph", type="default", children=dcc.Graph(id="diameter-graph")),
            html.H2("Diameter Table"),
            dcc.Loading(id="loading-diameter-table", type="default", children=dash_table.DataTable(
                id="diameter-table",
                columns=[
                    {"name": "RebarID", "id": "RebarID"},
                    {"name": "Diameter (mm)", "id": "Diameter_mm"}
                ],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center"},
                page_size=10
            )),
            html.H2("O3D Viewer Info"),
            html.Div(id="o3d-info", style={"margin": "20px", "fontWeight": "bold", "fontSize": "16px"})
        ]),
        # ------------------- ANALYSIS TAB ------------------- #
        dcc.Tab(label="Analysis", value="analysis", children=[
            html.H1("Planned vs. Actual by Each Layer"),
            html.Div(id="analysis-content", style={"margin": "20px"}),
            html.Button("Copy to All", id="copy-all", n_clicks=0, style={"margin": "20px"}),
            html.Button("Calculate Error", id="calc-error", n_clicks=0, style={"margin": "20px"}),
            dcc.Graph(id="error-boxplot"),
            html.Div(id="error-summary", style={"margin": "20px", "fontWeight": "bold"}),
            dash_table.DataTable(
                id="planned-spacing-table",
                columns=[
                    {"name": "Rebar Pairs",      "id": "RebarPair",    "editable": False},
                    {"name": "Observed Spacing", "id": "Observed",     "editable": False},
                    {"name": "Actual Spacing",   "id": "Actual",       "editable": True},
                    {"name": "Error",            "id": "Error",        "editable": False},
                    {"name": "Abs. Error",       "id": "AbsError",     "editable": False},
                    {"name": "% Error",          "id": "ErrorPct",     "editable": False},
                    {"name": "% Abs. Error",     "id": "AbsErrorPct",  "editable": False}
                ],
                data=[],
                editable=True,
                row_deletable=False,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center"}
            )
        ])
    ]),
    dcc.Store(id="latest-run", storage_type="memory")
])

###############################################################################
# CALLBACK: Sync Rotation Slider and Input
###############################################################################
@app.callback(
    [Output("rotation-slider", "value"),
     Output("rotation-input", "value"),
     Output("rotation-display", "children")],
    [Input("rotation-slider", "value"),
     Input("rotation-input", "value")]
)
def sync_rotation(slider_val, input_val):
    ctx = dash.callback_context
    if not ctx.triggered:
        value = slider_val if slider_val is not None else 0
    else:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == "rotation-slider":
            value = slider_val
        else:
            try:
                value = float(input_val)
            except Exception:
                value = slider_val
    value = max(-10, min(10, value))
    return value, value, f"Final Manual Offset: {value}°"

###############################################################################
# CALLBACK: Update Analysis Layer Dropdown
###############################################################################
@app.callback(
    [Output("layer-dropdown-analysis", "options"),
     Output("layer-dropdown-analysis", "value")],
    [Input("model-dropdown", "value"),
     Input("kmeans-slider", "value")]
)
def update_layer_dropdown(model_file, k_val):
    if not model_file:
        return [], None
    pts = load_point_cloud(model_file)
    pts *= SCALE_FACTOR
    layers_dict = split_layers_kmeans(pts, n_layers=k_val)
    opts = [{"label": k, "value": k} for k in layers_dict.keys()]
    return opts, (opts[0]["value"] if opts else None)

###############################################################################
# CALLBACK: Dashboard -> Store (Process All Layers)
###############################################################################
@app.callback(
    [Output("spacing-graph", "figure"),
     Output("spacing-table", "data"),
     Output("diameter-graph", "figure"),
     Output("diameter-table", "data"),
     Output("o3d-info", "children"),
     Output("conversion-info", "children"),
     Output("latest-run", "data")],
    [Input("run-analysis", "n_clicks"),
     Input("model-dropdown", "value"),
     Input("kmeans-slider", "value"),
     Input("layer-dropdown-analysis", "value"),
     Input("rotation-slider", "value"),
     Input("bins-slider", "value"),
     Input("sigma-slider", "value"),
     Input("peak-dist-slider", "value"),
     Input("min-points-slider", "value"),
     Input("disable-minpoints", "value"),
     Input("swap-axis", "value")]
)
def update_dashboard(n_clicks, model_file, k_val, selected_layer,
                     manual_offset, bins, sigma, peak_dist, min_pts,
                     disable_minpoints, swap_axis):
    global last_o3d_params
    start_time = time.time()
    if not model_file:
        return go.Figure(), [], go.Figure(), [], "", "", {}

    pts = load_point_cloud(model_file)
    pts *= SCALE_FACTOR

    layers_dict = split_layers_kmeans(pts, n_layers=k_val)

    new_o3d_params = {"model": model_file, "kmeans": k_val}
    if (last_o3d_params is None) or (last_o3d_params["model"] != model_file) or (last_o3d_params["kmeans"] != k_val):
        pcd_list = []
        o3d_info_str = "Complete Layers: "
        for layer_label, layer_pts in layers_dict.items():
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(layer_pts)
            color = layer_colors.get(layer_label, [0.5, 0.5, 0.5])
            pcd.paint_uniform_color(color)
            pcd_list.append(pcd)
            o3d_info_str += f"{layer_label} color {color}; "
        start_o3d_viewer_multiple(pcd_list, "Complete Layers")
        last_o3d_params = new_o3d_params
    else:
        o3d_info_str = "Using previously opened O3D viewer for complete layers."

    if selected_layer not in layers_dict:
        return go.Figure(), [], go.Figure(), [], o3d_info_str, "", {}

    all_layers_data = []
    for layer_name, layer_pts in layers_dict.items():
        reoriented_3d = align_layer_with_pca(layer_pts)
        pts_2d = reoriented_3d[:, :2]
        ransac_aligned, _ = align_points_ransac(pts_2d)
        theta_manual = np.deg2rad(manual_offset)
        R_manual = np.array([
            [np.cos(theta_manual), -np.sin(theta_manual)],
            [np.sin(theta_manual),  np.cos(theta_manual)]
        ])
        final_pts = ransac_aligned @ R_manual.T
        swap_view_bool = ("swap" in swap_axis)
        effective_min = 0 if ("disable" in disable_minpoints) else min_pts
        projection_axis = 1 if swap_view_bool else 0
        spacing_fig_tmp, spacings = create_spacing_figure(
            final_pts, bins, sigma, peak_dist, effective_min,
            projection_axis=projection_axis,
            swap_view=swap_view_bool
        )
        swap_str = "Yes" if swap_view_bool else "No"
        spacing_table_tmp = []
        spacing_table_tmp.append({"RebarPair": f"{layer_name} (SwapAxis={swap_str})",
                                  "Spacing_mm": ""})
        for i, sp in enumerate(spacings, start=1):
            spacing_table_tmp.append({
                "RebarPair": f"R{i}-R{i+1}",
                "Spacing_mm": round(sp * 1000, 2)
            })
        layer_result = {
            "layer_name": layer_name,
            "swap_axis": swap_str,
            "spacing_table": spacing_table_tmp
        }
        all_layers_data.append(layer_result)

    selected_layer_pts = layers_dict[selected_layer]
    reoriented_3d_sel = align_layer_with_pca(selected_layer_pts)
    pts_2d_sel = reoriented_3d_sel[:, :2]
    ransac_aligned_sel, _ = align_points_ransac(pts_2d_sel)
    theta_manual_sel = np.deg2rad(manual_offset)
    R_manual_sel = np.array([
        [np.cos(theta_manual_sel), -np.sin(theta_manual_sel)],
        [np.sin(theta_manual_sel),  np.cos(theta_manual_sel)]
    ])
    final_pts_sel = ransac_aligned_sel @ R_manual_sel.T
    swap_view_sel = ("swap" in swap_axis)
    proj_axis_sel = 1 if swap_view_sel else 0
    eff_min_sel = 0 if ("disable" in disable_minpoints) else min_pts
    spacing_fig, spacings_sel = create_spacing_figure(
        final_pts_sel, bins, sigma, peak_dist, eff_min_sel,
        projection_axis=proj_axis_sel,
        swap_view=swap_view_sel
    )
    spacing_table = []
    spacing_table.append({"RebarPair": f"{selected_layer}", "Spacing_mm": ""})
    for i, sp in enumerate(spacings_sel, start=1):
        spacing_table.append({
            "RebarPair": f"R{i}-R{i+1}",
            "Spacing_mm": round(sp*1000, 2)
        })
    diam_fig, diam_table = create_diameter_figure(
        final_pts_sel, bins, sigma, peak_dist, eff_min_sel,
        projection_axis=proj_axis_sel,
        swap_view=swap_view_sel
    )
    data_proj = final_pts_sel[:, proj_axis_sel]
    proj_range = data_proj.max() - data_proj.min()
    bin_width_mm = (proj_range / bins) * 1000
    peak_sep_mm = (proj_range / bins * peak_dist) * 1000
    conv_info = (f"Approx. bin width: {bin_width_mm:.1f} mm, "
                 f"Min peak separation: {peak_sep_mm:.1f} mm, "
                 f"SwapAxis: {'Yes' if swap_view_sel else 'No'}")
    processing_time = time.time() - start_time
    params = {
        "model_file": model_file,
        "kmeans_clusters": k_val,
        "selected_layer": selected_layer,
        "manual_offset": manual_offset,
        "bins": bins,
        "sigma": sigma,
        "peak_distance": peak_dist,
        "min_points": min_pts,
        "swap_axis": "Yes" if swap_view_sel else "No"
    }
    log_analysis(params, processing_time)
    spacing_fig_b64 = fig_to_base64(spacing_fig)
    latest_data = {
        "params": params,
        "processing_time_sec": processing_time,
        "all_layers": all_layers_data,
        "spacing_fig": spacing_fig_b64,
        "spacing_table_sel": spacing_table
    }
    return (spacing_fig, spacing_table,
            diam_fig, diam_table,
            o3d_info_str, conv_info,
            latest_data)

###############################################################################
# CALLBACK: Analysis Tab - Display Analysis and Grouped Table
###############################################################################
@app.callback(
    [Output("analysis-content", "children"),
     Output("planned-spacing-table", "data")],
    [Input("tabs", "value"),
     Input("latest-run", "data"),
     Input("copy-all", "n_clicks"),
     Input("calc-error", "n_clicks")],
    [State("planned-spacing-table", "data")]
)
def display_analysis(tab_value, latest_data, copy_n_clicks, calc_n_clicks, planned_data):
    ctx = dash.callback_context
    if tab_value != "analysis" or not latest_data:
        return ("No analysis data available for this session.", [])
    all_layers = latest_data.get("all_layers", [])
    if not all_layers:
        return ("No multi-layer data was captured.", [])
    combined_rows = []
    # If there is previously entered table data, preserve "Actual" spacing by matching RebarPair.
    if planned_data:
        user_data = {row["RebarPair"]: row.get("Actual", "") for row in planned_data if row.get("Observed")}
    else:
        user_data = {}
    for layer_info in all_layers:
        layer_name = layer_info["layer_name"]
        swap_str   = layer_info["swap_axis"]
        spacing_table = layer_info["spacing_table"]
        # Insert header for the layer
        combined_rows.append({
            "RebarPair": f"{layer_name} (SwapAxis={swap_str})",
            "Observed": "",
            "Actual": "",
            "Error": "",
            "AbsError": "",
            "ErrorPct": "",
            "AbsErrorPct": ""
        })
        for row in spacing_table:
            rp = row["RebarPair"]
            obs_val = row["Spacing_mm"]
            # If a user entry was saved, use it; otherwise start blank.
            actual_val = user_data.get(rp, "")
            new_row = {
                "RebarPair": rp,
                "Observed": obs_val,
                "Actual": actual_val,
                "Error": "",
                "AbsError": "",
                "ErrorPct": "",
                "AbsErrorPct": ""
            }
            combined_rows.append(new_row)
    # Handle "Copy to All" button
    if copy_n_clicks:
        first_val = None
        for row in combined_rows:
            if row["Observed"]:
                first_val = row["Actual"]
                break
        if first_val is not None:
            for row in combined_rows:
                if row["Observed"]:
                    row["Actual"] = first_val
    # Handle "Calculate Error" button:
    if calc_n_clicks:
        for row in combined_rows:
            if not row["Observed"]:
                continue  # header row skip
            try:
                obs = float(row["Observed"])
                act = float(row["Actual"])
                err = obs - act
                row["Error"] = f"{err:.2f}"
                row["AbsError"] = f"{abs(err):.2f}"
                if act != 0:
                    pct = (err / act) * 100.0
                    row["ErrorPct"] = f"{pct:.2f}"
                    row["AbsErrorPct"] = f"{abs(pct):.2f}"
                else:
                    row["ErrorPct"] = ""
                    row["AbsErrorPct"] = ""
            except Exception as e:
                row["Error"] = ""
                row["AbsError"] = ""
                row["ErrorPct"] = ""
                row["AbsErrorPct"] = ""
    spacing_img_b64 = latest_data.get("spacing_fig", "")
    content = html.Div([
        html.H2("Planned vs. Actual by Each Layer"),
        html.Div([
            html.Img(src=spacing_img_b64, style={"width": "1200px", "border": "1px solid black"})
        ]),
        html.H4("Statistical Analysis (All Layers)"),
        html.Div("Global statistics will be shown here.")
    ])
    return content, combined_rows

###############################################################################
# CALLBACK: BoxPlot & Summary (Optional)
###############################################################################
@app.callback(
    [Output("error-boxplot", "figure"),
     Output("error-summary", "children")],
    [Input("planned-spacing-table", "data")]
)
def update_boxplot(table_data):
    if not table_data or len(table_data) < 2:
        return go.Figure(), ""
    errors = []
    for row in table_data:
        if row.get("RebarPair", "").startswith("Layer") or not row.get("Error"):
            continue
        try:
            errors.append(float(row["Error"]))
        except:
            pass
    if not errors:
        return go.Figure(), ""
    fig = go.Figure()
    fig.add_trace(go.Box(y=errors, name="Error (mm)"))
    mean_err = np.mean(errors)
    mae = np.mean(np.abs(errors))
    summary = f"Mean Error: {mean_err:.2f} mm, Mean Absolute Error: {mae:.2f} mm"
    return fig, summary

###############################################################################
# RUN THE APP
###############################################################################
if __name__ == "__main__":
    app.run(debug=True)
