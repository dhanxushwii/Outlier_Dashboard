import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from utils.dataset_loader import load_dataset
from utils.algorithm_runner import run_algorithm

# Check if selection is done
if "confirmed" not in st.session_state or not st.session_state.confirmed:
    st.error("Please complete the selection on the home page first.")
    st.stop()

# Load session selections
dataset_name = st.session_state.dataset
selected_algos = st.session_state.algorithms
heatmap_type = st.session_state.heatmap_type or "interpolated"
colormap = st.session_state.colormap

# Load and prepare dataset
df = load_dataset(dataset_name)
df_numeric = df.select_dtypes(include=[np.number])

# Drop constant columns
df_numeric = df_numeric.loc[:, df_numeric.std() > 0]

if df_numeric.shape[1] >= 1:
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')

    # Impute missing values
    X_imputed = imputer.fit_transform(df_numeric)

    # Standardize the data
    X = scaler.fit_transform(X_imputed)

    # Add tiny random noise to avoid flat collapse
    X += np.random.normal(0, 1e-5, X.shape)

    # PCA for plotting only
    X_2d = PCA(n_components=2).fit_transform(X)
else:
    st.error("No numeric features found for selected dataset. Heatmaps cannot be generated.")
    st.stop()

# Save PCA scatter limits
x_padding = (X_2d[:, 0].max() - X_2d[:, 0].min()) * 0.1
y_padding = (X_2d[:, 1].max() - X_2d[:, 1].min()) * 0.1

xlim = (X_2d[:, 0].min() - x_padding, X_2d[:, 0].max() + x_padding)
ylim = (X_2d[:, 1].min() - y_padding, X_2d[:, 1].max() + y_padding)

@st.cache_data(show_spinner="Interpolating heatmap...", max_entries=64)
def cached_interpolation(X_2d, scores, xlim, ylim):
    grid_x = np.linspace(xlim[0], xlim[1], 120)
    grid_y = np.linspace(ylim[0], ylim[1], 120)
    xx, yy = np.meshgrid(grid_x, grid_y)
    interpolator = RBFInterpolator(X_2d, scores, neighbors=20, smoothing=0.1)
    zz = interpolator(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, zz

# Rest of the original code remains unchanged...

def interpolate_heatmap(X_2d, scores, style, colormap, algo_name, xlim, ylim, figsize=(2.5, 2.0)):
    xx, yy, zz = cached_interpolation(X_2d, scores, xlim, ylim)

    zz = np.nan_to_num(zz, nan=np.nanmin(zz))
    zz = np.clip(zz, np.min(zz), np.percentile(zz, 99))

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(colormap)

    if style in ["interpolated", "raw"]:
        zz = gaussian_filter(zz, sigma=2.0)
        contour = ax.contourf(xx, yy, zz, levels=60, cmap=cmap, alpha=0.95, antialiased=True)
    elif style == "binary":
        zz = (zz >= 0.5).astype(int)
        contour = ax.imshow(zz, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap=cmap, origin='lower', aspect='auto')
    elif style == "threshold":
        threshold_val = np.percentile(zz, 95)
        zz = (zz >= threshold_val).astype(int)
        contour = ax.imshow(zz, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap=cmap, origin='lower', aspect='auto')
    elif style == "ranked":
        flat = zz.flatten()
        ranks = flat.argsort().argsort() / len(flat)
        zz = ranks.reshape(zz.shape)
        contour = ax.contourf(xx, yy, zz, levels=50, cmap=cmap, alpha=0.95, antialiased=True)
    else:
        raise ValueError(f"Unknown heatmap style: {style}")

    ax.scatter(X_2d[:, 0], X_2d[:, 1], s=4, c='black', alpha=0.7, linewidths=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(algo_name, fontsize=10, pad=8, color='white')

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.01)
    fig.patch.set_facecolor('none')
    return fig, contour

# Organize selected algorithms
categories = {
    "Proximity-based": selected_algos.get("proximity", []),
    "Probabilistic": selected_algos.get("probabilistic", []),
    "Ensembles": selected_algos.get("ensemble", []),
    "Linear Models": selected_algos.get("linear", [])
}

flat_algos = []
category_names = []
for cat_name, algo_list in categories.items():
    flat_algos.extend(algo_list)
    category_names.extend([cat_name] * len(algo_list))

# UI Layout
st.markdown("---")
header_cols = st.columns([1, 1, 1, 1, 0.3])
unique_categories = list(categories.keys())
for idx, category in enumerate(unique_categories):
    header_cols[idx].markdown(f"<h5 style='text-align: center;'>{category}</h5>", unsafe_allow_html=True)

heatmap_height = 2.0
container = st.container()
cols = container.columns([1, 1, 1, 1, 0.3])
contour_ref = None
rendered_colorbar = False

# Generate heatmaps
for idx, algo in enumerate(flat_algos):
    col_idx = idx % 4
    with cols[col_idx]:
        try:
            scores = run_algorithm(algo, X)  # ✅ use full standardized features, not PCA
            fig, contour_ref = interpolate_heatmap(X_2d, scores, heatmap_type, colormap, algo, xlim, ylim, figsize=(2.5, heatmap_height))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"{algo} failed: {e}")

# Draw colormap bar
if not rendered_colorbar and contour_ref is not None:
    with cols[-1]:
        dpi = 100
        fig_width_inches = 0.89
        fig_height_inches = 3.430
        fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches), dpi=dpi)

        gradient = np.linspace(0, 1, 256).reshape(256, 1)
        ax.imshow(gradient[::-1], aspect='auto', cmap=colormap, extent=[0, 1, 0, 1])

        ax.set_xticks([])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Likely Outlier", "Likely Inlier"], fontsize=5.8, color='white')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.subplots_adjust(left=0.4, right=0.6, top=1.0, bottom=0.0)
        fig.patch.set_facecolor('none')

        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        rendered_colorbar = True


















