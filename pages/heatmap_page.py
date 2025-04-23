'''import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

from utils.dataset_loader import load_dataset
from utils.algorithm_runner import run_algorithm

st.set_page_config(page_title="Outlier Heatmap Viewer", layout="wide")

if "confirmed" not in st.session_state or not st.session_state.confirmed:
    st.error("Please complete the selection on the home page first.")
    st.stop()

dataset_name = st.session_state.dataset
selected_algos = st.session_state.algorithms
heatmap_type = st.session_state.heatmap_type
colormap = st.session_state.colormap

# Load and reduce data
df = load_dataset(dataset_name)
X = df.values
X_2d = PCA(n_components=2).fit_transform(X)

def interpolate_heatmap(X_2d, scores, style, colormap, algo_name, figsize=(2.5, 2.5)):
    threshold = np.percentile(scores, 95)
    scores = np.where(scores >= threshold, scores, 0)

    interpolator = RBFInterpolator(X_2d, scores, neighbors=20, smoothing=0.1)
    grid_x = np.linspace(-1, 1, 100)
    grid_y = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(grid_x, grid_y)
    zz = interpolator(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    zz = np.nan_to_num(zz, nan=np.nanmin(zz))
    zz = np.clip(zz, np.min(zz), np.percentile(zz, 99))

    if style == "interpolated":
        zz = gaussian_filter(zz, sigma=1.5)
    elif style == "binary":
        zz = (zz >= 0.5).astype(int)
    elif style == "ranked":
        flat = zz.flatten()
        ranks = flat.argsort().argsort() / len(flat)
        zz = ranks.reshape(zz.shape)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(colormap)
    contour = ax.contourf(xx, yy, zz, levels=100, cmap=cmap, alpha=0.95, antialiased=True)

    ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        s=5, c='black', alpha=1.0, linewidths=0
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title(f"{algo_name} ({len(X_2d)} pts)", fontsize=9, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor('none')

    return fig, contour

categories = {
    "Proximity-based": selected_algos.get("proximity", []),
    "Probabilistic": selected_algos.get("probabilistic", []),
    "Ensembles": selected_algos.get("ensemble", []),
    "Linear Models": selected_algos.get("linear", [])
}

st.markdown("---")
plots = []
contour_ref = None

# Plot headers and heatmaps
cols = st.columns(4)
for idx, (category, algos) in enumerate(categories.items()):
    with cols[idx]:
        st.markdown(f"<h5 style='text-align: center; color: white;'>{category}</h5>", unsafe_allow_html=True)

max_rows = max(len(v) for v in categories.values())
for i in range(max_rows):
    row = st.columns(4)
    for j, (cat_name, algo_list) in enumerate(categories.items()):
        if i < len(algo_list):
            algo = algo_list[i]
            with row[j]:
                try:
                    scores = run_algorithm(algo, df.values)
                    fig, contour_ref = interpolate_heatmap(X_2d, scores, heatmap_type, colormap, algo)
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"{algo} failed: {e}")
        else:
            row[j].empty()

# Add Colorbar BELOW grid
if contour_ref is not None:
    colbar = st.columns([1, 1, 1, 1])
    with colbar[1]:
        bar_height = 3.0
        fig, ax = plt.subplots(figsize=(0.6, bar_height))
        norm = plt.Normalize(vmin=0, vmax=1)
        gradient = np.linspace(1, 0, 256).reshape(-1, 1)
        ax.imshow(gradient, aspect='auto', cmap=colormap)
        ax.set_xticks([])
        ax.set_yticks([0, 255])
        ax.set_yticklabels(["Likely Outlier", "Likely Inlier"], fontsize=9)
        ax.yaxis.tick_right()
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.subplots_adjust(left=0.3, right=0.7, top=0.998, bottom=0.002)
        fig.patch.set_facecolor('none')
        st.pyplot(fig, use_container_width=True) '''


















































import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

from utils.dataset_loader import load_dataset
from utils.algorithm_runner import run_algorithm

st.set_page_config(page_title="Outlier Heatmap Viewer", layout="wide")

if "confirmed" not in st.session_state or not st.session_state.confirmed:
    st.error("Please complete the selection on the home page first.")
    st.stop()

dataset_name = st.session_state.dataset
selected_algos = st.session_state.algorithms
heatmap_type = st.session_state.heatmap_type or "interpolated"
colormap = st.session_state.colormap

df = load_dataset(dataset_name)
X = df.values
X_2d = PCA(n_components=2).fit_transform(X)

@st.cache_data(show_spinner="Interpolating heatmap...", max_entries=64)
def cached_interpolation(X_2d, scores):
    interpolator = RBFInterpolator(X_2d, scores, neighbors=20, smoothing=0.1)
    grid_x = np.linspace(-1, 1, 100)
    grid_y = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(grid_x, grid_y)
    zz = interpolator(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, zz


def interpolate_heatmap(X_2d, scores, style, colormap, algo_name, figsize=(2.5, 2.0)):
    xx, yy, zz = cached_interpolation(X_2d, scores)

    zz = np.nan_to_num(zz, nan=np.nanmin(zz))
    zz = np.clip(zz, np.min(zz), np.percentile(zz, 99))

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(colormap)

    if style in ["interpolated", "raw"]:
        zz = gaussian_filter(zz, sigma=3.0)
        contour = ax.contourf(xx, yy, zz, levels=50, cmap=cmap, alpha=0.95, antialiased=True)
    elif style == "binary":
        zz = (zz >= 0.5).astype(int)
        contour = ax.imshow(zz, extent=[-1, 1, -1, 1], cmap=cmap, origin='lower', aspect='auto')
    elif style == "threshold":
        threshold_val = np.percentile(zz, 95)
        zz = (zz >= threshold_val).astype(int)
        contour = ax.imshow(zz, extent=[-1, 1, -1, 1], cmap=cmap, origin='lower', aspect='auto')
    elif style == "ranked":
        flat = zz.flatten()
        ranks = flat.argsort().argsort() / len(flat)
        zz = ranks.reshape(zz.shape)
        contour = ax.contourf(xx, yy, zz, levels=50, cmap=cmap, alpha=0.95, antialiased=True)
    else:
        raise ValueError(f"Unknown heatmap type: {style}")

    ax.scatter(X_2d[:, 0], X_2d[:, 1], s=4, c='black', alpha=0.7, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title(algo_name, fontsize=10, pad=8, color='white')

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.01)
    fig.patch.set_facecolor('none')
    return fig, contour

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


st.markdown("---")
header_cols = st.columns([1, 1, 1, 1, 0.3])
unique_categories = list(categories.keys())
for idx, category in enumerate(unique_categories):
    header_cols[idx].markdown(f"<h5 style='text-align: center;'>{category}</h5>", unsafe_allow_html=True)


heatmap_height = 2.0
colorbar_height = heatmap_height * 2 + 0.8


container = st.container()
cols = container.columns([1, 1, 1, 1, 0.3])
contour_ref = None
rendered_colorbar = False


for idx, algo in enumerate(flat_algos):
    col_idx = idx % 4
    with cols[col_idx]:
        try:
            scores = run_algorithm(algo, df.values)
            fig, contour_ref = interpolate_heatmap(X_2d, scores, heatmap_type, colormap, algo, figsize=(2.5, heatmap_height))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"{algo} failed: {e}")


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

