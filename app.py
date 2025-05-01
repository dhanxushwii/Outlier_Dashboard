
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter

from utils.dataset_loader import load_dataset
from utils.algorithm_runner import run_algorithm

st.set_page_config(page_title="Outlier Dashboard", layout="wide")

# --- Styling ---
st.markdown("""
<h1 style='text-align: center; color: white; font-size: 32px; margin-bottom: 1rem;'>
    OutlierVisualizer: A Comparative Dashboard for Outlier Detection Algorithms
</h1>
<style>
    [data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="stToolbar"] { display: none !important; }
    .block-container { padding: 1rem !important; }
    .highlight-header { background-color: #3b82f6; color: white; padding: 8px 10px; font-weight: 600; font-size: 18px; margin-bottom: 4px; }
    .algo-group {
    background-color: #717D7E;
    padding: 4px 8px;
    border-radius: 4px;
    margin-bottom: 6px;
    line-height: 1.2;
}
    .algo-group strong { color: white; font-size: 16px; }
    .stButton>button {
        height: 2.8rem; width: 100%;
        font-weight: 500; background-color: #3b82f6; color: white;
        border-radius: 0 !important;
    }
    .stButton>button[key="continue_btn"] {
        border-radius: 12px !important;
        margin-top: 20px !important;
        width: 40% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Caching ---
@st.cache_data(show_spinner=False)
def load_cached_dataset(name):
    df = load_dataset(name)
    df_numeric = df.select_dtypes(include=[np.number]).dropna()
    return SimpleImputer().fit_transform(df_numeric)

@st.cache_data(show_spinner=False)
def get_pca_coords(X):
    X2 = PCA(n_components=2).fit_transform(X)
    X2 = X2 / np.max(np.abs(X2)) + np.random.normal(0, 0.015, size=X2.shape)
    xpad, ypad = (X2[:, 0].ptp() * 0.1, X2[:, 1].ptp() * 0.1)
    return X2, (X2[:, 0].min() - xpad, X2[:, 0].max() + xpad), (X2[:, 1].min() - ypad, X2[:, 1].max() + ypad)

@st.cache_data(show_spinner=False)
def get_scores_cached(algo, X):
    return run_algorithm(algo, X)

# --- Session state defaults ---
for key, default in [
    ("dataset", None), ("algorithms", {}), ("heatmap_type", None),
    ("colormap", None), ("confirmed", False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Category setup ---
categories = {
    "Proximity-based": ["CBLOF", "HBOS", "KNN", "LOF", "DBSCAN"],
    "Probabilistic": ["ABOD", "GMM", "KDE", "ECOD", "COPOD"],
    "Ensembles": ["FB", "IForest", "LSCP", "INNE"],
    "Linear Models": ["MCD", "OCSVM", "PCA", "LMDD"]
}
key_map = {k: k.lower().replace(" ", "") for k in categories}

# --- Layout columns ---
col1, col2, col3, col4 = st.columns([2, 4, 1.5, 1.5])

# --- Dataset Selection ---
with col1:
    st.markdown("<div class='highlight-header'>Dataset Selection</div>", unsafe_allow_html=True)
    dataset = st.selectbox(
    label="",
    options=[
        "Bank", "BeijingClimate", "Breast-cancer-wisconsin", "CardioIsomap", "CoilDensmap",
        "Iris", "Nhanes", "ObesityDataSet", "PIRSensor"
    ],
    key="dataset_select",
    label_visibility="collapsed"
)


    if dataset and st.session_state.get("dataset") != dataset:
        st.session_state["confirmed"] = False
        st.session_state["algorithms"] = {}
        st.session_state["heatmap_type"] = None
        st.session_state["colormap"] = None
        for group, algos in categories.items():
            for algo in algos:
                st.session_state[f"{group}-{algo}"] = False

        X = load_cached_dataset(dataset)
        X2d, xlim, ylim = get_pca_coords(X)
        st.session_state["pca"] = {"X2d": X2d, "xlim": xlim, "ylim": ylim}
        st.session_state["dataset"] = dataset

    if "pca" in st.session_state:
        X2d = st.session_state["pca"]["X2d"]
        xlim = st.session_state["pca"]["xlim"]
        ylim = st.session_state["pca"]["ylim"]
        fig, ax = plt.subplots()
        ax.scatter(X2d[:,0], X2d[:,1], s=6, color='#3b82f6', alpha=0.7, edgecolors='none')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"PCA Scatter of {dataset}")
        st.pyplot(fig, use_container_width=True)

# --- Algorithm Selection ---
with col2:
    st.markdown("<div class='highlight-header'>Algorithm Selection</div>", unsafe_allow_html=True)
    inner = st.columns(len(categories))
    for i, (group, algos) in enumerate(categories.items()):
        with inner[i]:
            st.markdown(f"<div class='algo-group'><strong>{group}</strong></div>", unsafe_allow_html=True)
            selected_count = 0
            for algo in algos:
                selected = st.checkbox(
                    label=algo,
                    key=f"{group}-{algo}",
                    disabled=st.session_state["confirmed"]  # ✅ only disable checkboxes after "Explore"
                )
                if selected:
                    selected_count += 1
            if not st.session_state["confirmed"] and selected_count != 2:
                st.markdown("<span style='color:#facc15; font-size:13px;'>⚠️ Pick 2 algorithms</span>", unsafe_allow_html=True)

# --- Heatmap + Colormap (always active!) ---
with col3:
    st.markdown("<div class='highlight-header'>Heatmap Techniques</div>", unsafe_allow_html=True)
    st.radio("", ["raw", "threshold", "interpolated", "binary", "ranked"],
             key="heatmap_type", label_visibility="collapsed", index=None)

with col4:
    st.markdown("<div class='highlight-header'>Colormap</div>", unsafe_allow_html=True)
    st.radio("", ["viridis", "plasma", "terrain", "coolwarm", "turbo"],
         index=None, key="colormap", label_visibility="collapsed")


# --- Explore Button ---
explore_disabled = not (
    st.session_state.get("dataset_select") and
    st.session_state.get("heatmap_type") and
    st.session_state.get("colormap") and
    all(
        sum([
            st.session_state.get(f"{group}-{algo}", False)
            for algo in algos
        ]) == 2 for group, algos in categories.items()
    )
)

_, mid, _ = st.columns([1, 1, 1])
with mid:
    st.button("Explore", key="continue_btn", disabled=explore_disabled,
        on_click=lambda: st.session_state.update({
            "confirmed": True,
            "algorithms": {
                key_map[group]: [
                    algo for algo in algos if st.session_state.get(f"{group}-{algo}", False)
                ] for group, algos in categories.items()
            }
        }))

# --- Visualization ---
if st.session_state.get("confirmed") and st.session_state.get("heatmap_type"):
    X = StandardScaler().fit_transform(load_cached_dataset(st.session_state["dataset"]))
    X2d = st.session_state["pca"]["X2d"]
    xlim = st.session_state["pca"]["xlim"]
    ylim = st.session_state["pca"]["ylim"]

    def render_heatmap(X2d, scores, method, cmap, title):
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        if method == "threshold":
            scores = (scores >= np.percentile(scores, 95)).astype(float)
        elif method == "binary":
            scores = (scores >= 0.5).astype(float)
        elif method == "ranked":
            scores = scores.argsort().argsort() / len(scores)

        gx, gy = np.linspace(xlim[0], xlim[1], 60), np.linspace(ylim[0], ylim[1], 60)
        xx, yy = np.meshgrid(gx, gy)
        interpolator = RBFInterpolator(X2d, scores, neighbors=20, smoothing=0.1)
        zz = interpolator(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        if method in ["threshold", "interpolated", "ranked"]:
            zz = gaussian_filter(zz, sigma=1.5)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, zz, levels=60, cmap=cmap)
        ax.scatter(X2d[:,0], X2d[:,1], s=4, c="black", alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(title, fontsize=8, color="white")
        fig.patch.set_facecolor("none")
        return fig

    cols = st.columns([1,1,1,1,0.3])
    cat_order = list(categories.keys())
    idx_map = {cat: i for i, cat in enumerate(cat_order)}

    for cat in cat_order:
        for algo in st.session_state["algorithms"].get(key_map[cat], []):
            with cols[idx_map[cat]]:
                try:
                    scores = get_scores_cached(algo, X)
                    fig = render_heatmap(X2d, scores, st.session_state["heatmap_type"],
                                         st.session_state["colormap"], algo)
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"{algo} failed: {e}")

    with cols[-1]:
        fig, ax = plt.subplots(figsize=(0.9, 3.29), dpi=100)
        gradient = np.linspace(0,1,256).reshape(256,1)
        ax.imshow(gradient[::-1], aspect="auto", cmap=st.session_state["colormap"], extent=[0,1,0,1])
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 1.02, "Outlier", transform=ax.transAxes, ha="center", va="bottom", fontsize=6, color="white")
        ax.text(0.5, -0.02, "Inlier", transform=ax.transAxes, ha="center", va="top", fontsize=6, color="white")
        for s in ax.spines.values():
            s.set_visible(False)
        fig.subplots_adjust(left=0.4, right=0.6, top=1.0, bottom=0.0)
        fig.patch.set_facecolor("none")
        st.pyplot(fig, use_container_width=False)

