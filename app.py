
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.dataset_loader import load_dataset


st.set_page_config(page_title="Outlier Dashboard", layout="wide")


st.markdown("""
<style>
    [data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="stToolbar"] { display: none !important; }
    .css-18e3th9, .block-container { padding: 1rem !important; }
    div[data-testid="column"] > div > h4, .highlight-header { border-radius: 0 !important; }
    .highlight-header { background-color: #3b82f6; color: white; padding: 8px 10px; font-weight: 600; font-size: 18px; margin-bottom: 4px; }
    .algo-group { background-color: #717D7E; padding: 8px; border-radius: 4px; margin-bottom: 4px; }
    .algo-group strong { color: white; font-size: 16px; }
    .stButton>button {
        height: 2.8rem;
        width: 100%;
        font-weight: 500;
        background-color: #3b82f6;
        color: white;
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

# --- Session State Initialization ---
for key, default in [("dataset", None), ("algorithms", {}), ("heatmap_type", None), ("colormap", "viridis"), ("confirmed", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Header ---
st.markdown("<h2 style='text-align:center; margin-bottom:1rem;'>Toward an Outlier Uncertainty Model – A Comparative Analysis</h2>", unsafe_allow_html=True)

# --- Algorithm Metadata ---
df_algorithms_full = {
    "CBLOF": "Cluster-based Local Outlier Factor",
    "HBOS": "Histogram-based Outlier Score",
    "KNN": "K-Nearest Neighbors",
    "LOF": "Local Outlier Factor",
    "DBSCA": "Density-based Spatial Clustering of Applications with Noise",
    "ABOD": "Angle-based Outlier Detector",
    "GMM": "Gaussian Mixture Model",
    "KDE": "Kernel Density Estimation",
    "ECOD": "Empirical Cumulative Outlier Detection",
    "COPOD": "Copula-based Outlier Detection",
    "FB": "Feature Bagging Ensemble",
    "IForest": "Isolation Forest",
    "LSCP": "Locally Selective Combination in Parallel",
    "INNE": "Isolation Nearest Neighbor Ensemble",
    "MCD": "Minimum Covariance Determinant",
    "OCSVM": "One-Class Support Vector Machine",
    "PCA": "Principal Component Analysis",
    "LMDD": "Linear Method for Deviation Detection"
}

categories = {
    "Proximity-based": ["CBLOF", "HBOS", "KNN", "LOF", "DBSCA"],
    "Probabilistic": ["ABOD", "GMM", "KDE", "ECOD", "COPOD"],
    "Ensembles": ["FB", "IForest", "LSCP", "INNE"],
    "Linear Models": ["MCD", "OCSVM", "PCA", "LMDD"]
}
key_map = {k: v for k, v in zip(categories.keys(), ["proximity", "probabilistic", "ensemble", "linear"])}

# --- Main Layout ---
col1, col2, col3, col4 = st.columns([2, 4, 1.5, 1.5])

# === Dataset Section ===
with col1:
    st.markdown("<div class='highlight-header'>Dataset Selection</div>", unsafe_allow_html=True)
    dataset = st.selectbox(
        "Choose Dataset",
        [
            "AirQualityUCI",
            "Bank",
            "BeijingClimate",
            "Breast-cancer-wisconsin",
            "Diabetes",
            "Iris",
            "Nhanes",
            "ObesityDataSet",
            "PIRSensor",
            "Students"
        ],
        label_visibility="collapsed"
    )

    if dataset:
        try:
            df = load_dataset(dataset)

            # Convert to numeric only
            df_numeric = df.select_dtypes(include=[np.number])

            # Drop rows with NaNs
            df_numeric = df_numeric.dropna()

            if df_numeric.shape[0] >= 2 and df_numeric.shape[1] >= 2:
                X2 = PCA(n_components=2).fit_transform(df_numeric.values)

                # Normalize and add tiny jitter
                X2 = X2 / np.max(np.abs(X2))
                X2 += np.random.normal(0, 0.015, size=X2.shape)

                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(X2[:, 0], X2[:, 1], s=6, color='#3b82f6', alpha=0.7, edgecolors='none')
                x_padding = (X2[:, 0].max() - X2[:, 0].min()) * 0.1
                y_padding = (X2[:, 1].max() - X2[:, 1].min()) * 0.1
                ax.set_xlim(X2[:, 0].min() - x_padding, X2[:, 0].max() + x_padding)
                ax.set_ylim(X2[:, 1].min() - y_padding, X2[:, 1].max() + y_padding)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"PCA Scatter of {dataset}")
                st.pyplot(fig, use_container_width=True)

            else:
                # Fallback random scatter if PCA not possible
                st.warning(f"No usable numeric features for PCA in {dataset}. Showing random scatter instead.")
                n_samples = len(df)
                x = np.random.uniform(low=-1, high=1, size=n_samples)
                y = np.random.uniform(low=-1, high=1, size=n_samples)

                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(x, y, s=6, color='blue', alpha=0.7, edgecolors='none')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Random Scatter of {dataset}")
                st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"PCA failed: {e}")

# === Algorithm Selection Section ===
selected_algos = {}
with col2:
    st.markdown("<div class='highlight-header'>Algorithm Selection</div>", unsafe_allow_html=True)
    inner_cols = st.columns(len(categories))
    for idx, (group, algos) in enumerate(categories.items()):
        with inner_cols[idx]:
            st.markdown(f"<div class='algo-group'><strong>{group}</strong></div>", unsafe_allow_html=True)
            keys = [f"{group}-{a}" for a in algos]
            count = sum(st.session_state.get(k, False) for k in keys)
            picks = []
            for algo in algos:
                k = f"{group}-{algo}"
                checked = st.checkbox(
                    label=algo,
                    key=k,
                    help=df_algorithms_full[algo],
                    disabled=(count >= 2 and not st.session_state.get(k)),
                    label_visibility="visible"
                )
                if checked:
                    picks.append(algo)
            selected_algos[key_map[group]] = picks

# === Heatmap Technique Section ===
with col3:
    st.markdown("<div class='highlight-header'>Heatmap Techniques</div>", unsafe_allow_html=True)
    heatmap_type = st.radio(
        "",
        ["raw", "threshold", "interpolated", "binary", "ranked"],
        label_visibility="collapsed"
    )

# === Colormap Section ===
with col4:
    st.markdown("<div class='highlight-header'>Colormap</div>", unsafe_allow_html=True)
    cmaps = ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "turbo"]
    for cmap in cmaps:
        def on_cb(c=cmap):
            st.session_state['colormap'] = c
        chk = (st.session_state['colormap'] == cmap)
        cb_col, bar_col, _ = st.columns([0.5, 3, 0.5])
        cb_col.checkbox("", value=chk, key=f"cb_{cmap}", on_change=on_cb, label_visibility="collapsed")
        fig, ax = plt.subplots(figsize=(3, 0.3), dpi=100)
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.axis('off')
        fig.tight_layout(pad=0)
        bar_col.pyplot(fig, use_container_width=True)
        plt.close(fig)

# === Continue Button ===
_, c1, _ = st.columns([1, 1, 1])
with c1:
    if st.button("Explore", key="continue_btn"):
        if all(len(v) == 2 for v in selected_algos.values()):
            st.session_state['dataset'] = dataset
            st.session_state['algorithms'] = selected_algos
            st.session_state['heatmap_type'] = heatmap_type
            st.session_state['confirmed'] = True
            st.switch_page("pages/heatmap_page.py")
        else:
            st.error("Select exactly two algorithms per category before continuing.")
