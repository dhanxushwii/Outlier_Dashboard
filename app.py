import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.dataset_loader import load_dataset

st.set_page_config(page_title="Outlier Dashboard", layout="wide")

st.markdown("""
<style>
    /* Hide default UI */
    [data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="stToolbar"] { display: none !important; }
    .css-18e3th9, .block-container { padding: 1rem !important; }
    /* Headers */
    div[data-testid="column"] > div > h4, .highlight-header { border-radius: 0 !important; }
    .highlight-header { background-color: #3b82f6; color: white; padding: 8px 10px; font-weight: 600; font-size: 18px; margin-bottom: 4px; }
    /* Algorithm group container */
    .algo-group { background-color: #717D7E; padding: 8px; border-radius: 4px; margin-bottom: 4px; }
    .algo-group strong { color: white; font-size: 16px; }
    /* Explore button */
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

for key, default in [("dataset", None), ("algorithms", {}), ("heatmap_type", None), ("colormap", "viridis"), ("confirmed", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("<h2 style='text-align:center; margin-bottom:1rem;'>Toward an Outlier Uncertainty Model – A Comparative Analysis</h2>", unsafe_allow_html=True)

df_algorithms_full = {
    "CBLOF": "Cluster-based Local Outlier Factor",
    "HBOS": "Histogram-based Outlier Score",
    "KNN": "K-Nearest Neighbors",
    "LOF": "Local Outlier Factor",
    "DBSCAN": "Density-based Spatial Clustering of Applications with Noise",
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
    "Proximity-based": ["CBLOF", "HBOS", "KNN", "LOF", "DBSCAN"],
    "Probabilistic": ["ABOD", "GMM", "KDE", "ECOD", "COPOD"],
    "Ensembles": ["FB", "IForest", "LSCP", "INNE"],
    "Linear Models": ["MCD", "OCSVM", "PCA", "LMDD"]
}
key_map = {k: v for k, v in zip(categories.keys(), ["proximity", "probabilistic", "ensemble", "linear"]) }

col1, col2, col3, col4 = st.columns([2, 4, 1.5, 1.5])

with col1:
    st.markdown("<div class='highlight-header'>Dataset Selection</div>", unsafe_allow_html=True)
    dataset = st.selectbox(
        "Choose Dataset",
        ["AirQualityUCI", "bank", "BeijingClimate", "nhanes", "students", "ObesityDataSet"],
        label_visibility="collapsed"
    )
    if dataset:
        df = load_dataset(dataset)
        X2 = PCA(n_components=2).fit_transform(df.values)
        fig, ax = plt.subplots(figsize=(4,3))
        ax.scatter(X2[:,0], X2[:,1], s=10, alpha=0.7)
        ax.set(xticks=[], yticks=[], aspect='equal', title=f"PCA Scatter of {dataset}")
        st.pyplot(fig, use_container_width=True)

selected_algos = {}
with col2:
    st.markdown("<div class='highlight-header'>Algorithm Selection</div>", unsafe_allow_html=True)
    for group, algos in categories.items():
        st.markdown(f"<div class='algo-group'><strong>{group}</strong>", unsafe_allow_html=True)
        keys = [f"{group}-{a}" for a in algos]
        count = sum(st.session_state.get(k, False) for k in keys)
        cols = st.columns(len(algos))
        picks = []
        for i, algo in enumerate(algos):
            k = f"{group}-{algo}"
            val = cols[i].checkbox(
                algo,
                key=k,
                help=df_algorithms_full[algo],
                disabled=(count >= 2 and not st.session_state.get(k))
            )
            if val:
                picks.append(algo)
        selected_algos[key_map[group]] = picks
        st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='highlight-header'>Heatmap Techniques</div>", unsafe_allow_html=True)
    heatmap_type = st.radio(
        "",
        ["raw", "threshold", "interpolated", "binary", "ranked"],
        label_visibility="collapsed"
    )

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

