import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.dataset_loader import load_dataset

st.set_page_config(page_title="Outlier Dashboard", layout="wide")

hide_sidebar = """
    <style>
        [data-testid="stSidebarNav"] { display: none !important; }
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }
        .css-18e3th9 { padding: 1rem !important; }
        .block-container { padding: 1rem !important; }
        label[data-testid="stCheckboxLabel"] > div {
            white-space: nowrap !important;
            overflow: visible !important;
            color: white !important;
            font-size: 13px !important;
            padding: 2px 0 !important;
        }
        div[data-testid="column"] > div > h4 {
            background-color: #1e293b;
            padding: 6px 10px;
            border-radius: 0 !important;
            color: white;
            text-align: center;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .highlight-header {
            background-color: #3b82f6;
            padding: 6px 10px;
            color: white;
            border-radius: 0 !important;
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .stButton>button {
            height: 2.8rem;
            width: 100%;
            font-weight: 500;
            background-color: #3b82f6;
            color: white;
            border-radius: 0 !important;
        }
        /* Round corners and lower position for the Explore button */
        .stButton>button:last-of-type {
            border-radius: 12px !important;
            margin-top: 20px !important;
        }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)


for key, default in [
    ("dataset", None),
    ("algorithms", {}),
    ("heatmap_type", None),
    ("colormap", "viridis"),
    ("confirmed", False)
]:
    if key not in st.session_state:
        st.session_state[key] = default


st.markdown(
    "<h2 style='text-align: center; margin-bottom: 1.5rem;'>Toward an Outlier Uncertainty Model – A Comparative Analysis</h2>",
    unsafe_allow_html=True
)


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
key_map = {
    "Proximity-based": "proximity",
    "Probabilistic": "probabilistic",
    "Ensembles": "ensemble",
    "Linear Models": "linear"
}


col1, col2, col3, col4 = st.columns([2, 4, 1.5, 1.5])


DATASET_OPTIONS = [
    "AirQualityUCI",
    "bank",
    "BeijingClimate",
    "nhanes",
    "students",
    "ObesityDataSet"
]


with col1:
    st.markdown("<div class='highlight-header'>Dataset Selection</div>", unsafe_allow_html=True)
    dataset = st.selectbox("Choose Dataset", DATASET_OPTIONS, label_visibility="collapsed")
    if dataset:
        df = load_dataset(dataset)
        X_2d = PCA(n_components=2).fit_transform(df.values)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(X_2d[:, 0], X_2d[:, 1], s=10, alpha=0.7)
        ax.set_title(f"PCA Scatter of {dataset}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)


selected_algos = {}
with col2:
    st.markdown("<div class='highlight-header'>Algorithm Selection</div>", unsafe_allow_html=True)
    for idx, (group, algos) in enumerate(categories.items()):
        if idx > 0:
            st.markdown("---")
        st.markdown(f"**{group}**")
        group_keys = [f"{group}-{algo}" for algo in algos]
        selected_count = sum(st.session_state.get(key, False) for key in group_keys)
        algo_cols = st.columns(5)
        picks = []
        for j, algo in enumerate(algos):
            key = f"{group}-{algo}"
            disabled = selected_count >= 2 and not st.session_state.get(key, False)
            new_val = algo_cols[j].checkbox(algo, key=key, help=df_algorithms_full[algo], disabled=disabled)
            if new_val:
                picks.append(algo)
        selected_algos[key_map[group]] = picks


with col3:
    st.markdown("<div class='highlight-header'>Heatmap Techniques</div>", unsafe_allow_html=True)
    heatmap_type = st.radio(
        "Style",
        ["raw", "threshold", "interpolated", "binary", "ranked"],
        label_visibility="collapsed"
    )


with col4:
    st.markdown("<div class='highlight-header'>Colormap</div>", unsafe_allow_html=True)
    cmaps = ["viridis", "plasma", "inferno"]
    for cmap in cmaps:
        def on_cb(c=cmap):
            st.session_state['colormap'] = c
        checked = (st.session_state['colormap'] == cmap)
        cb_col, bar_col, _ = st.columns([0.5, 3, 0.5])
        cb_col.checkbox(
            "", 
            value=checked,
            key=f"cb_{cmap}",
            on_change=on_cb,
            label_visibility="collapsed"
        )
        
        fig, ax = plt.subplots(figsize=(3, 0.3), dpi=100)
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.axis('off')
        fig.tight_layout(pad=0)
        bar_col.pyplot(fig, use_container_width=True)
        plt.close(fig)

div1, div2, div3 = st.columns([1, 1, 1])
with div2:
    if st.button("Explore", key="continue_btn"):
        if all(len(v) == 2 for v in selected_algos.values()):
            st.session_state.dataset = dataset
            st.session_state.algorithms = selected_algos
            st.session_state.heatmap_type = heatmap_type
            st.session_state.confirmed = True
            st.switch_page("pages/heatmap_page.py")
        else:
            st.error("Select exactly two algorithms per category before continuing.")
