import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from utils.dataset_loader import load_dataset
from utils.algorithm_runner import run_algorithm

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
    .stButton>button:disabled {
        pointer-events: auto !important;
        background-color: #93c5fd !important;
        color: white !important;
        cursor: not-allowed !important;
        opacity: 1 !important;
    }
    .stButton>button:disabled:hover {
        background-color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)


for key, default in [
    ("dataset", None),
    ("algorithms", {}),
    ("heatmap_type", None),
    ("colormap", "viridis"),
    ("confirmed", False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

def clear_confirmation():
    st.session_state["confirmed"] = False

st.markdown(
    "<h2 style='text-align:center;'>Toward an Outlier Uncertainty Model – A Comparative Analysis</h2>",
    unsafe_allow_html=True
)

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
key_map = {
    "Proximity-based": "proximity",
    "Probabilistic":    "probabilistic",
    "Ensembles":        "ensemble",
    "Linear Models":    "linear"
}

def reset_algos():
    for group, algos in categories.items():
        for algo in algos:
            st.session_state[f"{group}-{algo}"] = False
    st.session_state["algorithms"] = {}
    st.session_state["confirmed"] = False


    st.session_state["heatmap_type"] = None
    st.session_state["colormap"] = None


    for cmap in ["viridis", "plasma", "terrain", "coolwarm", "turbo"]:
        st.session_state[f"cb_{cmap}"] = False



col1, col2, col3, col4 = st.columns([2, 4, 1.5, 1.5])

with col1:
    st.markdown("<div class='highlight-header'>Dataset Selection</div>", unsafe_allow_html=True)
    dataset = st.selectbox(
        "Choose Dataset",
        [
            "AirQualityUCI", "Bank", "BeijingClimate", "Breast-cancer-wisconsin",
            "Diabetes", "Iris", "Nhanes", "PIRSensor"
        ],
        key="dataset_select",
        on_change=reset_algos,
        label_visibility="collapsed"
    )

    if dataset:
        try:
            df = load_dataset(dataset)
            df_numeric = df.select_dtypes(include=[np.number]).dropna()
            if df_numeric.shape[0] >= 2 and df_numeric.shape[1] >= 2:
                X2 = PCA(n_components=2).fit_transform(df_numeric.values)
                X2 = X2 / np.max(np.abs(X2))
                X2 += np.random.normal(0, 0.015, size=X2.shape)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(X2[:,0], X2[:,1], s=6, color='#3b82f6', alpha=0.7, edgecolors='none')
                padx = (X2[:,0].max()-X2[:,0].min())*0.1
                pady = (X2[:,1].max()-X2[:,1].min())*0.1
                ax.set_xlim(X2[:,0].min()-padx, X2[:,0].max()+padx)
                ax.set_ylim(X2[:,1].min()-pady, X2[:,1].max()+pady)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"PCA Scatter of {dataset}")
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning(f"No usable numeric features for PCA in {dataset}. Showing random scatter instead.")
                n = len(df)
                x = np.random.uniform(-1,1,n)
                y = np.random.uniform(-1,1,n)
                fig, ax = plt.subplots(figsize=(5,4))
                ax.scatter(x,y,s=6,color='blue',alpha=0.7,edgecolors='none')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"Random Scatter of {dataset}")
                st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"PCA failed: {e}")

selected_algos = {}
with col2:
    st.markdown("<div class='highlight-header'>Algorithm Selection</div>", unsafe_allow_html=True)
    inner = st.columns(len(categories))
    for i, (group, algos) in enumerate(categories.items()):
        with inner[i]:
            st.markdown(f"<div class='algo-group'><strong>{group}</strong></div>", unsafe_allow_html=True)
            count = sum(st.session_state.get(f"{group}-{a}", False) for a in algos)
            picks = []
            for algo in algos:
                key = f"{group}-{algo}"
                checked = st.checkbox(
                    label=algo,
                    key=key,
                    help=df_algorithms_full[algo],
                    disabled=(count >= 2 and not st.session_state.get(key)),
                    label_visibility="visible",
                    on_change=clear_confirmation,
                )
                if checked:
                    picks.append(algo)
            selected_algos[key_map[group]] = picks
            if len(picks) != 2:
                st.markdown(
                "<div style='color: #facc15; font-size: 0.85rem; margin-top: 4px;'>⚠️ Pick 2 algorithms</div>",
                unsafe_allow_html=True
            )

with col3:
    st.markdown("<div class='highlight-header'>Heatmap Techniques</div>", unsafe_allow_html=True)
    heatmap_type = st.radio(
        "",
        ["raw", "threshold", "interpolated", "binary", "ranked"],
        key="heatmap_type", 
        label_visibility="collapsed",
        format_func=lambda s: s.capitalize()
    )

with col4:
    st.markdown("<div class='highlight-header'>Colormap</div>", unsafe_allow_html=True)
    cmaps = ["viridis", "plasma", "terrain", "coolwarm", "turbo"]

    st.markdown("""
    <style>
      div.stCheckbox > label > div[data-baseweb="checkbox"] > input[type="checkbox"] {
          border-radius: 50% !important;
      }
    </style>
    """, unsafe_allow_html=True)

    def select_cmap(chosen):
        st.session_state["colormap"] = chosen
        for other in cmaps:
            if other != chosen:
                st.session_state[f"cb_{other}"] = False

    for cmap in cmaps:
        cb_col, bar_col, _ = st.columns([0.5, 3, 0.5])
        cb_col.checkbox(
            label="",
            key=f"cb_{cmap}",
            value=(st.session_state["colormap"] == cmap),
            on_change=select_cmap,
            args=(cmap,),
            label_visibility="collapsed"
        )
        fig, ax = plt.subplots(figsize=(3, 0.3), dpi=100)
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.axis("off")
        bar_col.pyplot(fig, use_container_width=True)
        plt.close(fig)
        bar_col.markdown(
            f"<div style='text-align:center; margin-top:-10px; font-size:0.9rem;'>{cmap.capitalize()}</div>",
            unsafe_allow_html=True
        )

def explore_callback(sel_algos, hm_type):
    st.session_state["dataset"]      = st.session_state["dataset_select"]
    st.session_state["algorithms"]   = sel_algos
    st.session_state["heatmap_type"] = hm_type
    st.session_state["confirmed"]    = True
    for group, algos in categories.items():
        for algo in algos:
            st.session_state[f"{group}-{algo}"] = False

explore_disabled = not all(len(v) == 2 for v in selected_algos.values())

_, mid_col, _ = st.columns([1, 1, 1])
with mid_col:
    st.button(
        "Explore",
        key="continue_btn",
        disabled=explore_disabled,
        on_click=explore_callback,
        args=(selected_algos, heatmap_type)
    )

if st.session_state.get("confirmed"):
    dataset_name   = st.session_state["dataset"]
    selected_algos = st.session_state["algorithms"]
    heatmap_type   = st.session_state["heatmap_type"] or "interpolated"
    colormap       = st.session_state["colormap"]

    df = load_dataset(dataset_name)
    df_numeric = df.select_dtypes(include=[np.number]).loc[:, lambda d: d.std() > 0]

    if df_numeric.shape[1] < 1:
        st.error("No numeric features found for selected dataset. Heatmaps cannot be generated.")
    else:
        scaler = StandardScaler()
        imputer = SimpleImputer(strategy="mean")
        X_imp = imputer.fit_transform(df_numeric)
        X = scaler.fit_transform(X_imp)
        X += np.random.normal(0, 1e-5, X.shape)
        X_2d = PCA(n_components=2).fit_transform(X)

        x_pad = (X_2d[:,0].max() - X_2d[:,0].min()) * 0.1
        y_pad = (X_2d[:,1].max() - X_2d[:,1].min()) * 0.1
        xlim = (X_2d[:,0].min() - x_pad, X_2d[:,0].max() + x_pad)
        ylim = (X_2d[:,1].min() - y_pad, X_2d[:,1].max() + y_pad)

        @st.cache_data(show_spinner="Interpolating heatmap...", max_entries=64)
        def cached_interpolation(X2, scores, xlim, ylim):
            gx = np.linspace(xlim[0], xlim[1], 120)
            gy = np.linspace(ylim[0], ylim[1], 120)
            xx, yy = np.meshgrid(gx, gy)
            interp = RBFInterpolator(X2, scores, neighbors=20, smoothing=0.1)
            zz = interp(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            return xx, yy, zz

        def interpolate_heatmap(X2d, scores, style, cmap_name, algo, xlim, ylim, figsize=(2.5,2.0)):
            xx, yy, zz = cached_interpolation(X2d, scores, xlim, ylim)
            zz = np.nan_to_num(zz, nan=np.nanmin(zz))
            zz = np.clip(zz, zz.min(), np.percentile(zz, 99))
            fig, ax = plt.subplots(figsize=figsize)
            cmap = plt.get_cmap(cmap_name)

            if style in ["interpolated","raw"]:
                zz = gaussian_filter(zz, sigma=2.0)
                ax.contourf(xx, yy, zz, levels=60, cmap=cmap, alpha=0.95)
            elif style == "binary":
                mask = (zz >= 0.5).astype(int)
                ax.imshow(mask, extent=[*xlim,*ylim], cmap=cmap, origin="lower", aspect="auto")
            elif style == "threshold":
                th = np.percentile(zz, 95)
                mask = (zz >= th).astype(int)
                ax.imshow(mask, extent=[*xlim,*ylim], cmap=cmap, origin="lower", aspect="auto")
            elif style == "ranked":
                flat = zz.flatten()
                ranks = flat.argsort().argsort() / len(flat)
                zz = ranks.reshape(zz.shape)
                ax.contourf(xx, yy, zz, levels=50, cmap=cmap, alpha=0.95)
            else:
                raise ValueError(f"Unknown heatmap style: {style}")

            ax.scatter(X2d[:,0], X2d[:,1], s=4, c="black", alpha=0.7, linewidths=0)
            ax.set_title(
                algo,
                pad=8,
                fontsize=8,
                color="white",
                weight="bold"
            )
            fig.subplots_adjust(left=0, right=1, top=0.80, bottom=0.01)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.patch.set_facecolor("none")
            return fig

        st.markdown(
            f"<h3 style='text-align:center; margin-top:1.5rem; margin-bottom:0.5rem;'>Data: {dataset_name}</h3>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        cat_order = ["Proximity-based", "Probabilistic", "Ensembles", "Linear Models"]
        headers = st.columns([1,1,1,1,0.3])
        for i, cat in enumerate(cat_order):
            headers[i].markdown(f"<h5 style='text-align:center'>{cat}</h5>", unsafe_allow_html=True)

        container = st.container()
        cols = container.columns([1,1,1,1,0.3])
        idx_map = {cat: i for i, cat in enumerate(cat_order)}

        for cat in cat_order:
            for algo in st.session_state["algorithms"].get(key_map[cat], []):
                with cols[idx_map[cat]]:
                    try:
                        scores = run_algorithm(algo, X)
                        fig = interpolate_heatmap(
                            X_2d, scores, heatmap_type,
                            st.session_state["colormap"], algo,
                            xlim, ylim
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"{algo} failed: {e}")

        with cols[-1]:
            fig, ax = plt.subplots(figsize=(0.9, 3.29), dpi=100) #height adjust

            grad = np.linspace(0,1,256).reshape(256,1)
            ax.imshow(grad[::-1], aspect="auto",
                      cmap=st.session_state["colormap"],
                      extent=[0,1,0,1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.5, 1.02,
                "Outlier",
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=6,
                color="white"
            )
            ax.text(
                0.5, -0.02,
                "Inlier",
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=6,
                color="white"
            )
            for s in ax.spines.values():
                s.set_visible(False)
            fig.subplots_adjust(left=0.4, right=0.6, top=1.0, bottom=0.0)
            fig.patch.set_facecolor("none")
            st.pyplot(fig, use_container_width=False)






