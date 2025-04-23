import streamlit as st


st.set_page_config(page_title="Outlier Dashboard", layout="wide")

hide_sidebar = """
    <style>
        [data-testid="stSidebarNav"] { display: none !important; }
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }
        .css-18e3th9 { padding: 1.5rem 1rem 1rem 1rem; }
        .block-container { padding-top: 0.5rem; }
        
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
            border-radius: 6px;
            color: white;
            text-align: center;
            font-size: 14px;
            margin-bottom: 8px;
        }

        .highlight-box {
            background-color: #e0f2fe;
            padding: 10px;
            border-radius: 8px;
        }

        .highlight-header {
            background-color: #3b82f6;
            padding: 6px 10px;
            color: white;
            border-radius: 6px;
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 8px;
        }

        .stRadio > div label, .stSelectbox > div label {
            color: white !important;
            font-size: 13px !important;
        }

        .stButton>button {
            height: 2.8rem;
            width: 100%;
            font-weight: 500;
            background-color: #3b82f6;
            color: white;
        }

        .stWarning {
            font-size: 12px !important;
            padding: 8px !important;
        }

        .stSelectbox > div > div {
            padding: 4px 8px;
        }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "algorithms" not in st.session_state:
    st.session_state.algorithms = {}
if "heatmap_type" not in st.session_state:
    st.session_state.heatmap_type = None
if "colormap" not in st.session_state:
    st.session_state.colormap = "viridis"
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False

st.markdown(
    "<h2 style='text-align: center; margin-bottom: 1.2rem;'>Toward an Outlier Uncertainty Model – A Comparative Analysis</h2>",
    unsafe_allow_html=True
)


algorithms_full = {
    "CBLOF": "CBLOF – Cluster-based local outlier factor",
    "HBOS": "HBOS – Histogram-based outlier detection",
    "KNN": "KNN – K-nearest neighbors",
    "LOF": "LOF – Local outlier factor",
    "DBSCAN": "DBSCAN – Density-based clustering",

    "ABOD": "ABOD – Angle-based outlier detection",
    "GMM": "GMM – Gaussian mixture modeling",
    "KDE": "KDE – Kernel density estimation",
    "ECOD": "ECOD – Empirical cumulative outlier detection",
    "COPOD": "COPOD – Copula-based outlier detection",

    "FeatureBagging": "Feature Bagging",
    "IForest": "Isolation Forest",
    "LSCP": "LSCP – Locally selective outlier ensembles",
    "INNE": "INNE – Isolation nearest-neighbor ensembles",

    "MCD": "MCD – Minimum covariance determinant",
    "OCSVM": "OCSVM – One-class SVM",
    "PCA": "PCA – Principal component analysis",
    "LMDD": "LMDD – Linear method for deviation detection"
}

categories = {
    "Proximity-based": ["CBLOF", "HBOS", "KNN", "LOF", "DBSCAN"],
    "Probabilistic": ["ABOD", "GMM", "KDE", "ECOD", "COPOD"],
    "Ensembles": ["FeatureBagging", "IForest", "LSCP", "INNE"],
    "Linear Models": ["MCD", "OCSVM", "PCA", "LMDD"]
}
key_map = {
    "Proximity-based": "proximity",
    "Probabilistic": "probabilistic",
    "Ensembles": "ensemble",
    "Linear Models": "linear"
}


st.markdown("---")
col1, col2, col3, col4, col5 = st.columns([1.2, 2.8, 1.2, 1.2, 0.8])


with col1:
    st.markdown("<div class='highlight-header'>Dataset Selection</div>", unsafe_allow_html=True)
    with st.container(border=True):
        dataset = st.selectbox(
            "Choose Dataset",
            ["AirQualityUCI", "bank", "BeijingClimate", "nhanes", "students", "ObesityDataSet"],
            label_visibility="collapsed"
        )


selected_algos = {}
with col2:
    st.markdown("<div class='highlight-header'>Algorithm Selection</div>", unsafe_allow_html=True)
    with st.container(border=True):
        algo_cols = st.columns(4)
        for i, (group, algos) in enumerate(categories.items()):
            with algo_cols[i]:
                st.markdown(f"**{group}**")
                selected = []
                for algo in algos:
                    label = algorithms_full[algo]
                    if st.checkbox(label, key=f"{group}-{algo}"):
                        selected.append(algo)
                if len(selected) != 2:
                    st.warning("Pick 2 algorithms", icon="⚠️")

with col3:
    st.markdown("<div class='highlight-header'>Heatmap Techniques</div>", unsafe_allow_html=True)
    with st.container(border=True):
        heatmap_type = st.radio(
            "Style",
            ["raw", "threshold", "interpolated", "binary", "ranked"],
            label_visibility="collapsed"
        )


with col4:
    st.markdown("<div class='highlight-header'>Colormap</div>", unsafe_allow_html=True)
    with st.container(border=True):
        colormap = st.selectbox(
            "Colormap",
            ["viridis", "plasma", "magma", "inferno", "cividis", "coolwarm", "cubehelix", "turbo"],
            label_visibility="collapsed"
        )

with col5:
    st.markdown("<div style='visibility: hidden;'>Spacer</div>", unsafe_allow_html=True)
    if st.button("Continue", use_container_width=True):
        selected_algos = {
            key_map[group]: [algo for algo in categories[group] if st.session_state.get(f"{group}-{algo}", False)]
            for group in categories
        }
        
        if all(len(v) == 2 for v in selected_algos.values()):
            st.session_state.dataset = dataset
            st.session_state.algorithms = selected_algos
            st.session_state.heatmap_type = heatmap_type
            st.session_state.colormap = colormap
            st.session_state.confirmed = True
            st.switch_page("pages/heatmap_page.py")
        else:
            st.error("Please select exactly 2 algorithms in each category", icon="❌")
