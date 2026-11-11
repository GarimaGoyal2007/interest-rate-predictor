import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import importlib.util

st.set_page_config(page_title="Stochastic Interest Rate Lab", page_icon="📈", layout="wide")

st.markdown("""
<style>
.main {background:#f7f9fc}
.block-container {padding-top:1.5rem}

.hero {
    padding:22px;
    border-radius:16px;
    background:linear-gradient(110deg,#102a43,#176b87);
    color:white;
}

.hero h1 {color:white}
.hero p {color:#e6f3f8}

div[data-testid="stMetric"] {
    background:white;
    border:1px solid #e2e8f0;
    padding:14px;
    border-radius:12px;
}

div[data-testid="stMetric"] label {
    color:#16324f !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color:#102a43 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color:#16324f !important;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.title("📈 Interest Rate Lab")
page = st.sidebar.radio("NAVIGATION", [
    "Overview", "Upload Data", "Preprocessing", "Explore Data",
    "Model & Prediction", "About Project"
])
st.markdown('<div class="hero"><h1>Stochastic Interest Rate Modelling</h1><p>Interactive dashboard for yield-curve data, preprocessing and stochastic model results.</p></div>', unsafe_allow_html=True)

if "raw" not in st.session_state:
    st.session_state.raw = None
if "clean" not in st.session_state:
    st.session_state.clean = None
if "model_result" not in st.session_state:
    st.session_state.model_result = None

def nums(df):
    return list(df.select_dtypes(include=np.number).columns)

def get_backend():
    path = Path(__file__).with_name("backend.py")
    if not path.exists():
        return None, "backend.py is not present beside app.py."
    try:
        spec = importlib.util.spec_from_file_location("project_backend", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "run_model"):
            return None, "backend.py must contain run_model(dataframe, horizon, n_paths)."
        return module, None
    except Exception as e:
        return None, str(e)

if page == "Overview":
    st.subheader("Project overview")
    st.write("Upload historical interest-rate data, clean it, explore yield curves and connect your actual stochastic model.")
    if st.session_state.raw is None:
        st.info("Start from **Upload Data** in the sidebar.")
        a,b,c = st.columns(3)
        a.metric("Dataset", "Not uploaded")
        b.metric("Preprocessing", "Pending")
        c.metric("Model", "Not connected")
    else:
        d = st.session_state.raw
        a,b,c,dcol = st.columns(4)
        a.metric("Rows", len(d))
        b.metric("Columns", len(d.columns))
        c.metric("Numeric columns", len(nums(d)))
        dcol.metric("Missing cells", int(d.isna().sum().sum()))
        st.dataframe(d.head(10), use_container_width=True)

elif page == "Upload Data":
    st.subheader("Upload historical yield-curve data")
    st.write("Choose a CSV with a date column and numeric maturity columns (for example 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y).")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        try:
            st.session_state.raw = pd.read_csv(file)
            st.session_state.clean = None
            st.success("CSV uploaded successfully.")
        except Exception as e:
            st.error(f"Unable to read CSV: {e}")
    if st.session_state.raw is not None:
        st.dataframe(st.session_state.raw.head(30), use_container_width=True)
        st.download_button("Download uploaded CSV", st.session_state.raw.to_csv(index=False), "uploaded_data.csv", "text/csv")

elif page == "Preprocessing":
    st.subheader("Data preprocessing")
    if st.session_state.raw is None:
        st.warning("Upload a CSV first.")
    else:
        method = st.selectbox("Missing-value method", ["Interpolate then fill remaining gaps", "Fill with median"])
        if st.button("Run preprocessing", type="primary"):
            d = st.session_state.raw.copy()
            for col in nums(d):
                if method == "Interpolate then fill remaining gaps":
                    d[col] = d[col].interpolate(limit_direction="both").ffill().bfill()
                else:
                    d[col] = d[col].fillna(d[col].median())
            st.session_state.clean = d
            st.success("Preprocessing completed.")
        if st.session_state.clean is not None:
            x,y = st.columns(2)
            x.metric("Missing before", int(st.session_state.raw.isna().sum().sum()))
            y.metric("Missing after", int(st.session_state.clean.isna().sum().sum()))
            st.dataframe(st.session_state.clean.head(30), use_container_width=True)
            st.download_button("Download processed data", st.session_state.clean.to_csv(index=False), "processed_data.csv", "text/csv")
        st.caption("Use the same cleaning method as your validated Colab backend for final project results.")

elif page == "Explore Data":
    st.subheader("Explore historical interest rates")
    d = st.session_state.clean if st.session_state.clean is not None else st.session_state.raw
    if d is None:
        st.warning("Upload a CSV first.")
    elif not nums(d):
        st.warning("No numeric columns found in this file.")
    else:
        date_options = ["Row number"] + list(d.columns)
        date_col = st.selectbox("X-axis / date column", date_options)
        chosen = st.multiselect("Select rate/tenor columns", nums(d), default=nums(d)[:min(4,len(nums(d)))])
        if chosen:
            plot = d.copy()
            if date_col != "Row number":
                plot[date_col] = pd.to_datetime(plot[date_col], errors="coerce")
                plot = plot.sort_values(date_col)
                x = date_col
            else:
                plot["_row"] = range(len(plot))
                x = "_row"
            st.plotly_chart(px.line(plot, x=x, y=chosen, title="Historical interest-rate trends"), use_container_width=True)
            st.dataframe(plot[chosen].describe().T, use_container_width=True)
            curve = pd.DataFrame({"Tenor":chosen, "Rate":plot[chosen].iloc[-1].values})
            st.plotly_chart(px.line(curve, x="Tenor", y="Rate", markers=True, title="Latest yield curve"), use_container_width=True)
        missing = d.isna().sum().rename("Missing values").reset_index()
        missing.columns = ["Column","Missing values"]
        st.plotly_chart(px.bar(missing, x="Column", y="Missing values"), use_container_width=True)

elif page == "Model & Prediction":
    st.subheader("Run your stochastic model")
    d = st.session_state.clean if st.session_state.clean is not None else st.session_state.raw
    if d is None:
        st.warning("Upload your dataset first.")
    else:
        col1,col2 = st.columns(2)
        horizon = col1.slider("Forecast horizon (days)", 1, 365, 30)
        paths = col2.select_slider("Simulation paths", options=[100,250,500,1000,2000], value=1000)
        backend, error = get_backend()
        if backend is None:
            st.warning("The dashboard is ready, but the actual Colab model has not yet been connected.")
            st.code("def run_model(dataframe, horizon, n_paths):\n    # Call your real calibration and simulation functions here\n    return {\n        'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2},\n        'predictions': predictions_dataframe,\n        'simulation': simulation_dataframe\n    }", language="python")
            st.caption(error)
            st.info("Create backend.py beside app.py and implement run_model using your actual notebook functions. This dashboard deliberately does not invent predictions.")
        else:
            if st.button("Run model", type="primary"):
                try:
                    with st.spinner("Running backend model..."):
                        st.session_state.model_result = backend.run_model(d.copy(), horizon, paths)
                    st.success("Model completed.")
                except Exception as e:
                    st.error(f"Backend error: {e}")
            result = st.session_state.model_result
            if result:
                metrics = result.get("metrics", {})
                if metrics:
                    cards = st.columns(len(metrics))
                    for card,(name,value) in zip(cards,metrics.items()):
                        card.metric(name, f"{float(value):.5f}")
                pred = result.get("predictions")
                if isinstance(pred, pd.DataFrame):
                    st.markdown("#### Predictions")
                    st.dataframe(pred, use_container_width=True)
                    st.plotly_chart(px.line(pred, y=nums(pred), title="Predicted rates"), use_container_width=True)
                    st.download_button("Download predictions", pred.to_csv(index=False), "predictions.csv", "text/csv")
                sim = result.get("simulation")
                if isinstance(sim, pd.DataFrame):
                    st.markdown("#### Stochastic simulation")
                    st.plotly_chart(px.line(sim, y=nums(sim), title="Simulation paths"), use_container_width=True)
                    st.dataframe(sim.head(50), use_container_width=True)

elif page == "About Project":
    st.subheader("About")
    st.write("""
    **Stochastic Interest Rate Modelling and Prediction** studies the evolution
    of interest rates using historical yield-curve data and stochastic models.
    """)
    st.markdown("**Project workflow:** Data collection → preprocessing → model calibration → simulation/prediction → evaluation.")
    st.markdown("**Technology:** Python, Pandas, NumPy, SciPy, Scikit-learn, Plotly and Streamlit.")
    st.info("Edit this description to match the exact model and methods implemented in your final Colab notebook.")
