import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path
import backend

st.set_page_config(page_title="Stochastic Interest Rate Lab", page_icon="📈", layout="wide")

TENORS = backend.TENORS
TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve&page&_format=csv"
)

st.markdown("""
<style>
:root {
    --navy: #10243a;
    --blue: #176b87;
    --teal: #11a7a0;
    --ink: #172b4d;
    --muted: #61738a;
    --line: #dce5ef;
}
.stApp {background: #f4f7fb; color: var(--ink);}
.block-container {padding-top: 1.15rem; padding-bottom: 3rem; max-width: 1500px;}
[data-testid="stHeader"] {background: rgba(244,247,251,.92);}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#10243a 0%,#173d59 100%);}
[data-testid="stSidebar"] * {color: #f3f7fb !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {font-weight: 600;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#c7d7e6 !important;}
.hero {
    padding: 30px 34px; border-radius: 22px;
    background: linear-gradient(115deg,#10243a 0%,#176b87 62%,#11a7a0 100%);
    color: white; box-shadow: 0 12px 30px rgba(16,36,58,.16);
    margin: 4px 0 22px 0; position: relative; overflow: hidden;
}
.hero:after {
    content:""; position:absolute; width:240px; height:240px; border-radius:50%;
    right:-65px; top:-105px; border:34px solid rgba(255,255,255,.10);
}
.hero h1 {color:white !important; font-size:2.25rem !important; margin:0 0 8px 0;}
.hero p {color:#e6f3f8 !important; font-size:1.05rem; margin:0; max-width:850px;}
.eyebrow {font-size:.76rem; letter-spacing:.13em; text-transform:uppercase;
    color:#a9e7e3; font-weight:800; margin-bottom:9px;}
h1,h2,h3 {color:#172b4d !important; letter-spacing:-.02em;}
h2 {margin-top:1.2rem !important;}
div[data-testid="stMetric"] {
    background: #ffffff; border:1px solid #dce5ef; border-radius:16px;
    padding:18px 20px; box-shadow:0 5px 16px rgba(31,55,80,.06);
    min-height:112px;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label p {
    color:#52677f !important; font-size:.88rem !important; font-weight:650 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] > div {
    color:#10243a !important; font-size:1.85rem !important; font-weight:800 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {font-weight:650;}
.stAlert {border-radius:13px;}
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border:1px solid #dce5ef; border-radius:14px; overflow:hidden;
}
div[data-testid="stPlotlyChart"] {
    background:#fff; border:1px solid #e0e8f1; border-radius:16px;
    padding:10px 10px 2px 10px; box-shadow:0 5px 16px rgba(31,55,80,.045);
}
.stButton button, .stDownloadButton button {
    border-radius:10px; font-weight:700; border:0;
}
.stButton button[kind="primary"] {background:linear-gradient(90deg,#176b87,#11a7a0);}
hr {border-color:#dce5ef;}
.small-note {color:#61738a; font-size:.88rem;}
@media (max-width: 800px) {
    .hero {padding:22px;}
    .hero h1 {font-size:1.65rem !important;}
}
</style>
""", unsafe_allow_html=True)


def standardize_treasury(df):
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    lookup = {" ".join(str(c).lower().split()): c for c in df.columns}
    aliases = {
        "Date": ["date"],
        "3 Mo": ["3 mo", "3 month", "3 months"],
        "6 Mo": ["6 mo", "6 month", "6 months"],
        "1 Yr": ["1 yr", "1 year"],
        "2 Yr": ["2 yr", "2 year"],
        "5 Yr": ["5 yr", "5 year"],
        "7 Yr": ["7 yr", "7 year"],
        "10 Yr": ["10 yr", "10 year"],
        "20 Yr": ["20 yr", "20 year"],
        "30 Yr": ["30 yr", "30 year"],
    }
    rename = {}
    for target, names in aliases.items():
        for name in names:
            if name in lookup:
                rename[lookup[name]] = target
                break
    df = df.rename(columns=rename)
    required = ["Date"] + TENORS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Treasury data is missing columns: " + ", ".join(missing))
    df = df[required].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in TENORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df.drop_duplicates("Date", keep="last").reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def load_default_data():
    year = datetime.now().year
    frames, errors = [], []
    for y in (year - 1, year):
        try:
            url = TREASURY_URL.format(year=y)
            frame = standardize_treasury(pd.read_csv(url))
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            errors.append(f"{y}: {exc}")
    if frames:
        data = pd.concat(frames, ignore_index=True)
        data = data.sort_values("Date").drop_duplicates("Date", keep="last")
        if len(data) >= 32:
            return data.reset_index(drop=True), "official"
    local = Path(__file__).parent / "data" / "historical_yields.csv"
    if local.exists():
        data = standardize_treasury(pd.read_csv(local))
        if len(data) >= 32:
            return data, "local"
    raise RuntimeError(
        "Automatic Treasury data loading failed. " + " | ".join(errors)
        + " Add data/historical_yields.csv as an offline fallback if needed."
    )


# Load data once for each visitor. No visitor upload is needed.
if "raw_data" not in st.session_state:
    try:
        with st.spinner("Loading historical Treasury rates..."):
            st.session_state.raw_data, st.session_state.data_source = load_default_data()
        st.session_state.clean_data, _ = backend._prepare_data(st.session_state.raw_data)
        st.session_state.model_result = None
        st.session_state.model_config = None
        st.session_state.model_error = None
    except Exception as exc:
        st.session_state.raw_data = None
        st.session_state.clean_data = None
        st.session_state.load_error = str(exc)

with st.sidebar:
    st.title("📈 Interest Rate Lab")
    page = st.radio("NAVIGATION", [
        "Overview", "Upload Data", "Preprocessing", "Explore Data",
        "Model & Prediction", "About Project"
    ])
    st.caption("Historical data loads automatically. Uploading is optional.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">QUANTITATIVE FINANCE · LIVE DATA DASHBOARD</div>
  <h1>Stochastic Interest Rate Lab</h1>
  <p>Explore Treasury yield curves, understand market movements and generate model-based interest-rate scenarios — with historical data loaded automatically.</p>
</div>
""", unsafe_allow_html=True)

raw = st.session_state.raw_data
clean = st.session_state.clean_data

# Generate a default forecast automatically so the visitor sees results
# without uploading a file or pressing a Run button.
DEFAULT_HORIZON = 30
DEFAULT_PATHS = 1000
if clean is not None and st.session_state.model_result is None and not st.session_state.get("model_error"):
    try:
        with st.spinner("Preparing the automatic 30-day stochastic forecast..."):
            st.session_state.model_result = backend.run_model(
                clean.copy(), DEFAULT_HORIZON, DEFAULT_PATHS
            )
            st.session_state.model_config = (
                DEFAULT_HORIZON, DEFAULT_PATHS, len(clean), str(clean["Date"].iloc[-1])
            )
    except Exception as exc:
        st.session_state.model_error = str(exc)

if page == "Overview":
    st.subheader("Market snapshot")
    st.markdown('<p class="small-note">A quick view of the latest available dataset and the model pipeline status.</p>', unsafe_allow_html=True)
    if raw is None:
        st.error(st.session_state.get("load_error", "Historical data could not be loaded."))
        st.info("The app owner should check the Treasury data connection or add a local CSV fallback.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Historical records", f"{len(clean):,}")
        c2.metric("Maturity tenors", len(TENORS))
        c3.metric("Missing cells after cleaning", int(clean.isna().sum().sum()))
        c4.metric("Data source", "U.S. Treasury" if st.session_state.data_source == "official" else "Local CSV")
        st.success("Live dataset ready · preprocessing completed · no visitor upload required.")
        last = clean.iloc[-1]
        curve = pd.DataFrame({"Maturity": TENORS, "Yield (%)": [last[t] for t in TENORS]})
        left, right = st.columns([1.15, 0.85], gap="large")
        with left:
            st.markdown("### Latest Treasury yield curve")
            fig_curve = px.line(
                curve, x="Maturity", y="Yield (%)", markers=True,
                title=f"Yield by maturity · {last['Date'].date()}"
            )
            fig_curve.update_traces(line=dict(width=3), marker=dict(size=9))
            fig_curve.update_layout(template="plotly_white", height=370,
                margin=dict(l=20,r=20,t=60,b=20), title_font_size=16,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#233b53"))
            st.plotly_chart(fig_curve, use_container_width=True)
        with right:
            st.markdown("### At a glance")
            st.metric("Latest observation", last["Date"].strftime("%d %b %Y"))
            st.metric("10-year Treasury yield", f"{last['10 Yr']:.2f}%")
            st.metric("30-year Treasury yield", f"{last['30 Yr']:.2f}%")
            st.markdown('<p class="small-note">Rates are shown as published by the U.S. Treasury. Latest date depends on the source update schedule.</p>', unsafe_allow_html=True)

        st.subheader("Latest historical observations")
        st.dataframe(clean.tail(8), use_container_width=True, hide_index=True)

        st.subheader("Automatic stochastic prediction")
        if st.session_state.model_result:
            metrics = st.session_state.model_result["metrics"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Evaluation MAE", f"{metrics['MAE']:.5f}")
            m2.metric("Evaluation RMSE", f"{metrics['RMSE']:.5f}")
            m3.metric("Evaluation R²", f"{metrics['R2']:.5f}")
            sim = st.session_state.model_result["simulation"]
            five_year = sim[sim["Tenor"] == "5 Yr"]
            forecast_fig = px.line(
                five_year, x="Future_Trading_Day", y="Mean_Prediction",
                title="5-year maturity · expected forecast path"
            )
            forecast_fig.update_traces(line=dict(width=3, color="#176b87"))
            forecast_fig.update_layout(template="plotly_white", height=360,
                margin=dict(l=20,r=20,t=60,b=20), title_font_size=16,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Future trading day", yaxis_title="Predicted yield (%)",
                font=dict(color="#233b53"))
            st.plotly_chart(forecast_fig, use_container_width=True)
            st.caption("Default scenario: 30 trading days · 1,000 simulation paths. Explore maturities and settings in Model & Prediction. Model outputs are scenarios, not guaranteed future rates.")
        elif st.session_state.get("model_error"):
            st.warning("Automatic prediction could not be generated: " + st.session_state.model_error)

elif page == "Upload Data":
    st.subheader("Optional: upload your own data")
    st.write("The official Treasury dataset is already loaded. Upload a CSV only if you want to replace it for this session.")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded is not None:
        try:
            candidate = standardize_treasury(pd.read_csv(uploaded))
            if len(candidate) < 32:
                st.error("Please provide at least 32 dated observations.")
            else:
                st.session_state.raw_data = candidate
                st.session_state.clean_data, _ = backend._prepare_data(candidate)
                st.session_state.data_source = "uploaded"
                st.session_state.model_result = None
                st.session_state.model_config = None
                st.success("Your dataset has been loaded and preprocessed.")
                st.rerun()
        except Exception as exc:
            st.error(f"Could not load CSV: {exc}")
    st.dataframe(raw.head(15) if raw is not None else pd.DataFrame(),
                 use_container_width=True, hide_index=True)

elif page == "Preprocessing":
    st.subheader("Automatic data preprocessing")
    if clean is None:
        st.error(st.session_state.get("load_error", "No dataset available."))
    else:
        a, b, c = st.columns(3)
        a.metric("Rows", len(clean))
        a.metric("Missing cells before cleaning", int(raw.isna().sum().sum()))
        c.metric("Missing cells after cleaning", int(clean.isna().sum().sum()))
        st.write("Dates and numeric values are standardized, duplicate dates are removed, and missing yield observations are interpolated and filled.")
        st.dataframe(clean.head(25), use_container_width=True, hide_index=True)
        st.download_button("Download cleaned CSV", clean.to_csv(index=False).encode("utf-8"),
                           "cleaned_interest_rate_data.csv", "text/csv")

elif page == "Explore Data":
    st.subheader("Explore historical interest rates")
    if clean is None:
        st.error(st.session_state.get("load_error", "No dataset available."))
    else:
        selected = st.multiselect("Select maturities", TENORS,
                                  default=["3 Mo", "1 Yr", "5 Yr", "10 Yr"])
        if selected:
            long_df = clean[["Date"] + selected].melt(
                id_vars="Date", var_name="Maturity", value_name="Yield (%)")
            trend_fig = px.line(long_df, x="Date", y="Yield (%)", color="Maturity",
                                title="Historical interest-rate trends")
            trend_fig.update_layout(template="plotly_white", height=470,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#233b53"), title_font_size=17)
            st.plotly_chart(trend_fig, use_container_width=True)
            curve = pd.DataFrame({"Maturity": selected,
                                  "Yield (%)": [clean.iloc[-1][t] for t in selected]})
            curve_fig = px.line(curve, x="Maturity", y="Yield (%)", markers=True,
                                title="Latest selected yield curve")
            curve_fig.update_traces(line=dict(width=3), marker=dict(size=9))
            curve_fig.update_layout(template="plotly_white", height=360,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#233b53"), title_font_size=17)
            st.plotly_chart(curve_fig, use_container_width=True)
        st.dataframe(clean[TENORS].describe().T, use_container_width=True)

elif page == "Model & Prediction":
    st.subheader("Stochastic model and prediction")
    if clean is None:
        st.error(st.session_state.get("load_error", "No dataset available."))
    else:
        x, y = st.columns(2)
        horizon = x.slider("Forecast horizon (trading days)", 7, 90, 30)
        paths = y.select_slider("Simulation paths", options=[100, 250, 500, 1000, 2000], value=1000)
        config = (horizon, paths, len(clean), str(clean["Date"].iloc[-1]))
        if st.session_state.model_result is None or st.session_state.model_config != config:
            try:
                with st.spinner("Calibrating the Vasicek model and generating simulations..."):
                    st.session_state.model_result = backend.run_model(clean.copy(), horizon, paths)
                    st.session_state.model_config = config
                    st.session_state.model_error = None
            except Exception as exc:
                st.session_state.model_result = None
                st.session_state.model_error = str(exc)
        if st.session_state.model_error:
            st.error("Model could not run: " + st.session_state.model_error)
        elif st.session_state.model_result:
            result = st.session_state.model_result
            st.success("Calibration, evaluation and stochastic simulation completed.")
            st.caption("Predictions are model-based scenarios, not guaranteed future rates.")
            metrics = result["metrics"]
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{metrics['MAE']:.5f}")
            m2.metric("RMSE", f"{metrics['RMSE']:.5f}")
            m3.metric("R²", f"{metrics['R2']:.5f}")
            sim = result["simulation"]
            tenor = st.selectbox("Choose maturity", TENORS, index=4)
            selected_sim = sim[sim["Tenor"] == tenor]
            st.plotly_chart(px.line(selected_sim, x="Future_Trading_Day",
                                    y="Mean_Prediction",
                                    title=f"{tenor}: expected future interest-rate path"),
                            use_container_width=True)
            st.dataframe(selected_sim, use_container_width=True, hide_index=True)
            with st.expander("Evaluation on held-out historical data"):
                st.dataframe(result["evaluation"], use_container_width=True, hide_index=True)
                st.dataframe(result["predictions"].head(100), use_container_width=True, hide_index=True)
            with st.expander("Calibrated Vasicek parameters"):
                st.dataframe(result["parameters"], use_container_width=True, hide_index=True)

elif page == "About Project":
    st.subheader("About this project")
    st.write("This academic prototype analyses historical U.S. Treasury par yield-curve data using a Vasicek stochastic interest-rate model.")
    st.markdown("**Workflow:** automatic data loading → preprocessing → maximum-likelihood calibration → held-out evaluation → stochastic simulation.")
    st.markdown("**Technology:** Python, Streamlit, Pandas, NumPy, SciPy, Scikit-learn and Plotly.")
    st.caption("Source: U.S. Department of the Treasury, Daily Treasury Par Yield Curve Rates. Outputs are estimates, not financial advice.")

st.markdown("---")
st.caption("Stochastic Interest Rate Lab · Historical rates load automatically for visitors.")
