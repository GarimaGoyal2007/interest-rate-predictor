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
.block-container {padding-top: 1.4rem;}
.hero {padding:24px;border-radius:16px;background:linear-gradient(110deg,#102a43,#176b87);color:white;}
.hero h1 {color:white;}
.hero p {color:#e6f3f8;}
div[data-testid="stMetric"] {background:white;border:1px solid #e2e8f0;padding:14px;border-radius:12px;}
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
<h1>Stochastic Interest Rate Modelling</h1>
<p>Historical yield curves, automatic preprocessing, Vasicek calibration and stochastic predictions.</p>
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
    st.subheader("Project overview")
    if raw is None:
        st.error(st.session_state.get("load_error", "Historical data could not be loaded."))
        st.info("The app owner should check the Treasury data connection or add a local CSV fallback.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Historical records", f"{len(clean):,}")
        c2.metric("Maturity tenors", len(TENORS))
        c3.metric("Missing cells after cleaning", int(clean.isna().sum().sum()))
        c4.metric("Data source", "U.S. Treasury" if st.session_state.data_source == "official" else "Local CSV")
        st.success("Data is ready automatically. Visitors do not need to upload a file.")
        st.subheader("Latest historical observations")
        st.dataframe(clean.tail(10), use_container_width=True, hide_index=True)
        last = clean.iloc[-1]
        curve = pd.DataFrame({"Maturity": TENORS, "Yield (%)": [last[t] for t in TENORS]})
        st.plotly_chart(px.line(curve, x="Maturity", y="Yield (%)", markers=True,
                                title=f"Latest yield curve — {last['Date'].date()}"),
                        use_container_width=True)

        st.subheader("Automatic stochastic prediction")
        if st.session_state.model_result:
            metrics = st.session_state.model_result["metrics"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Evaluation MAE", f"{metrics['MAE']:.5f}")
            m2.metric("Evaluation RMSE", f"{metrics['RMSE']:.5f}")
            m3.metric("Evaluation R²", f"{metrics['R2']:.5f}")
            sim = st.session_state.model_result["simulation"]
            five_year = sim[sim["Tenor"] == "5 Yr"]
            st.plotly_chart(
                px.line(five_year, x="Future_Trading_Day", y="Mean_Prediction",
                        title="Automatic 30-trading-day forecast — 5-year maturity"),
                use_container_width=True
            )
            st.caption("Default forecast: 30 trading days and 1,000 stochastic simulation paths. Open Model & Prediction to explore other maturities and settings.")
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
            st.plotly_chart(px.line(long_df, x="Date", y="Yield (%)", color="Maturity",
                                    title="Historical interest-rate trends"),
                            use_container_width=True)
            curve = pd.DataFrame({"Maturity": selected,
                                  "Yield (%)": [clean.iloc[-1][t] for t in selected]})
            st.plotly_chart(px.line(curve, x="Maturity", y="Yield (%)", markers=True,
                                    title="Latest selected yield curve"),
                            use_container_width=True)
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
