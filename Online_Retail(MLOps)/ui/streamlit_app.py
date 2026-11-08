"""Streamlit front end for the customer-segmentation API.

Env vars:
  API_URL      base URL of the API, e.g. http://localhost:8000 or the Lambda Function URL (no trailing slash)
  API_KEY      sent as the x-api-key header (leave empty if the API has no key)
  API_TIMEOUT  seconds to wait for the API (default 60; Lambda can need ~30 s after being idle)

Run:  streamlit run ui/streamlit_app.py
"""
import os

import pandas as pd
import streamlit as st

from segment_client import TEMPLATE_CSV, ApiClient, ApiError, normalize_batch, persona_map

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "60"))

st.set_page_config(page_title="Customer Segmentation", layout="wide")


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient(API_URL, API_KEY, API_TIMEOUT)


@st.cache_data(ttl=300, show_spinner=False)
def load_segments(api_url: str):
    """Segment descriptions; cached for 5 minutes. Returns None if the API has none."""
    return get_client().segments()


client = get_client()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model")
    if st.button("Check connection"):
        try:
            with st.spinner("Contacting the API (the first call after idle can take up to 30 seconds)..."):
                h = client.health()
            st.success(f"Connected - model v{h['version']} ({h['algorithm']})")
            for w in h.get("warnings", []):
                st.warning(w)
        except ApiError as exc:
            st.error(str(exc))
    st.caption(f"API: {API_URL}")
    st.caption("Segments are groups of customers with similar buying behaviour, based on Recency "
               "(days since last purchase), Frequency (number of orders) and Monetary value (total spend).")

st.title("Customer segmentation")
tab_one, tab_batch, tab_segments = st.tabs(["One customer", "Upload a file", "About the segments"])

# ------------------------------------------------------------------ one customer
with tab_one:
    with st.form("single"):
        c1, c2, c3 = st.columns(3)
        recency = c1.number_input("Recency - days since last purchase", min_value=0, value=30, step=1)
        frequency = c2.number_input("Frequency - number of orders", min_value=0, value=5, step=1)
        monetary = c3.number_input("Monetary - total spend", min_value=0.0, value=500.0, step=50.0)
        submitted = st.form_submit_button("Find segment")
    if submitted:
        try:
            with st.spinner("Asking the model (the first call after idle can take up to 30 seconds)..."):
                res = client.predict(recency, frequency, monetary)
                segs = load_segments(API_URL)
            seg = int(res["cluster"])
            persona = persona_map(segs).get(seg)
            m1, m2 = st.columns(2)
            m1.metric("Segment", seg)
            m2.metric("Looks like", persona or "n/a")
            if segs is not None and "segment" in segs:
                row = segs[segs["segment"] == seg]
                if not row.empty:
                    st.caption("Typical customer in this segment")
                    cols = [c for c in ["customers", "share", "recency_median", "frequency_median",
                                        "monetary_median"] if c in row]
                    st.dataframe(row[cols], hide_index=True)
        except ApiError as exc:
            st.error(str(exc))

# ------------------------------------------------------------------ batch
with tab_batch:
    st.write("Upload a CSV with the columns **recency**, **frequency** and **monetary** "
             "(and optionally **customer_id**).")
    st.download_button("Download a template", TEMPLATE_CSV, file_name="customers_template.csv",
                       mime="text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            raw = pd.read_csv(up)
            data, dropped = normalize_batch(raw)
            if dropped:
                st.warning(f"{dropped} row(s) were skipped: a value was missing, not a number, or negative.")
            st.write(f"{len(data)} customers ready.")
            if st.button("Segment these customers"):
                with st.spinner("Segmenting (the first call after idle can take up to 30 seconds)..."):
                    data["segment"] = client.predict_batch(data)
                    segs = load_segments(API_URL)
                names = persona_map(segs)
                if names:
                    data["looks_like"] = data["segment"].map(names)
                st.subheader("Customers per segment")
                counts = data["segment"].value_counts().sort_index()
                st.bar_chart(counts)
                st.subheader("Result")
                st.dataframe(data)
                st.download_button("Download result as CSV", data.to_csv(index=False),
                                   file_name="customers_segmented.csv", mime="text/csv")
        except ValueError as exc:
            st.error(str(exc))
        except ApiError as exc:
            st.error(str(exc))

# ------------------------------------------------------------------ about the segments
with tab_segments:
    try:
        with st.spinner("Loading segment descriptions..."):
            segs = load_segments(API_URL)
        if segs is None:
            st.info("This API does not provide segment descriptions. Export the model after running "
                    "src/models/evaluate.py to include them.")
        else:
            st.dataframe(segs, hide_index=True)
            if {"segment", "customers"} <= set(segs.columns):
                st.bar_chart(segs.set_index("segment")["customers"])
            st.caption("'Looks like' labels are rule-of-thumb hints computed from each segment's "
                       "median Recency / Frequency / Monetary values, not a precise classification.")
    except ApiError as exc:
        st.error(str(exc))
