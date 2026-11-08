"""Small client + helpers used by the Streamlit app (kept separate so they can be unit-tested)."""
from typing import Optional

import pandas as pd
import requests

REQUIRED = ["recency", "frequency", "monetary"]
TEMPLATE_CSV = (
    "customer_id,recency,frequency,monetary\n"
    "1001,12,25,5400.50\n"
    "1002,310,1,89.90\n"
    "1003,45,6,820.00\n"
)


class ApiError(Exception):
    """An error message that is safe and useful to show to the person using the app."""


class ApiClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 60, session=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs):
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        try:
            resp = getattr(self.session, method)(f"{self.base_url}{path}", headers=headers,
                                                 timeout=self.timeout, **kwargs)
        except requests.exceptions.Timeout as exc:
            raise ApiError("The API did not answer in time. If it was idle, the first request can take "
                           "up to 30 seconds - please try again.") from exc
        except requests.exceptions.RequestException as exc:
            raise ApiError(f"Cannot reach the API at {self.base_url}. Check that it is running and "
                           "that API_URL is correct.") from exc
        if resp.status_code == 401:
            raise ApiError("The API rejected the API key. Check API_KEY.")
        if resp.status_code == 503:
            raise ApiError("The API is running but has no model loaded yet.")
        if resp.status_code == 413:
            raise ApiError("Too many rows in one request.")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise ApiError(f"API error {resp.status_code}: {detail}")
        return resp

    def health(self) -> dict:
        return self._request("get", "/health").json()

    def segments(self) -> Optional[pd.DataFrame]:
        """Segment descriptions, or None if this API has none."""
        try:
            data = self._request("get", "/segments").json()["segments"]
        except ApiError as exc:
            if "404" in str(exc):
                return None
            raise
        return pd.DataFrame(data)

    def predict(self, recency: float, frequency: float, monetary: float) -> dict:
        return self._request("post", "/predict", json={"recency": recency, "frequency": frequency,
                                                       "monetary": monetary}).json()

    def predict_batch(self, df: pd.DataFrame, chunk: int = 500) -> list:
        """Segment for every row of df, sent in chunks to stay far below request-size limits."""
        out = []
        for i in range(0, len(df), chunk):
            rows = df.iloc[i:i + chunk][REQUIRED].astype(float).to_dict(orient="records")
            out.extend(self._request("post", "/predict/batch", json=rows).json()["clusters"])
        return out


def normalize_batch(df: pd.DataFrame):
    """Check an uploaded table. Returns (clean_df, n_rows_dropped); raises ValueError with a clear message."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}. Required columns: {', '.join(REQUIRED)}")
    before = len(df)
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED)
    df = df[(df[REQUIRED] >= 0).all(axis=1)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows: recency, frequency and monetary must be numbers >= 0.")
    return df, before - len(df)


def persona_map(segments: Optional[pd.DataFrame]) -> dict:
    """segment number -> persona hint text"""
    if segments is None or "segment" not in segments or "persona_hint" not in segments:
        return {}
    return {int(r["segment"]): str(r["persona_hint"]) for _, r in segments.iterrows()}
