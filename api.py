"""REST API backing Studio_Segment_WEB.html.

Upload a file, preview it, confirm numeric/categorical features, compute the K-Means
elbow curve, create clusters, generate cluster names/descriptions via an LLM, and
download the clustered result -- mirroring the logic in the read-only reference notebook:
C:\\Users\\Lior\\Studio_Segment_K-Means_Project.ipynb (never modified by this file).
"""

import io
import json
import os
import re
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Segment Studio K-Means API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Same Ollama credentials main.py uses for Step 4 -- kept server-side only, never sent to the browser.
OLLAMA_API_KEY = "fd5674b02970431f9edf3fb75cc1fd2b.AQiHkwqai524PWV5fhQI5mNw"
OLLAMA_MODEL = "gpt-oss:120b"

HERE = os.path.dirname(os.path.abspath(__file__))

FLOWER_KEYWORDS = {"iris", "petal", "sepal", "flower", "bloom", "plant", "species"}
ANIMAL_KEYWORDS = {"animal", "breed", "species", "weight", "legs", "paws", "claws", "dog", "cat", "bird", "fish", "zoo", "habitat", "predator", "prey"}
PRODUCT_KEYWORDS = {"product", "price", "sku", "cost", "category", "brand", "item", "sales", "revenue", "discount", "rating"}


def get_file_name(file: UploadFile) -> str:
    """Basename of the uploaded file without its extension -- the notebook's `file_name`."""
    return os.path.splitext(file.filename or "upload")[0]


def get_k_values(k_min: int, k_max: int) -> tuple:
    """Validate k_min/k_max, the range the notebook's elbow loop runs KMeans over."""
    if k_min < 1:
        raise HTTPException(status_code=400, detail="k_min must be at least 1.")
    if k_max < k_min:
        raise HTTPException(status_code=400, detail="k_max must be >= k_min.")
    return k_min, k_max


def _read_any(ext: str, contents: bytes):
    buffer = io.BytesIO(contents)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(buffer), False
    if ext == ".json":
        return pd.read_json(buffer), True
    if ext == ".tsv":
        return pd.read_csv(buffer, sep="\t"), False
    if ext == ".csv":
        return pd.read_csv(buffer), False
    # Unknown/.txt: sniff the delimiter, same idea as trying to "convert to CSV format".
    return pd.read_csv(buffer, sep=None, engine="python"), ext != ".txt"


def parse_upload(file: UploadFile, contents: bytes):
    """Read the upload into a DataFrame. Returns (file_name, df, was_converted).

    Mirrors the notebook's `file_name` assignment and its
    `if file_name == 'IRIS': original = original.drop('species', axis=1)` rule.
    """
    file_name = get_file_name(file)
    ext = os.path.splitext(file.filename or "")[1].lower()

    try:
        df, was_converted = _read_any(ext, contents)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f'Could not convert "{file.filename}" to CSV format: {exc}'
        ) from exc

    if df is None or df.empty or len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="Could not read any rows/columns from this file.")

    if file_name == "IRIS" and "species" in df.columns:
        df = df.drop("species", axis=1)

    return file_name, df, was_converted


def get_numeric_categorical(df: pd.DataFrame) -> tuple:
    """Mirrors the notebook's:
    numeric_features = [col for col in df.columns if df[col].dtype in [float, int]]
    categorical_features = [col for col in df.columns if col not in numeric_features]
    """
    numeric_features = [c for c in df.columns if df[c].dtype.kind in "iuf"]
    categorical_features = [c for c in df.columns if c not in numeric_features]
    return numeric_features, categorical_features


def parse_feature_lists(df: pd.DataFrame, numeric_json: str, categorical_json: str) -> tuple:
    """Feature lists confirmed by the user in Step 1 (after moving/deleting). Falls back to
    dtype-based auto-detection when not supplied, so the endpoints stay directly testable."""
    if numeric_json is None and categorical_json is None:
        return get_numeric_categorical(df)
    try:
        numeric_features = json.loads(numeric_json) if numeric_json else []
        categorical_features = json.loads(categorical_json) if categorical_json else []
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid feature lists: {exc}") from exc

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]
    return numeric_features, categorical_features


def build_feature_matrix(df: pd.DataFrame, numeric_features: list, categorical_features: list) -> np.ndarray:
    """Mirrors the notebook's Step-1b cell: standardize numeric_features, one-hot encode
    categorical_features (get_dummies, drop_first=True) and feed both into KMeans."""
    if not numeric_features and not categorical_features:
        raise HTTPException(status_code=400, detail="No features selected for clustering.")

    parts = []
    if numeric_features:
        parts.append(StandardScaler().fit_transform(df[numeric_features]))
    if categorical_features:
        dummies = pd.get_dummies(df[categorical_features], columns=categorical_features, drop_first=True)
        if dummies.shape[1]:
            parts.append(dummies.to_numpy(dtype=float))

    if not parts:
        raise HTTPException(status_code=400, detail="No features selected for clustering.")
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def run_kmeans(scaled: np.ndarray, k: int, n_init: int) -> KMeans:
    if k < 1:
        raise HTTPException(status_code=400, detail="k must be at least 1.")
    if k > len(scaled):
        raise HTTPException(status_code=400, detail=f"k ({k}) cannot exceed the number of rows ({len(scaled)}).")
    return KMeans(n_clusters=k, random_state=42, n_init=n_init).fit(scaled)


def df_to_records(df: pd.DataFrame) -> list:
    return json.loads(df.to_json(orient="records"))


def build_step3(labels: np.ndarray, k: int) -> list:
    counts = np.bincount(labels, minlength=k)
    return [{"cluster_id": i, "count": int(counts[i]), "name": None, "description": None} for i in range(k)]


def cluster_feature_means(df: pd.DataFrame, numeric_features: list, labels: np.ndarray, k: int) -> dict:
    """Mirrors the notebook's `df.groupby('cluster').mean(numeric_only=True)`."""
    if not numeric_features:
        return {i: {} for i in range(k)}
    working = df[numeric_features].copy()
    working["cluster"] = labels
    means = working.groupby("cluster")[numeric_features].mean()
    return {i: ({c: float(means.loc[i, c]) for c in numeric_features} if i in means.index else {}) for i in range(k)}


def cluster_categorical_modes(df: pd.DataFrame, categorical_features: list, labels: np.ndarray, k: int) -> dict:
    """Most common original category value per cluster -- the modern equivalent of the
    notebook's step_4 `_most_common` columns (dead code there since get_dummies removes
    every object-dtype column before that groupby runs)."""
    if not categorical_features:
        return {i: {} for i in range(k)}
    working = df[categorical_features].copy()
    working["cluster"] = labels
    modes = {}
    for i in range(k):
        subset = working[working["cluster"] == i]
        row = {}
        for c in categorical_features:
            m = subset[c].mode()
            row[c] = m.iat[0] if not m.empty else None
        modes[i] = row
    return modes


def build_table1(df: pd.DataFrame, numeric_features: list, labels: np.ndarray, k: int) -> tuple:
    """Mirrors the notebook's `table_1`: count + <feature>_mean per cluster."""
    means = cluster_feature_means(df, numeric_features, labels, k)
    counts = np.bincount(labels, minlength=k)
    columns = ["cluster", "count"] + [f"{c}_mean" for c in numeric_features]
    rows = []
    for i in range(k):
        row = {"cluster": i, "count": int(counts[i])}
        for c in numeric_features:
            v = means.get(i, {}).get(c)
            row[f"{c}_mean"] = round(v, 4) if v is not None else None
        rows.append(row)
    return columns, rows


def build_table2(df: pd.DataFrame, categorical_features: list, labels: np.ndarray, k: int) -> tuple:
    """Mirrors the notebook's `table_2`: most-common original category value per cluster."""
    modes = cluster_categorical_modes(df, categorical_features, labels, k)
    columns = ["cluster"] + [f"{c}_most_common" for c in categorical_features]
    rows = []
    for i in range(k):
        row = {"cluster": i}
        for c in categorical_features:
            row[f"{c}_most_common"] = modes.get(i, {}).get(c)
        rows.append(row)
    return columns, rows


def detect_subject(file_name: str, columns: list) -> str:
    """Heuristic classification of what the dataset is about, used to steer Step 4 naming
    toward real-world categories (only when the data actually looks like it's about
    plants, animals, or products -- otherwise Step 4 falls back to generic descriptive
    names). Matches whole words only: a naive substring check would flag e.g. a banking
    dataset's "education" column as animal data, since it contains "cat"."""
    tokens = set(re.findall(r"[a-z0-9]+", f"{file_name} {' '.join(columns)}".lower()))
    scores = {
        "flower": sum(1 for kw in FLOWER_KEYWORDS if kw in tokens),
        "animal": sum(1 for kw in ANIMAL_KEYWORDS if kw in tokens),
        "product": sum(1 for kw in PRODUCT_KEYWORDS if kw in tokens),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "generic"


SUBJECT_NOUNS = {
    "flower": "plant/flower species",
    "animal": "animal species or breeds",
    "product": "product categories",
}


def build_llm_prompt(numeric_features: list, categorical_features: list, step_3: list, means: dict, modes: dict, subject: str) -> str:
    lines = []
    for row in step_3:
        cid = row["cluster_id"]
        parts = [f"{c} avg={means.get(cid, {}).get(c, 0):.2f}" for c in numeric_features]
        parts += [f"most common {c}={modes.get(cid, {}).get(c)}" for c in categorical_features if modes.get(cid, {}).get(c) is not None]
        lines.append(f"Cluster {cid} (count={row['count']}): {', '.join(parts)}")

    all_features = numeric_features + categorical_features
    subject_directive = ""
    if subject in SUBJECT_NOUNS:
        subject_directive = (
            f"\nThis dataset appears to represent real-world {SUBJECT_NOUNS[subject]}. Using your knowledge "
            f"(informed by publicly available information), identify the specific real {subject} category each "
            'cluster most likely corresponds to based on its feature averages, and use that real name (not a '
            'generic made-up label) as the cluster\'s "name". Explain the match in "description".\n'
        )

    return (
        "You are analyzing the results of a K-Means clustering on a dataset with these features: "
        f"{', '.join(all_features)}.\n"
        'For each cluster below, write a short catchy "name" (2-4 words) and a one-sentence '
        '"description" that characterizes the cluster based on how its feature averages compare '
        f"to the other clusters.{subject_directive}\n"
        "Clusters:\n" + "\n".join(lines) + "\n\n"
        "Respond ONLY with a JSON array, no extra text, no markdown fences, in exactly this shape:\n"
        '[{"cluster_id":0,"name":"...","description":"..."}]'
    )


def extract_json_array(text: str):
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "Studio_Segment_WEB.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/preview")
async def preview(file: UploadFile = File(...)):
    """Step 1: parse the upload and return file_name, the table, the suggested
    numeric/categorical feature split (for the user to confirm), and the detected subject."""
    file_name, df, was_converted = parse_upload(file, await file.read())
    numeric_features, categorical_features = get_numeric_categorical(df)
    return {
        "file_name": file_name,
        "columns": df.columns.tolist(),
        "rows": df_to_records(df),
        "converted": was_converted,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "subject": detect_subject(file_name, df.columns.tolist()),
    }


@app.post("/elbow")
async def elbow(
    file: UploadFile = File(...),
    k_min: int = Form(2),
    k_max: int = Form(10),
    numeric_features: str = Form(None),
    categorical_features: str = Form(None),
):
    """Step 2: WCSS for k in [k_min, k_max] with n_init=5 -- matches the notebook's elbow loop."""
    _, df, _ = parse_upload(file, await file.read())
    numeric, categorical = parse_feature_lists(df, numeric_features, categorical_features)
    scaled = build_feature_matrix(df, numeric, categorical)

    k_min, k_max = get_k_values(k_min, k_max)
    if k_min > len(df):
        raise HTTPException(status_code=400, detail="k_min exceeds the number of rows.")
    k_max = min(k_max, len(df))

    wcss = [{"k": k, "wcss": float(run_kmeans(scaled, k, n_init=5).inertia_)} for k in range(k_min, k_max + 1)]
    return {"numeric_features": numeric, "categorical_features": categorical, "wcss": wcss}


@app.post("/clusters")
async def clusters(
    file: UploadFile = File(...),
    k: int = Form(...),
    numeric_features: str = Form(None),
    categorical_features: str = Form(None),
):
    """Step 3: cluster with n_init=10 (matches the notebook's final KMeans call). Also
    returns table_1/table_2 (the notebook's step_4 dataframes), for Step 4's display."""
    _, df, _ = parse_upload(file, await file.read())
    numeric, categorical = parse_feature_lists(df, numeric_features, categorical_features)
    scaled = build_feature_matrix(df, numeric, categorical)
    model = run_kmeans(scaled, k, n_init=10)
    table1_columns, table1_rows = build_table1(df, numeric, model.labels_, k)
    table2_columns, table2_rows = build_table2(df, categorical, model.labels_, k)
    return {
        "step_3": build_step3(model.labels_, k),
        "table_1": {"columns": table1_columns, "rows": table1_rows},
        "table_2": {"columns": table2_columns, "rows": table2_rows},
    }


@app.post("/name-clusters")
async def name_clusters(
    file: UploadFile = File(...),
    k: int = Form(...),
    numeric_features: str = Form(None),
    categorical_features: str = Form(None),
):
    """Step 4: re-cluster deterministically (same seed/n_init as Step 3), then ask the LLM
    for a name + description per cluster, based on per-cluster feature means/modes. If the
    dataset looks like it's about plants/animals/products, the prompt asks for real category
    names rather than generic descriptive ones."""
    file_name, df, _ = parse_upload(file, await file.read())
    numeric, categorical = parse_feature_lists(df, numeric_features, categorical_features)
    scaled = build_feature_matrix(df, numeric, categorical)
    model = run_kmeans(scaled, k, n_init=10)
    step_3 = build_step3(model.labels_, k)
    means = cluster_feature_means(df, numeric, model.labels_, k)
    modes = cluster_categorical_modes(df, categorical, model.labels_, k)
    subject = detect_subject(file_name, df.columns.tolist())
    prompt = build_llm_prompt(numeric, categorical, step_3, means, modes, subject)

    try:
        response = requests.post(
            "https://ollama.com/api/chat",
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.7},
            },
            timeout=60,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach the LLaMA API: {exc}") from exc

    parsed = extract_json_array(content)
    if not parsed:
        raise HTTPException(status_code=502, detail="Could not parse a JSON array from the model response.")

    by_id = {item.get("cluster_id"): item for item in parsed}
    for row in step_3:
        item = by_id.get(row["cluster_id"])
        if item:
            row["name"] = item.get("name") or ""
            row["description"] = item.get("description") or ""

    return {"step_3": step_3, "subject": subject}


@app.post("/download")
async def download(
    file: UploadFile = File(...),
    k: int = Form(...),
    names: str = Form("[]"),
    numeric_features: str = Form(None),
    categorical_features: str = Form(None),
):
    """Step 5: recompute the same clustering, attach cluster_name, return file_name_clustered.csv.

    Mirrors `original['cluster_name'] = df['cluster_name']; original.to_csv(file_name_clustered)`
    -- including pandas' default to_csv() index column.
    """
    file_name, df, _ = parse_upload(file, await file.read())
    numeric, categorical = parse_feature_lists(df, numeric_features, categorical_features)
    scaled = build_feature_matrix(df, numeric, categorical)
    model = run_kmeans(scaled, k, n_init=10)

    try:
        name_by_id = {int(item["cluster_id"]): item.get("name", "") for item in json.loads(names)}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid names payload: {exc}") from exc

    df["cluster_name"] = [name_by_id.get(int(c), "") for c in model.labels_]

    file_name_clustered = f"{file_name}_clustered.csv"
    csv_text = df.to_csv()

    # HTTP header VALUES must be Latin-1 -- a non-ASCII file_name (e.g. Hebrew) raised a
    # UnicodeEncodeError deep in the ASGI server while sending these headers, which aborted
    # the response after CORS headers had already been decided, so the browser's fetch saw a
    # bare network error ("Failed to fetch") instead of any real HTTP status. Fixed the
    # standard way (RFC 6266/5987): an ASCII-safe fallback in `filename=`, the real name
    # percent-encoded in `filename*=`. X-File-Name-Clustered is a custom header (not
    # interpreted by the browser itself), so it's simply percent-encoded outright -- the
    # frontend decodeURIComponent()s it back.
    ascii_fallback = file_name_clustered.encode("ascii", "ignore").decode("ascii") or "clustered.csv"
    encoded_name = quote(file_name_clustered)

    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{encoded_name}",
            "X-File-Name-Clustered": encoded_name,
            "Access-Control-Expose-Headers": "Content-Disposition, X-File-Name-Clustered",
        },
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8010, reload=True)
