# streamlit/dashboard.py
import os, io, time, uuid, joblib
from datetime import datetime
from urllib.parse import urljoin
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------- Config ----------
BACKEND_URL = os.getenv("BACKEND_URL", "")
CSV_PATH = os.getenv("CSV_PATH", "/app/frontend/coinryze_history.csv")
PRED_CSV = os.getenv("PRED_CSV", "/app/frontend/predictions.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/rf_model.joblib")
S3_BUCKET = os.getenv("S3_BUCKET")
USE_S3 = bool(S3_BUCKET)

if USE_S3:
    import boto3
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))

SIZE_THRESHOLD = int(os.getenv("SIZE_THRESHOLD", "25"))
MAX_HISTORY_FOR_MODEL = int(os.getenv("MAX_HISTORY_FOR_MODEL", "50"))

st.set_page_config(page_title="CoinryzeAnalyzer", layout="wide")
st.title("🎯 CoinryzeAnalyzer — Manual Input, Prediction & Model Training")

# ---------- Helpers ----------
def pretty_ts(ts=None):
    return (ts or datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S")

def ensure_files():
    os.makedirs(os.path.dirname(CSV_PATH) or ".", exist_ok=True)
    if not os.path.exists(CSV_PATH):
        pd.DataFrame(columns=["issue_id","timestamp","number","color","size","odd_even"]).to_csv(CSV_PATH,index=False)
    if not os.path.exists(PRED_CSV):
        pd.DataFrame(columns=["pred_id","created_at","source","last_numbers","predicted_number",
                              "predicted_color","predicted_size","odd_even","confidence","backend_used"]).to_csv(PRED_CSV,index=False)
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

def upload_to_s3(local_path, s3_key):
    try:
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        st.sidebar.success(f"Uploaded {s3_key} → s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        st.sidebar.error(f"S3 upload failed: {e}")

def load_history():
    try:
        return pd.read_csv(CSV_PATH, dtype=str)
    except:
        return pd.DataFrame(columns=["issue_id","timestamp","number","color","size","odd_even"])

def save_history(df):
    df.to_csv(CSV_PATH, index=False)
    if USE_S3:
        upload_to_s3(CSV_PATH, os.path.basename(CSV_PATH))

def load_predictions():
    try:
        return pd.read_csv(PRED_CSV, dtype=str)
    except:
        return pd.DataFrame(columns=["pred_id","created_at","source","last_numbers",
                                     "predicted_number","predicted_color","predicted_size",
                                     "odd_even","confidence","backend_used"])

def append_prediction(row: dict):
    dfp = load_predictions()
    dfp = pd.concat([dfp, pd.DataFrame([row])], ignore_index=True)
    dfp.to_csv(PRED_CSV, index=False)
    if USE_S3:
        upload_to_s3(PRED_CSV, os.path.basename(PRED_CSV))

def infer_attrs_from_number(n, size_threshold=SIZE_THRESHOLD):
    color = "Purple" if (n % 5 == 0 and n != 0) else ("Green" if n % 2 == 0 else "Red")
    size = "Big" if n >= size_threshold else "Small"
    odd_even = "Even" if n % 2 == 0 else "Odd"
    return color, size, odd_even

def pretty_predict_display(pred):
    st.write(f"🎯 **Predicted Number:** {pred['predicted_number']}")
    st.write(f"🟥 **Color:** {pred['predicted_color']}")
    st.write(f"🟩 **Size:** {pred['predicted_size']}")
    st.write(f"⚫ **Odd/Even:** {pred['odd_even']}")
    st.write(f"📊 Confidence: {pred.get('confidence',0):.2f}")

# ---------- Prediction logic ----------
def predict_backend(seq):
    try:
        resp = requests.post(urljoin(BACKEND_URL, "/predict"), json={"last_numbers": seq}, timeout=7)
        resp.raise_for_status()
        return resp.json(), True
    except Exception:
        return None, False

def predict_local_markov(df, seq):
    if df.shape[0] < 2:
        return {"predicted_number": df["number"].astype(int).mode().iloc[0] if not df.empty else 0, "confidence":0.5}
    transitions = {}
    nums = df["number"].astype(int).tolist()
    for i in range(len(nums)-1):
        a,b = nums[i],nums[i+1]
        transitions.setdefault(a,{}); transitions[a][b]=transitions[a].get(b,0)+1
    last = seq[-1]
    if last in transitions:
        nexts = transitions[last]
        pred = max(nexts.items(), key=lambda x:x[1])[0]
        conf = nexts[pred]/sum(nexts.values())
        return {"predicted_number":int(pred),"confidence":float(conf)}
    return {"predicted_number":nums[-1],"confidence":0.3}

def predict_with_model(seq, model):
    X = pd.DataFrame({"prev":seq[:-1], "curr":seq[1:]}) if len(seq)>1 else pd.DataFrame({"prev":[0],"curr":[seq[-1] if seq else 0]})
    y_pred = model.predict(X[["curr"]])
    return int(y_pred[-1])

def make_prediction(df, seq):
    if BACKEND_URL:
        res, ok = predict_backend(seq)
        if ok and res:
            pred_num = int(res.get("predicted_number", 0))
            color,size,odd_even = infer_attrs_from_number(pred_num)
            return {"predicted_number":pred_num,"predicted_color":color,"predicted_size":size,"odd_even":odd_even,"confidence":res.get("confidence",0.5),"backend_used":True}
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            pred_num = predict_with_model(seq, model)
            color,size,odd_even = infer_attrs_from_number(pred_num)
            return {"predicted_number":pred_num,"predicted_color":color,"predicted_size":size,"odd_even":odd_even,"confidence":0.7,"backend_used":False}
        except Exception as e:
            st.warning(f"Model load failed: {e}")
    mark = predict_local_markov(df, seq)
    pred_num = int(mark["predicted_number"])
    color,size,odd_even = infer_attrs_from_number(pred_num)
    return {"predicted_number":pred_num,"predicted_color":color,"predicted_size":size,"odd_even":odd_even,"confidence":mark.get("confidence",0.0),"backend_used":False}

# ---------- Model training ----------
def train_rf_model(df):
    if df.shape[0] < 10:
        st.warning("Need at least 10 rows to train.")
        return None, None
    df["num"] = df["number"].astype(int)
    df["next"] = df["num"].shift(-1)
    df = df.dropna()
    X = df[["num"]]
    y = df["next"].astype(int)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    joblib.dump(model, MODEL_PATH)
    if USE_S3:
        upload_to_s3(MODEL_PATH, os.path.basename(MODEL_PATH))
    return model, acc

# ---------- UI ----------
ensure_files()
df_hist = load_history()
df_preds = load_predictions()

st.sidebar.header("⚙️ Settings")
auto_append = st.sidebar.checkbox("Auto Append & Predict", value=True)
SIZE_THRESHOLD = st.sidebar.number_input("Size threshold (>= is Big)", 1,100, SIZE_THRESHOLD)
if USE_S3:
    st.sidebar.success(f"S3 Enabled ({S3_BUCKET})")
else:
    st.sidebar.info("S3 not configured")

# Manual input
st.subheader("✍️ Add Manual Draw")
with st.form("add_draw", clear_on_submit=True):
    cols = st.columns(3)
    issue = cols[0].text_input("issue_id", value=f"M{int(time.time())}")
    ts = cols[1].text_input("timestamp", value=pretty_ts())
    num = cols[2].number_input("number", 0,999,0)
    cols2 = st.columns(3)
    color = cols2[0].selectbox("color", ["Green","Red","Purple","Red-purple",""])
    size = cols2[1].selectbox("size", ["Small","Big",""])
    oe = cols2[2].selectbox("odd_even", ["Even","Odd",""])
    submit = st.form_submit_button("Add Row")
if submit:
    ts = pretty_ts(pd.to_datetime(ts, errors='coerce'))
    if not oe:
        oe = "Even" if num%2==0 else "Odd"
    if not size:
        size = "Big" if num>=SIZE_THRESHOLD else "Small"
    if not color:
        color = "Green" if num%2==0 else "Red"
    new = {"issue_id":issue,"timestamp":ts,"number":int(num),"color":color,"size":size,"odd_even":oe}
    df_hist = pd.concat([df_hist,pd.DataFrame([new])],ignore_index=True)
    save_history(df_hist)
    st.success("Row added ✅")
    if auto_append:
        seq = df_hist["number"].astype(int).tolist()[-MAX_HISTORY_FOR_MODEL:]
        pred = make_prediction(df_hist, seq)
        row = {
            "pred_id":str(uuid.uuid4()),
            "created_at":pretty_ts(),
            "source":"manual_append",
            "last_numbers":",".join(map(str,seq)),
            "predicted_number":pred["predicted_number"],
            "predicted_color":pred["predicted_color"],
            "predicted_size":pred["predicted_size"],
            "odd_even":pred["odd_even"],
            "confidence":pred["confidence"],
            "backend_used":pred["backend_used"]
        }
        append_prediction(row)
        st.info(f"Predicted next number: {row['predicted_number']}")

# Editable table
st.subheader("📜 Edit History")
if not df_hist.empty:
    edited = st.experimental_data_editor(df_hist, use_container_width=True, num_rows="dynamic")
    if st.button("💾 Save Edits"):
        edited.to_csv(CSV_PATH,index=False)
        if USE_S3: upload_to_s3(CSV_PATH, os.path.basename(CSV_PATH))
        st.success("Saved to CSV + S3 ✅")

# Train model
st.markdown("---")
st.subheader("🧠 Train RandomForest Model")
if st.button("Train Model"):
    model, acc = train_rf_model(df_hist)
    if model:
        st.success(f"Model trained and saved (accuracy={acc:.2f})")
    else:
        st.warning("Training skipped.")
if os.path.exists(MODEL_PATH):
    st.sidebar.success("Model loaded ✅")

# Predict
st.markdown("---")
st.subheader("🔮 Predict Next Number")
last_n = st.number_input("Use last N numbers", 1,500,30)
if st.button("Predict Now"):
    seq = df_hist["number"].astype(int).tolist()[-last_n:]
    if seq:
        pred = make_prediction(df_hist, seq)
        pretty_predict_display(pred)
        if st.button("Append prediction"):
            row = {
                "pred_id":str(uuid.uuid4()),
                "created_at":pretty_ts(),
                "source":"manual_predict",
                "last_numbers":",".join(map(str,seq)),
                **pred
            }
            append_prediction(row)
            st.info("Prediction appended ✅")

# Predictions table
st.markdown("---")
st.subheader("📈 Predictions History")
dfp = load_predictions()
if not dfp.empty:
    st.dataframe(dfp.sort_values("created_at", ascending=False))
else:
    st.info("No predictions yet.")
