import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(page_title="NSL-KDD Intrusion Detector", layout="wide")

# The 41 expected feature columns
FEATURE_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

# Load model and preprocessors once and keep them cached in memory
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("lstm_model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, preprocessor, label_encoder

model, preprocessor, label_encoder = load_artifacts()

st.title("Network Intrusion Detection System (NSL-KDD LSTM)")
st.write("Upload a CSV file containing network connection features to detect cyberattacks.")

uploaded_file = st.file_uploader("Upload network traffic CSV", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    # Check if all 41 required feature columns exist
    missing_cols = [c for c in FEATURE_COLS if c not in raw_df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing {len(missing_cols)} required columns, e.g.: {missing_cols[:5]}")
        st.stop()

    # Keep only the 41 feature columns in the exact training order
    X_input = raw_df[FEATURE_COLS]

    with st.spinner("Analyzing network traffic with LSTM..."):
        X_processed = preprocessor.transform(X_input).astype(np.float32)
        X_lstm = X_processed.reshape((X_processed.shape[0], 1, X_processed.shape[1]))
        probs = model.predict(X_lstm, verbose=0)
        pred_indices = np.argmax(probs, axis=1)
        pred_labels = label_encoder.inverse_transform(pred_indices)
        confidences = np.max(probs, axis=1) * 100

    # Combine predictions with the uploaded rows
    results_df = raw_df.copy()
    results_df.insert(0, "Confidence (%)", np.round(confidences, 2))
    results_df.insert(0, "Predicted Class", pred_labels)

    # --- Top KPI Summary ---
    total_rows = len(results_df)
    normal_count = int((results_df["Predicted Class"] == "normal").sum())
    attack_count = total_rows - normal_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Connections Scanned", f"{total_rows:,}")
    col2.metric("Normal Traffic", f"{normal_count:,}")
    col3.metric("Attacks Detected", f"{attack_count:,}")

    # --- Attack Type Counts ---
    st.subheader("Detection Counts by Class")
    counts = results_df["Predicted Class"].value_counts().reindex(
        ["normal", "DoS", "Probe", "R2L", "U2R"], fill_value=0
    )
    count_df = counts.rename_axis("Traffic Type").reset_index(name="Count")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(count_df, use_container_width=True, hide_index=True)
    with c2:
        st.bar_chart(counts)

    # --- Prediction Table with Filter ---
    st.subheader("Prediction Table")
    selected_classes = st.multiselect(
        "Filter by predicted class:",
        options=["normal", "DoS", "Probe", "R2L", "U2R"],
        default=["DoS", "Probe", "R2L", "U2R", "normal"]
    )
    filtered_df = results_df[results_df["Predicted Class"].isin(selected_classes)]
    st.dataframe(filtered_df, use_container_width=True)

    # Download button for predictions
    csv_out = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions as CSV",
        data=csv_out,
        file_name="intrusion_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("Upload `sample_traffic.csv` above to see predictions and attack counts.")