import streamlit as st
import sys
from pathlib import Path
from threading import Lock
import pandas as pd
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from deployment.predict import PredictionPipeline  # uses your working predict.py


st.set_page_config(
    page_title="DNA Classifier",
    layout="wide",
)

st.title(" DNA Sequence Classification")
st.markdown("Predict DNA sequence functions using trained ML models")


@st.cache_resource(show_spinner="Loading models...")
def load_pipeline():
    return PredictionPipeline()

pipeline = load_pipeline()
model_lock = Lock()


# DETECT AVAILABLE MODELS (NO CNN CRASH)

available_models = list(pipeline.baseline_models.keys())
if pipeline.cnn_model is not None:
    available_models.insert(0, "cnn")


# SIDEBAR

with st.sidebar:
    st.header("⚙️ Settings")

    if not available_models:
        st.error(" No models available")
        st.stop()

    model = st.selectbox(
        "Select Model",
        available_models
    )

    if "cnn" not in available_models:
        st.warning(" CNN model not found. Using classical models only.")


# SINGLE PREDICTION

st.subheader("Single Sequence Prediction")

example_seq = (
    "ATGATGATGATGATGATGATGATGATGATGATGATG"
    "ATGATGATGATGATGATGATGATGATGATGATGATG"
)

if st.button(" Load Example"):
    st.session_state.sequence = example_seq

sequence = st.text_area(
    "Paste DNA Sequence (A, T, G, C only)",
    height=120,
    key="sequence"
)

if st.button(" Predict", type="primary"):
    if not sequence:
        st.warning("Please enter a DNA sequence.")
    else:
        with st.spinner("Running prediction..."):
            with model_lock:
                result = pipeline.predict(sequence, use_model=model)

        if "error" in result:
            st.error(result["error"])
        else:
            col1, col2, col3 = st.columns(3)

            col1.metric("Prediction", result["prediction"])
            col2.metric("Confidence", f"{result['confidence']:.1%}")
            col3.metric("Sequence Length", result["sequence_length"])

            probs_df = pd.DataFrame(
                result["probabilities"].items(),
                columns=["Class", "Probability"]
            )

            fig = px.bar(
                probs_df,
                x="Class",
                y="Probability",
                title="Probability Distribution",
                text_auto=".2f"
            )

            st.plotly_chart(fig, use_container_width=True)


# BATCH FASTA PREDICTION

st.divider()
st.subheader("Batch Prediction (FASTA)")

uploaded_file = st.file_uploader(
    "Upload FASTA file",
    type=["fasta", "fa", "txt"]
)

if uploaded_file:
    fasta_content = uploaded_file.read().decode("utf-8")

    sequences, headers = [], []
    current_seq, current_header = "", None

    for line in fasta_content.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_seq:
                sequences.append(current_seq)
                headers.append(current_header)
            current_header = line[1:]
            current_seq = ""
        else:
            current_seq += line

    if current_seq:
        sequences.append(current_seq)
        headers.append(current_header)

    st.info(f"Found **{len(sequences)}** sequences")

    if st.button(" Predict All", type="primary"):
        results = []
        progress = st.progress(0)

        with model_lock:
            for i, seq in enumerate(sequences):
                r = pipeline.predict(seq, use_model=model)
                r["header"] = headers[i]
                results.append(r)
                progress.progress((i + 1) / len(sequences))

        results_df = pd.DataFrame([
            {
                "Header": r["header"],
                "Prediction": r.get("prediction", "ERROR"),
                "Confidence": f"{r.get('confidence', 0):.1%}",
                "Length": r.get("sequence_length", 0)
            }
            for r in results
        ])

        st.dataframe(results_df, use_container_width=True)

        st.download_button(
            "⬇ Download Results (CSV)",
            results_df.to_csv(index=False),
            file_name="dna_predictions.csv",
            mime="text/csv"
        )


# FOOTER

st.divider()
st.caption(" DNA Classifier • Streamlit UI • Powered by PyTorch & ML models")
