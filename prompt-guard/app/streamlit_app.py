from __future__ import annotations

import pandas as pd
import streamlit as st

from src.explain import highlight_risky_tokens
from src.predict import PromptAnalyzer
from src.preprocess import ATTACK_PATTERNS


st.set_page_config(page_title="PromptGuard — Injection Attack Detector", layout="wide")
st.title("PromptGuard — Injection Attack Detector")

if "analyzers" not in st.session_state:
    st.session_state["analyzers"] = {}


def get_analyzer(model_name: str) -> PromptAnalyzer:
    if model_name not in st.session_state["analyzers"]:
        st.session_state["analyzers"][model_name] = PromptAnalyzer(model_type=model_name)
    return st.session_state["analyzers"][model_name]


with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Model", ["classical", "bert"], format_func=lambda x: x.upper())
    show_patterns = st.checkbox("Show attack pattern list", value=False)
    if show_patterns:
        st.markdown("\n".join(f"- `{pattern}`" for pattern in ATTACK_PATTERNS))
    st.markdown("### About")
    st.write("PromptGuard classifies prompts into SAFE, SUSPICIOUS, and MALICIOUS with explanations.")

tab_single, tab_batch = st.tabs(["Single Prompt", "Batch Mode"])

with tab_single:
    user_text = st.text_area("Enter prompt", height=220, placeholder="Paste prompt text here...")
    if st.button("Analyze Prompt", type="primary"):
        if not user_text.strip():
            st.warning("Please enter prompt text.")
        else:
            analyzer = get_analyzer(selected_model)
            result = analyzer.predict(user_text)

            color = {"SAFE": "green", "SUSPICIOUS": "orange", "MALICIOUS": "red"}[result["label"]]
            st.markdown(
                f"<span style='background:{color};color:white;padding:6px 10px;border-radius:12px;'>"
                f"{result['label']}</span>",
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Confidence: {result['confidence']:.3f}")
                st.progress(min(max(result["confidence"], 0.0), 1.0))
            with col2:
                st.metric("Risk Score", f"{result['risk_score']:.1f}/100")

            with st.expander("Detected attack patterns", expanded=True):
                if result["attack_patterns"]:
                    st.markdown("\n".join(f"- `{pattern}`" for pattern in result["attack_patterns"]))
                else:
                    st.write("No known attack patterns found.")

            st.subheader("Token highlights")
            html = highlight_risky_tokens(user_text, result["token_highlights"])
            st.markdown(html or user_text, unsafe_allow_html=True)

            st.subheader("Explanation")
            st.info(result["explanation"])

with tab_batch:
    uploaded = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        if "text" not in frame.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            if st.button("Analyze Batch"):
                analyzer = get_analyzer(selected_model)
                outputs = analyzer.batch_predict(frame["text"].astype(str).tolist())
                out_df = pd.concat([frame.reset_index(drop=True), pd.DataFrame(outputs)], axis=1)
                st.dataframe(out_df.head(50), use_container_width=True)
                csv_data = out_df.to_csv(index=False)
                st.download_button("Download results CSV", data=csv_data, file_name="promptguard_results.csv")
