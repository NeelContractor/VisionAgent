import tempfile
import os
import streamlit as st
from backend import run_agent

st.set_page_config(
    page_title="Vision Agent",
    page_icon="🧿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e2e2e8;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 3rem 4rem; max-width: 1100px; }

/* ── Header ── */
.vg-header {
    display: flex;
    align-items: baseline;
    gap: 14px;
    border-bottom: 1px solid #1e1e2e;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.vg-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    color: #e2e2e8;
    letter-spacing: -0.5px;
}
.vg-title span { color: #7c6af7; }
.vg-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    background: #1a1a2e;
    color: #7c6af7;
    border: 1px solid #2e2e4e;
    border-radius: 4px;
    padding: 2px 8px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #0f0f1a;
    border: 1.5px dashed #2a2a3e;
    border-radius: 12px;
    padding: 0.5rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #7c6af7; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* ── Image preview ── */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #1e1e2e;
}

/* ── Text input ── */
[data-testid="stTextInput"] input {
    background: #0f0f1a !important;
    border: 1.5px solid #2a2a3e !important;
    border-radius: 8px !important;
    color: #e2e2e8 !important;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.95rem;
    padding: 0.6rem 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #7c6af7 !important;
    box-shadow: 0 0 0 3px rgba(124, 106, 247, 0.12) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #4a4a6a !important; }

/* ── Button ── */
[data-testid="stButton"] button {
    background: #7c6af7 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px;
    padding: 0.55rem 1.6rem !important;
    transition: background 0.2s, transform 0.1s;
}
[data-testid="stButton"] button:hover {
    background: #9580ff !important;
    transform: translateY(-1px);
}
[data-testid="stButton"] button:active { transform: translateY(0); }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #7c6af7 !important; }

/* ── Answer card ── */
.answer-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #7c6af7;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.2rem;
    font-size: 0.97rem;
    line-height: 1.75;
    color: #d4d4e8;
}
.answer-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #7c6af7;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* ── Pipeline steps ── */
.pipeline {
    display: flex;
    gap: 8px;
    align-items: center;
    margin: 1.5rem 0 0.5rem;
}
.step {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #4a4a6a;
    letter-spacing: 0.5px;
}
.step.active { color: #7c6af7; }
.step-arrow { color: #2a2a3e; font-size: 0.7rem; }

/* ── Error / info ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 0.88rem;
}

/* ── Section labels ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #4a4a6a;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vg-header">
    <div class="vg-title">🧿 vision<span>agent</span></div>
    <div class="vg-badge">groq · cloud</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-label">Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop an image or click to browse",
        type=["jpg", "jpeg", "png", "webp", "gif"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.image(uploaded_file, width='stretch')

with col_right:
    st.markdown('<div class="section-label">Query</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Question",
        placeholder="What animal is in this image?",
        label_visibility="collapsed",
    )

    # Pipeline indicator
    st.markdown("""
    <div class="pipeline">
        <span class="step">VISION</span>
        <span class="step-arrow">→</span>
        <span class="step">RESEARCH</span>
        <span class="step-arrow">→</span>
        <span class="step">WRITER</span>
    </div>
    """, unsafe_allow_html=True)

    analyze_btn = st.button("Run Analysis", type="primary", disabled=not (uploaded_file and query))

    if uploaded_file and query and analyze_btn:
        suffix = os.path.splitext(uploaded_file.name)[-1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("Running vision pipeline…"):
                result = run_agent(tmp_path, query)

            st.markdown(f"""
            <div class="answer-card">
                <div class="answer-label">Answer</div>
                {result}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"**Pipeline error:** {e}")
            st.info(
                "Check Ollama is running: `ollama serve`\n\n"
                "Pull required models:\n"
                "```\nollama pull llava-phi3\nollama pull llama3.2:3b\n```"
            )
        finally:
            os.unlink(tmp_path)

    elif not uploaded_file:
        st.markdown("""
        <div style="color:#2a2a4a; font-size:0.85rem; font-family:'IBM Plex Mono',monospace; margin-top:1rem;">
            ← upload an image to begin
        </div>
        """, unsafe_allow_html=True)