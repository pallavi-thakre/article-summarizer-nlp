import streamlit as st
import json
from final_summarizer import summarize_text
from article_cleaning import clean_text

# =========================
# STREAMLIT PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Article Summarizer",
    page_icon="📝",
    layout="centered"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📝 AI-Powered Article Summarizer")
st.write("Upload or paste your article, choose summary length, and get a high-quality summary instantly.")

# =========================
# INPUT TYPE SELECTION
# =========================
input_type = st.selectbox(
    "Select article input type:",
    ["Paste as Text", "Upload TXT file", "Upload JSON file"]
)

article = ""

if input_type == "Paste as Text":
    article = st.text_area("Paste your article text here:", height=200)

elif input_type == "Upload TXT file":
    txt_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if txt_file is not None:
        article = txt_file.read().decode("utf-8")

elif input_type == "Upload JSON file":
    json_file = st.file_uploader("Upload a .json file", type=["json"])
    if json_file is not None:
        try:
            data = json.load(json_file)
            # Assuming JSON contains article under key 'article' or first text value
            if isinstance(data, dict):
                article = data.get("article", "")
            elif isinstance(data, list) and len(data) > 0:
                article = data[0].get("article", "")
        except Exception as e:
            st.error(f"Error reading JSON: {e}")

# =========================
# SUMMARY LENGTH SELECTION
# =========================
summary_type = st.selectbox(
    "Select summary length:",
    ["Detailed", "Medium", "Concise"]
)

length_map = {
    "Detailed": (248, 150),   # (max_length, min_length)
    "Medium": (124, 80),
    "Concise": (80, 20)
}

# =========================
# GENERATE SUMMARY BUTTON
# =========================


if st.button("Generate Summary"):
    if not article.strip():
        st.warning("Please provide article text first.")
    else:
        with st.spinner("Cleaning and summarizing your article..."):
            max_len, min_len = length_map[summary_type]
            clean_article = clean_text(article)
            summary = summarize_text(
                clean_article,
                max_summary_length=max_len,
                min_summary_length=min_len
            )
        st.subheader("Summary")
        st.success(summary)




