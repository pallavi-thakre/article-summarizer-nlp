# import streamlit as st
# import torch
# import json
# from transformers import LEDTokenizer, LEDForConditionalGeneration

# # Load tokenizer and model only once using cache
# @st.cache_resource
# def load_model():
#     tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
#     model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
#     return tokenizer, model

# tokenizer, model = load_model()

# # === Function to summarize raw long text ===
# def summarize_raw_text(long_text):
#     long_text = long_text.strip()  # Clean input

#     inputs = tokenizer(
#         long_text,
#         return_tensors="pt",
#         max_length=16384,
#         truncation=True
#     )

#     global_attention_mask = torch.zeros_like(inputs["input_ids"])
#     global_attention_mask[:, ::512] = 1  # Better than only first token

#     summary_ids = model.generate(
#         input_ids=inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         global_attention_mask=global_attention_mask,
#         max_length=300,              # Allow longer summary
#         min_length=30,               # Prevent too short ones
#         num_beams=4,                 # Improves quality
#         repetition_penalty=2.0,      # Reduces repetition
#         length_penalty=1.0,          # Balanced length
#         early_stopping=True
#     )

#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# # === Function to summarize articles from JSON ===
# def summarize_json_articles(articles):
#     summarized_articles = []

#     for idx, article in enumerate(articles):
#         title = article.get("Title", f"Untitled {idx+1}")
#         content = article.get("Content", "").strip()

#         if not content:
#             continue

#         # Combine title and content for better model context
#         combined_input = f"{title}\n\n{content}"

#         summary = summarize_raw_text(combined_input)

#         summarized_articles.append({
#             "Title": title,
#             "Content": content,
#             "Summary": summary
#         })

#     return summarized_articles

# # === Streamlit UI ===
# st.title(" LED Text Summarizer")

# uploaded_file = st.file_uploader("Upload a JSON (.json) or plain text (.txt) file", type=["json", "txt"])

# if uploaded_file:
#     file_type = uploaded_file.type

#     try:
#         if file_type == "application/json":
#             st.info("📘 JSON detected. Summarizing each article...")

#             articles = json.load(uploaded_file)

#             if isinstance(articles, list):
#                 summarized = summarize_json_articles(articles)
#                 st.success(f"✅ Done! Summarized {len(summarized)} articles.")

#                 for article in summarized:
#                     st.markdown(f"### 🔹 {article['Title']}")
#                     st.markdown(f"**Summary:**\n{article['Summary']}")
#                     st.markdown("---")
#             else:
#                 st.error("❌ JSON file is not a list of articles.")

#         elif file_type == "text/plain":
#             st.info("📝 Text file detected. Generating summary...")

#             raw_text = uploaded_file.read().decode("utf-8")

#             if not raw_text.strip():
#                 st.warning("❗ Uploaded text is empty.")
#             else:
#                 summary = summarize_raw_text(raw_text)
#                 st.markdown("### 🧾 Summary")
#                 st.write(summary)

#         else:
#             st.error("❌ Unsupported file type uploaded.")

#     except Exception as e:
#         st.error(f"⚠️ Error during summarization: {e}")
import streamlit as st
import torch
import json
from transformers import LEDTokenizer, LEDForConditionalGeneration
from PIL import Image
import os

# === Try to load logo safely ===
#####################################################################################################################
# logo_path = "logo text summarization.png"  # ✅ Adjust if needed
#
# if os.path.exists(logo_path):
#     logo = Image.open(logo_path)
#     col1, col2 = st.columns([1, 5])
#     with col1:
#         st.image(logo, width=80)
#     with col2:
#         st.title("LED Text Summarizer")
# else:
#     st.title("LED Text Summarizer")  # Show title anyway
#     st.warning("⚠️ Logo not found. Please check the logo path.")
#
# # === Load tokenizer and model only once using cache ===
# @st.cache_resource
# def load_model():
#     tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
#     model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
#     return tokenizer, model
#
# tokenizer, model = load_model()
#
# # === Function to summarize raw long text ===
# def summarize_raw_text(long_text):
#     long_text = long_text.strip()
#
#     inputs = tokenizer(
#         long_text,
#         return_tensors="pt",
#         max_length=16384,
#         truncation=True
#     )
#
#     global_attention_mask = torch.zeros_like(inputs["input_ids"])
#     global_attention_mask[:, ::512] = 1
#
#     summary_ids = model.generate(
#         input_ids=inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         global_attention_mask=global_attention_mask,
#         max_length=300,
#         min_length=30,
#         num_beams=4,
#         repetition_penalty=2.0,
#         length_penalty=1.0,
#         early_stopping=True
#     )
#
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#
# # === Function to summarize articles from JSON ===
# def summarize_json_articles(articles):
#     summarized_articles = []
#
#     for idx, article in enumerate(articles):
#         title = article.get("Title", f"Untitled {idx+1}")
#         content = article.get("Content", "").strip()
#
#         if not content:
#             continue
#
#         combined_input = f"{title}\n\n{content}"
#         summary = summarize_raw_text(combined_input)
#
#         summarized_articles.append({
#             "Title": title,
#             "Content": content,
#             "Summary": summary
#         })
#
#     return summarized_articles
#
# # === Streamlit UI ===
# uploaded_file = st.file_uploader("Upload a JSON (.json) or plain text (.txt) file", type=["json", "txt"])
#
# if uploaded_file:
#     file_type = uploaded_file.type
#
#     try:
#         if file_type == "application/json":
#             st.info("📘 JSON detected. Summarizing each article...")
#
#             articles = json.load(uploaded_file)
#
#             if isinstance(articles, list):
#                 summarized = summarize_json_articles(articles)
#                 st.success(f"✅ Done! Summarized {len(summarized)} articles.")
#
#                 for article in summarized:
#                     st.markdown(f"### 🔹 {article['Title']}")
#                     st.markdown(f"**Summary:**\n{article['Summary']}")
#                     st.markdown("---")
#             else:
#                 st.error("❌ JSON file is not a list of articles.")
#
#         elif file_type == "text/plain":
#             st.info("📝 Text file detected. Generating summary...")
#
#             raw_text = uploaded_file.read().decode("utf-8")
#
#             if not raw_text.strip():
#                 st.warning("❗ Uploaded text is empty.")
#             else:
#                 summary = summarize_raw_text(raw_text)
#                 st.markdown("### 🧾 Summary")
#                 st.write(summary)
#
#         else:
#             st.error("❌ Unsupported file type uploaded.")
#
#     except Exception as e:
#         st.error(f"⚠️ Error during summarization: {e}")

#######################################################################################
import streamlit as st
import os
import json
import torch
from PIL import Image
from transformers import LEDTokenizer, LEDForConditionalGeneration

# === Logo Setup ===
logo_path = "logo text summarization.png"  # ✅ Adjust path if needed

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(logo, width=80)
    with col2:
        st.title("LED Text Summarizer")
else:
    st.title("Articles Text Summarizer")
    st.warning("⚠️ Logo not found. Please check the logo path.")

# === Sidebar: Summary Length Control ===
st.sidebar.header("🛠️ Summary Length Settings")
min_len = st.sidebar.slider("Minimum Summary Length", min_value=10, max_value=300, value=30, step=10)
max_len = st.sidebar.slider("Maximum Summary Length", min_value=50, max_value=1000, value=300, step=50)

if min_len >= max_len:
    st.sidebar.error("⚠️ Minimum length must be less than maximum length!")

# === Load tokenizer and model only once using cache ===
@st.cache_resource
def load_model():
    tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
    model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
    return tokenizer, model

tokenizer, model = load_model()

# === Function to summarize raw long text ===
def summarize_raw_text(long_text, min_len, max_len):
    long_text = long_text.strip()

    inputs = tokenizer(
        long_text,
        return_tensors="pt",
        max_length=16384,
        truncation=True
    )

    global_attention_mask = torch.zeros_like(inputs["input_ids"])
    global_attention_mask[:, ::512] = 1

    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        global_attention_mask=global_attention_mask,
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        repetition_penalty=2.0,
        length_penalty=1.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# === Function to summarize articles from JSON ===
def summarize_json_articles(articles, min_len, max_len):
    summarized_articles = []

    for idx, article in enumerate(articles):
        title = article.get("Title", f"Untitled {idx+1}")
        content = article.get("Content", "").strip()

        if not content:
            continue

        combined_input = f"{title}\n\n{content}"
        summary = summarize_raw_text(combined_input, min_len, max_len)

        summarized_articles.append({
            "Title": title,
            "Content": content,
            "Summary": summary
        })

    return summarized_articles

# === Streamlit UI ===
uploaded_file = st.file_uploader("Upload a JSON (.json) or plain text (.txt) file", type=["json", "txt"])

if uploaded_file and min_len < max_len:
    file_type = uploaded_file.type

    try:
        if file_type == "application/json":
            st.info("📘 JSON detected. Summarizing each article...")

            articles = json.load(uploaded_file)

            if isinstance(articles, list):
                summarized = summarize_json_articles(articles, min_len, max_len)
                st.success(f"✅ Done! Summarized {len(summarized)} articles.")

                for article in summarized:
                    st.markdown(f"### 🔹 {article['Title']}")
                    st.markdown(f"**Summary:**\n{article['Summary']}")
                    st.markdown("---")
            else:
                st.error("❌ JSON file is not a list of articles.")

        elif file_type == "text/plain":
            st.info("📝 Text file detected. Generating summary...")

            raw_text = uploaded_file.read().decode("utf-8")

            if not raw_text.strip():
                st.warning("❗ Uploaded text is empty.")
            else:
                summary = summarize_raw_text(raw_text, min_len, max_len)
                st.markdown("### 🧾 Summary")
                st.write(summary)

        else:
            st.error("❌ Unsupported file type uploaded.")

    except Exception as e:
        st.error(f"⚠️ Error during summarization: {e}")
