from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
# from .article_cleaning import clean_text

# =========================
# 1. LOAD YOUR FINE-TUNED MODEL
# =========================
# Replace with the path where your fine-tuned model is saved
MODEL_PATH = "./finetuned_model_path"

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# Set to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# length=124
# =========================
# 2. SUMMARIZATION FUNCTION
# =========================
def summarize_text(article, max_input_length=512, max_summary_length=150, min_summary_length=30):
    """
    Generates a summary for a given article using the fine-tuned T5 model.
    """
    # Prepend task prefix (T5 expects this format unless you changed it during training)
    input_text = "summarize: " + article.strip()

    # Tokenize input
    inputs = tokenizer(
        input_text,
        max_length=max_input_length,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    ).to(device)

    # Generate summary ids
    summary_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_summary_length,
        min_length=min_summary_length,
        length_penalty=2.0,
        num_beams=6,
        early_stopping=True
    )

    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# =========================
# 3. TEST ON NEW ARTICLE
# =========================
# if __name__ == "__main__":
    # article=""
# Cleaning= the Given article
# clean_article=clean_text(article)

#     new_article = """
#     Teenagers are using dating apps more than we previously knew, according to research published this week in the Journal of Psychopathology and Clinical Science. The study found that 23.5% of teens ages 13 through 18 used dating apps over a six-month period, which is more than past estimates.
# The study is believed to be the first to track how teens use dating apps by recording their keyboard activity rather than relying on self-reports, according to the researchers.
# The study found that teens who used dating apps didn’t generally have more symptoms of mental health challenges after six months than those who didn’t. However, those who used dating apps frequently were more likely to have symptoms of major depressive disorders.
#     """

    # print("\nGenerated Summary:\n", summary)
# summary = summarize_text(clean_article)
# print("\nCleaned Article:\n", clean_article)  # this is to be given as streamlit summary output

