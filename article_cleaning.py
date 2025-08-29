import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords once
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
    text = re.sub(r'[^\w\s]', '', text)   # Remove special characters
    text = re.sub(r'\d+', '', text)       # Remove digits
    words = text.lower().split()
    cleaned = ' '.join([word for word in words if word not in STOP_WORDS])
    return cleaned

# # Example usage:
# raw_text = "This is an example sentence, with numbers like 123 and special characters!!!"
# print(clean_text(raw_text))
