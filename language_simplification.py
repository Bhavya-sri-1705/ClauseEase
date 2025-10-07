from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import re

# Download sentence tokenizer
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# ----------- Load Simplification Model -----------
model_name = "tuner007/pegasus_paraphrase"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
simplifier = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# ----------- Text Cleaning Function -----------
def clean_text(text):
    text = text.replace('\xa0', ' ')  # replace non-breaking spaces
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII
    text = re.sub(r'\s+', ' ', text).strip()  # normalize whitespace
    return text

# ----------- Simplification Function -----------
def simplify_text(text, max_length=60):
    text = clean_text(text)
    if not text:
        return "[ERROR] Empty or invalid text."

    sentences = sent_tokenize(text)
    simplified_sentences = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue  # skip blanks

        try:
            simplified = simplifier(
                sent[:1000],  # truncate very long sentences
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
            simplified_sentences.append(simplified[0]['generated_text'])
        except Exception as e:
            simplified_sentences.append(f"[Error simplifying sentence: {e}]")

    return " ".join(simplified_sentences)

# ----------- Example Usage -----------
complex_clause = """
Notwithstanding anything to the contrary contained herein, the Lessee shall indemnify and hold harmless the Lessor from any liability arising out of the Lessee's use of the premises, including but not limited to, claims of third parties.
"""

simplified_clause = simplify_text(complex_clause)

print("Original Clause:\n", complex_clause)
print("\nSimplified Clause:\n", simplified_clause)
