import re
import nltk
import spacy

from nltk.tokenize import sent_tokenize

# Initialize NLTK and spaCy
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")


# ----------- TEXT CLEANING -----------

def clean_text(text: str) -> str:
    """
    Normalize and clean legal text.
    """
    text = text.replace('\xa0', ' ')  # non-breaking space
    text = re.sub(r'[\t\r\f\v]', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r"[’‘]", "'", text)
    return text.strip()
def segment_clauses(text: str) -> list:
    """
    Segment text into clauses based on legal numbering patterns.
    Supports: 1.1, 2.3(a), 3.1.2, etc.
    """
    pattern = re.compile(r'(?=\n?\d{1,2}(\.\d{1,2})+[\)\.]?\s)')
    parts = pattern.split(text)
    clauses = []

    if parts:
        temp = ""
        for part in parts:
            if re.match(r'\d{1,2}(\.\d{1,2})+[\)\.]?', part.strip()):
                if temp:
                    clauses.append(temp.strip())
                temp = part
            else:
                temp += part
        if temp:
            clauses.append(temp.strip())
    else:
        clauses = [text.strip()]
    return clauses
def split_sentences(text: str) -> list:
    """
    Split a clause into individual sentences.
    """
    return sent_tokenize(text)


# ----------- NAMED ENTITY EXTRACTION -----------

def extract_entities(text: str) -> list:
    """
    Extract named entities from a clause using spaCy.
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
def preprocess_clause(clause_text: str) -> dict:
    """
    Process a single clause:
    - Clean text
    - Split into sentences
    - Extract entities
    """
    cleaned = clean_text(clause_text)
    sentences = split_sentences(cleaned)
    entities = extract_entities(cleaned)

    return {
        "raw_text": clause_text,
        "cleaned_text": cleaned,
        "sentences": sentences,
        "entities": entities
    }
# ----------- BATCH PROCESSING -----------

def preprocess_contract_text(raw_text: str) -> list:
    """
    Preprocess an entire contract's text:
    - Clean
    - Segment into clauses
    - Process each clause
    """
    cleaned_text = clean_text(raw_text)
    clauses = segment_clauses(cleaned_text)

    processed = []
    for clause in clauses:
        result = preprocess_clause(clause)
        processed.append(result)

    return processed
#from text_preprocessing import preprocess_contract_text

with open("sample_contract.txt", "r", encoding="utf-8") as file:
    contract_text = file.read()

processed = preprocess_contract_text(contract_text)

# Example: print the first clause's data
print(processed[0])
