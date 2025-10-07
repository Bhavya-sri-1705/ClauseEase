import re  

# Dictionary of legal terms
legal_terms = {
    "indemnity": "Security or protection against a loss or other financial burden.",
    "arbitration": "A method of resolving disputes outside the courts.",
    "force majeure": "Unforeseeable circumstances that prevent someone from fulfilling a contract.",
    "breach": "A violation of a law, duty, or other form of obligation.",
    "jurisdiction": "The official power to make legal decisions and judgments."
}

# Example contract text
contract_text = """
In case of breach of contract, the parties agree to resolve the matter through arbitration. 
No party shall be liable for failure due to force majeure events.
"""

# Function to recognize legal terms
def recognize_legal_terms(text, term_dict):
    found_terms = {}
    for term in term_dict:
        # Using regex to search exact word match (case insensitive)
        if re.search(r'\b' + re.escape(term) + r'\b', text.lower()):
            found_terms[term] = term_dict[term]
    return found_terms

# Function to highlight terms in the text
def highlight_terms(text, terms):
    highlighted_text = text
    for term in terms:
        # Replace case-insensitive with highlighted version
        highlighted_text = re.sub(
            r'\b' + re.escape(term) + r'\b',
            f'[[{term.upper()}]]',
            highlighted_text,
            flags=re.IGNORECASE
        )
    return highlighted_text

# Run recognition
recognized = recognize_legal_terms(contract_text, legal_terms)

# Print recognized terms with definitions
print("Recognized Legal Terms:\n")
for term, definition in recognized.items():
    print(f"{term}: {definition}")

# Print highlighted contract text
print("\nHighlighted Contract Text:\n")
print(highlight_terms(contract_text, recognized.keys()))
