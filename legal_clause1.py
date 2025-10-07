from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load Legal-BERT model and tokenizer
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Clause label mapping
clause_labels = {
    0: "Confidentiality",
    1: "Termination",
    2: "Indemnity",
    3: "Dispute Resolution",
    4: "Governing Law"
}

# Function to detect clause type
def detect_clause_type(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return clause_labels[predicted_class]

# Test the function
sample_clause = "Any dispute shall be resolved by arbitration."
detected_type = detect_clause_type(sample_clause)
print(f"Detected Clause Type: {detected_type}")
