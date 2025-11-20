# ClauseEase – AI Contract Language Simplifier

ClauseEase is an AI tool that simplifies complex legal and contract language into clear, easy-to-understand text.

## Features
- AI-powered clause simplification
- Glossary support for legal terms
- Simple Streamlit interface
- Admin dashboard for logs
- Works with text and document inputs

## How to Run
1. Install dependencies:
pip install -r requirements.txt

#Run the Streamlit app:
streamlit run app.py

#Run the Admin dashboard:
python admin.py

Docker (Optional)
Build: docker build -t clauseease .
Run: docker run -d -p 8000:8000 clauseease

Project Structure
app.py          - Streamlit UI
admin.py        - Admin Dashboard
