# AI-Powered PDF Question Answering System

This project allows users to query PDF documents through an AI-driven system, which uses Google’s Gemini model for language processing. The application processes PDFs by embedding their content for easy retrieval, making it efficient to search for specific information within large documents. With a user-friendly Streamlit interface, users can upload a PDF and ask questions related to its content.

## Key Features
- **PDF Content Processing**: Efficiently splits and embeds document text for question-answer retrieval.
- **Generative Model Integration**: Utilizes Google’s Gemini model to generate responses.
- **Streamlit Interface**: Simple UI for uploading PDFs and querying them.

---

## Requirements
- **Python**: Version 3.7 or higher
- **Google Generative AI API Key**: Required for accessing Google’s Gemini model
- **Libraries**:
  - `Streamlit` (for the UI)
  - `LangChain` (for generative AI and text processing)
  - `FAISS` (for embedding and vector storage)

## Installation Guide

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Harsh772005/AI-Powered-PDF-Question-Answering-System.git
   cd AI-Powered-PDF-Question-Answering-System
2. **Install Dependencies**: Make sure you have Python 3.7+ and install the required libraries.
   ```bash
   pip install -r requirements.txt
3. **Set Up Google API Key**: You’ll need a Google API Key to access Google’s Gemini model. Set it as an environment variable:
   ``bash
   export GOOGLE_API_KEY="YOUR_API_KEY"
4. **To run the application**:
   streamlit run project_code.py
- This will launch the Streamlit interface. Here, you can:

- Upload a PDF: The application processes the content for querying.
- Ask Questions: Enter questions related to the document’s content, and the system will retrieve relevant answers based on the PDF’s text.
