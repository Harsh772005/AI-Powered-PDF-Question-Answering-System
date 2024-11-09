import warnings
import getpass
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

warnings.filterwarnings('ignore')

# API Key setup
if 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = getpass.getpass("Provide your Google API Key:")

# LLM setup
llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.7)

# Function to load and process PDF
def process_pdf(pdf_path):
    pdfLoader = PyPDFLoader(pdf_path)
    documents = pdfLoader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    vectordb = FAISS.from_documents(documents=texts, embedding=embeddings)
    return vectordb

# Prompt Template
prompt_template = """You are a helpful assistant to answer the question with the context without making changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Load PDF and create retriever
def setup_retriever(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={'k': 5})
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=False,
                                        chain_type_kwargs=chain_type_kwargs)
    return chain

# Streamlit UI
def main():
    st.title("AI-Powered PDF Question Answering System")
    st.write("Upload a PDF and ask a question to get started.")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        vectordb = process_pdf("uploaded_file.pdf")
        chain = setup_retriever(vectordb)

        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if question:
                result = chain(question)
                st.write("Answer:", result['result'])
            else:
                st.write("Please enter a question.")

if __name__ == "__main__":
    main()
