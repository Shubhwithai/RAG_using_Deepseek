import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings  # Using OpenAI embeddings as alternative
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from openai import OpenAI # Import OpenAI

# Load environment variables
load_dotenv()
 os.getenv("OPENROUTER_API_KEY") 

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits extracted text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    embeddings = OpenAIEmbeddings()  # Using OpenAI embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Sets up a conversational chain using OpenAI client with OpenRouter."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context." Do not provide incorrect answers.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """

    client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=os.getenv('OPENROUTER_API_KEY'), # Use OPENROUTER_API_KEY
    )

    model = client.chat.completions # Use OpenAI client for chat completions

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt, verbose=False) # Pass OpenAI client to load_qa_chain
    return chain

def user_input(user_question):
    """Handles user queries by retrieving answers from the vector store."""
    embeddings = OpenAIEmbeddings()  # Using OpenAI embeddings

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.markdown(f"### Reply:\n{response['output_text']}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat PDF", page_icon=":books:", layout="wide")
    st.title("Chat with PDF using OpenRouter (deepseek-r1) :smile:") # Updated title
    st.sidebar.header("Upload & Process PDF Files")
    st.sidebar.markdown(
        "Using OpenRouter's deepseek-r1 model for advanced conversational capabilities.") # Updated sidebar description

    with st.sidebar:
        pdf_docs = st.file_uploader(
            "Upload your PDF files:",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing your files..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed and indexed successfully!")

    st.markdown(
        "### Ask Questions from Your PDF Files :mag:\n"
        "Once you upload and process your PDFs, type your questions below."
    )

    user_question = st.text_input("Enter your question:", placeholder="What do you want to know?")

    if user_question:
        with st.spinner("Fetching your answer..."):
            user_input(user_question)

    st.sidebar.info(
        "**Note:** This app uses OpenRouter's deepseek-r1 model for answering questions accurately." # Updated note
    )

if __name__ == "__main__":
    main()
