import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from openai import OpenAI
import PyPDF2  # Add PyPDF2 for PDF text extraction

def get_pdf_text(pdf_files):
    """Extracts text from uploaded PDF files."""
    raw_text = ""
    for pdf_file in pdf_files:
        with pdf_file as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                raw_text += page.extract_text()
    return raw_text

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Sets up a conversational chain using OpenAI via OpenRouter."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context." Do not provide incorrect answers.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    def query_model(context, question):
        completion = OpenAI.Completion.create(
            model="deepseek/deepseek-r1",
            prompt=prompt_template.format(context=context, question=question),
            max_tokens=150
        )
        return completion.choices[0].text.strip()

    return query_model
    
def user_input(user_question):
    """Handles user queries by retrieving answers from the vector store."""
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    
    context = "\n".join([doc.page_content for doc in docs])
    response = chain(context, user_question)
    
    st.markdown(f"### Reply:\n{response}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat PDF", page_icon=":books:", layout="wide")
    st.title("Chat with PDF using OpenAI (deepseek-r1) via OpenRouter :smile:")
    
    st.sidebar.header("Upload & Process PDF Files")
    
    with st.sidebar:
        pdf_docs = st.file_uploader(
            "Upload your PDF files:",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing your files..."):
                raw_text = get_pdf_text(pdf_docs)  # This will now work
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
        "**Note:** This app uses OpenAI's deepseek-r1 model via OpenRouter for answering questions accurately."
    )

if __name__ == "__main__":
    main()
