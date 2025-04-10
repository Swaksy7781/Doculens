import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI with API key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API Key is missing. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

def get_pdf_text(pdf_docs):
    """
    Extract text from uploaded PDF documents
    """
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def get_text_chunks(text):
    """
    Split text into manageable chunks for processing
    """
    if not text:
        return []
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text into chunks: {str(e)}")
        return []

def get_vector_store(text_chunks):
    """
    Create and save vector embeddings of text chunks
    """
    if not text_chunks:
        return False
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def get_conversational_chain():
    """
    Create a conversational chain with the Gemini model
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

def user_input(user_question):
    """
    Process user question and generate response
    """
    try:
        # Check if vector store exists
        if not os.path.exists("faiss_index"):
            st.warning("Please upload and process PDF documents first.")
            return
        
        # Load embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Search for similar documents
        docs = new_db.similarity_search(user_question)
        
        # Check if we found any relevant documents
        if not docs:
            st.warning("No relevant information found in the documents. Try asking a different question.")
            return
        
        # Get chain and generate response
        chain = get_conversational_chain()
        if chain:
            with st.spinner("Generating response..."):
                response = chain(
                    {"input_documents": docs, "question": user_question},
                    return_only_outputs=True
                )
                
                # Display response with formatting
                st.markdown("### Response:")
                st.write(response["output_text"])
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")

def main():
    # Configure page
    st.set_page_config(
        page_title="PDF Chat with Gemini",
        page_icon="📄",
        layout="wide"
    )
    
    # Header
    st.title("📄 Chat with PDF using Gemini")
    st.markdown("""
    Upload your PDF documents and ask questions to get detailed answers powered by Google's Gemini Pro model.
    """)
    
    # Sidebar for PDF uploads
    with st.sidebar:
        st.header("📁 Document Upload")
        st.markdown("Upload your PDF files and click 'Process Documents' to analyze them.")
        
        # PDF upload
        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        # Process button
        if st.button("Process Documents"):
            if not pdf_docs:
                st.error("Please upload at least one PDF document.")
            else:
                with st.spinner("Reading and processing PDFs..."):
                    # Get the text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Check if text was extracted
                    if not raw_text:
                        st.error("Could not extract text from the provided PDFs.")
                    else:
                        # Split the text into chunks
                        st.info(f"Splitting text into chunks...")
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Create vector store
                        if text_chunks:
                            st.info(f"Creating embeddings with {len(text_chunks)} chunks...")
                            success = get_vector_store(text_chunks)
                            
                            if success:
                                st.success("✅ Documents processed successfully! You can now ask questions.")
        
        # Sidebar information
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application uses:
        - Google's Gemini Pro for answering questions
        - FAISS for efficient similarity search
        - LangChain for document processing
        """)
    
    # Main content area
    st.markdown("---")
    
    # Question input
    user_question = st.text_input(
        "Ask a question about your documents:",
        placeholder="What information does the document contain about...?"
    )
    
    # Process question when submitted
    if user_question:
        user_input(user_question)
    
    # Instructions if no question is asked
    else:
        st.info("👆 Enter your question above to get insights from your documents")
        
        # Example instructions when no documents are processed
        if not os.path.exists("faiss_index"):
            st.markdown("""
            ### Getting Started:
            1. Upload PDF documents using the sidebar
            2. Click "Process Documents" to analyze them
            3. Ask questions about the content of your documents
            """)

if __name__ == "__main__":
    main()
