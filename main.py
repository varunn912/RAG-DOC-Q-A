import os
from flask import Flask, render_template, request
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Global variable for the retrieval chain
retrieval_chain = None

def initialize_rag_pipeline():
    """
    Initializes the RAG pipeline: loads documents, creates embeddings,
    builds a vector store, and sets up the retrieval chain.
    This function runs only once when the application starts.
    """
    global retrieval_chain
    if retrieval_chain is None:
        print("Initializing RAG pipeline...")
        # 1. Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 2. Data Ingestion & Loading
        loader = PyPDFDirectoryLoader("./research_papers")
        docs = loader.load()

        # 3. Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        # 4. Vector Store (FAISS)
        vectors = FAISS.from_documents(final_documents, embeddings)

        # 5. LLM Initialization
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

        # 6. Prompt Template
        prompt_template = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate and detailed response based on the question.
            <context>
            {context}
            <context>
            Question: {input}
            """
        )

        # 7. Create Chains
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        print("RAG pipeline initialized successfully!")

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles both GET requests (displaying the page) and POST requests (processing the form).
    """
    if request.method == 'POST':
        user_query = request.form['query']
        if user_query and retrieval_chain:
            # Invoke the chain with the user's query
            response = retrieval_chain.invoke({'input': user_query})
            # Render the page with the results
            return render_template('index.html', answer=response['answer'], context=response['context'])
    
    # For a GET request, just render the initial page
    return render_template('index.html')

if __name__ == '__main__':
    # Initialize the RAG pipeline before starting the server
    initialize_rag_pipeline()
    # Run the Flask app
    app.run(debug=True)