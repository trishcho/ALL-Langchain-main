import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load the GROQ and OpenAI API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Set up the Streamlit app title
st.title("ChatGroq With Llama3 Demo")

# Print the current working directory to help diagnose path issues
st.write(f"Current working directory: {os.getcwd()}")

# Initialize the LLM with the Groq API key
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
st.write("LLM Initialized:", llm)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Define the directory path
directory_path = "../HUGGINGFACE/us_census"
# Check if the directory exists
if os.path.exists(directory_path):
    st.write(f"Directory '{directory_path}' exists.")
    st.write("Contents of the directory:")
    st.write(os.listdir(directory_path))
else:
    st.write(f"Directory '{directory_path}' does NOT exist. Please check the path.")

# Initialize session state if not already done
if "vectors" not in st.session_state:
    # Initialize embeddings
    st.session_state.embeddings = OpenAIEmbeddings()

    # Load documents from the specified directory
    st.session_state.loader = PyPDFDirectoryLoader(directory_path)
    st.session_state.docs = st.session_state.loader.load()

    # Debugging statement to check document loading
    st.write(f"Loaded {len(st.session_state.docs)} documents.")

    # Split the documents into smaller chunks
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

    # Debugging statement to check document chunking
    st.write(f"Created {len(st.session_state.final_documents)} document chunks after splitting.")

    # Ensure there are document chunks to process
    if st.session_state.final_documents:
        # Create vector embeddings from the document chunks
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.write("Vector embeddings created successfully.")
    else:
        st.write("No document chunks available to embed.")

# Input field for the user's question
prompt1 = st.text_input("Enter Your Question From Documents")

# Process the user's query if provided
if prompt1:
    # Check if vectors are initialized before processing the query
    if "vectors" in st.session_state:
        # Create the document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Measure the response time
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start} seconds")

        # Display the generated answer
        st.write(response['answer'])

        # Optionally display the document chunks used for the answer
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.write("Please embed the documents first by clicking 'Documents Embedding'.")
