# rag_chat.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import chromadb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create necessary directories
os.makedirs("docs", exist_ok=True)
os.makedirs("db", exist_ok=True)

def init_groq():
    api_key = st.session_state.get('api_key')
    if not api_key:
        api_key = st.text_input("Enter your Groq API Key:", type="password")
        if api_key:
            st.session_state['api_key'] = api_key
    
    if api_key:
        return ChatGroq(
            temperature=0.7,
            model_name="mixtral-8x7b-32768",
            api_key=api_key
        )
    return None

def load_documents(directory="docs"):
    """Load documents from the specified directory"""
    try:
        loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        texts = text_splitter.split_documents(documents)
        return texts
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return []

def init_rag():
    """Initialize RAG system"""
    # Load documents
    texts = load_documents()
    
    if not texts:
        st.warning("No documents found in the docs folder. Please upload some text files.")
        return None
    
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Initialize ChromaDB with persistent storage
        client = chromadb.PersistentClient(path="./db")
        
        # Create vector store
        vectorstore = Chroma(
            persist_directory="./db",
            embedding_function=embeddings,
            client=client,
            collection_name="your_collection"
        )
        
        # Add documents if they exist
        if texts:
            vectorstore.add_documents(texts)
        
        # Initialize memory with specific output key
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Initialize Groq
        llm = init_groq()
        
        if llm:
            # Create chain WITHOUT the memory_key in combine_docs_chain_kwargs
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                return_source_documents=True
            )
            
            return qa_chain
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None
    
    return None

def main():
    st.title("RAG-Enhanced Chat System")
    
    # Initialize RAG system
    if "qa_chain" not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            st.session_state.qa_chain = init_rag()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a document (.txt)", type=['txt'])
    if uploaded_file:
        # Save uploaded file
        with open(f"docs/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        # Reinitialize RAG system
        st.session_state.qa_chain = init_rag()
    
    if not st.session_state.qa_chain:
        st.warning("Please upload some documents and enter your API key to start chatting.")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    response = st.session_state.qa_chain({"question": prompt})
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                    
                    # Display sources if available
                    if response.get("source_documents"):
                        with st.expander("View Sources"):
                            for doc in response["source_documents"]:
                                st.markdown(f"```\n{doc.page_content}\n```")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("Please check if your API key is correct and try again.")
                # Reset the chat session
                st.session_state.qa_chain = None

if __name__ == "__main__":
    main()