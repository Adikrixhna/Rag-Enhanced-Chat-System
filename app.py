import io, os
import chromadb
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from Template import css, bot_template, user_template
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from embedding import SentenceEmbeddings

def get_pdf_pages(uploaded_files):
    all_pages = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            
            all_pages.extend(pages)
    
    return all_pages


def get_text_chunks(pages):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    return chunks


def get_vectorstore(doc_chunks):
    embeddings = SentenceEmbeddings(modelPath="./embedding_model")
    client = chromadb.PersistentClient(path="./vectorDB")
    collection = client.get_or_create_collection(name="PDF_Docs_new")
    db = Chroma.from_documents(
        documents=doc_chunks,
        collection_name="PDF_Docs_new",
        embedding=embeddings,
        persist_directory="./vectorDB",
    )
    return db


def get_conversation_chain(vectorstore):
    llm = ChatOllama(model="llama3")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
    return conversation_chain


def handle_userinput(user_question):
    # Ensure conversation is initialized before calling
    if st.session_state.conversation is None:
        st.warning("Please process the documents first.")
        return
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat Pdf Llama",)
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables if not already present
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat Pdf Llama")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                pages = get_pdf_pages(pdf_docs)

                # get the text chunks
                doc_chunks = get_text_chunks(pages)

                # create vector store
                vectorstore = get_vectorstore(doc_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
