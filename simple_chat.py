# simple_chat.py
import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq
def init_groq():
    # Explicitly set API key
    os.environ["GROQ_API_KEY"] = "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"  # Replace with your API key
    
    return ChatGroq(
        temperature=0.7,
        model_name="mixtral-8x7b-32768",
        api_key=os.environ["GROQ_API_KEY"]  # Explicitly pass API key
    )

# Create Streamlit interface
def main():
    st.title("Chat with Groq")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Initialize Groq
    if "groq_chat" not in st.session_state:
        st.session_state.groq_chat = init_groq()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Groq response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.groq_chat.invoke(prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})

if __name__ == "__main__":
    main()