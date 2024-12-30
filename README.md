# chatbot-rag-llama

# RAG-Based Content Engine

This repository contains a Retrieval-Augmented Generation (RAG) based application using Llama3, Sentence Transformer `all-mpnet-base-v2` as the embedding model, and ChromaDB as the vector database. The application consists of three main components:

1. `app.py`
2. `Template.py`
3. `embedding.py`

## Getting Started

### Prerequisites

1. **Ollama**: Install from the [Ollama website](https://www.ollama.com). Once installed, run the executable file and in the command prompt, type:
   ```sh
   ollama pull llama3
   ```
   This will download the Llama3 model locally.

2. **Python Libraries**:
   - `sentence-transformers`
   - `chromadb`
   - `langchain`

### Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/amaanbadure/chatbot-rag-llama.git
   cd chatbot-rag-llama
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```

4. Download the Sentence Transformer model and save it in the `embedding_model` folder (create the folder first):
   ```python
   from sentence_transformers import SentenceTransformer

   modelPath = "./embedding_model"
   model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
   model.save(modelPath)
   embeddings = SentenceTransformer(modelPath)
   ```

## Project Structure

- `app.py`: The main application file that initializes the RAG-based content engine, sets up the server, and handles user queries.
- `Template.py`: Contains the template for the application's UI and other utility functions.
- `embedding.py`: Custom embedding function implementation for LangChain. Downloads the embedding model from Hugging Face Hub and uses it to generate embeddings for input text.

## Usage

1. Ensure that the Ollama executable is running and the Llama3 model is downloaded.
2. Run the application:
   ```sh
   streamlit run app.py
   ```

3. Access the application through the provided URL and interact with the RAG-based content engine.

## Architecture

1. **User Query**: The user submits a query through the application's UI.
2. **Document Retrieval**: The query is processed using the custom embeddings from the Sentence Transformer model (`all-mpnet-base-v2`). Relevant documents are retrieved from ChromaDB based on the embeddings.
3. **Response Generation**: The retrieved documents and the user query are passed to the Llama3 model (running locally via Ollama) to generate a coherent and contextually accurate response.
4. **Result**: The generated response is displayed to the user.
