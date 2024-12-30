from sentence_transformers import SentenceTransformer

modelPath = "./embedding_model"
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model.save(modelPath)
embeddings = SentenceTransformer(modelPath)