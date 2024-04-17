from sentence_transformers import SentenceTransformer

# Pre-trained model
embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")

embeddings = embedder.encode(["Hello World", "Hallo Welt", "Hola mundo"])
print(embeddings)