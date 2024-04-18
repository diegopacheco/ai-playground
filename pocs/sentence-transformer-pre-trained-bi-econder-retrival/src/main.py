from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
docs = [
    "My first paragraph. That contains information",
    "Python is a programming language.",
]
document_embeddings = model.encode(docs)

query = "What is Python?"
query_embedding = model.encode(query)

for doc, score in sorted(
    zip(docs, document_embeddings @ query_embedding.T), 
    key=lambda x: x[1], reverse=True
):
    print(doc, score)