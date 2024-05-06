from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

text = "How big is London"
s1 = "London has 9,787,426 inhabitants at the 2011 census"
s2 = "London is known for its finacial district"
print(f"Text: {text} \nS1: {s1} \nS2: {s2}")

query_embedding = model.encode(text)
passage_embedding = model.encode([s1,s2])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))