from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def sentense_to_embeddings(sentence):
    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentence)

    # Print the embeddings
    for sentence, embedding in zip(sentence, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")    

    return embeddings

print("Magic The Gathering Cards")
sentense_to_embeddings(sentence = [
    "Magic The Gathering Cards"
])

def calculate_similarity(sentence1, sentence2):
    # Get the embeddings for the sentences
    embedding1 = model.encode([sentence1])
    embedding2 = model.encode([sentence2])

    # Calculate the cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)

    return similarity

# Test the function
sentence1 = "Magic Cards"
sentence2 = "Magic The Gathering Cards"
similarity = calculate_similarity(sentence1, sentence2)
print(f"The similarity between '{sentence1}' and '{sentence2}' is {similarity[0][0]}")

sentence1 = "Blue car is not a card"
sentence2 = "Magic The Gathering Cards"
similarity = calculate_similarity(sentence1, sentence2)
print(f"The similarity between '{sentence1}' and '{sentence2}' is {similarity[0][0]}")