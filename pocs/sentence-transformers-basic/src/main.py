from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# Define the sentences
sentences = ["Magic Cards", "Magic The Gathering Cards", "Blue car is not a card"]

# Get the embeddings for the sentences
embeddings = sentense_to_embeddings(sentences)

# Use KMeans to cluster the embeddings
kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)

# Use PCA to reduce the dimensionality of the embeddings to 2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot the clusters
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=kmeans.labels_)
plt.show()