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

#
# K-means clustering to group the sentence embeddings and then use 
# Principal Component Analysis (PCA) to reduce the dimensionality
# of the data for visualization
#

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

from top2vec import Top2Vec

# Define the documents
documents = ["Magic Cards", 
             "Magic The Gathering Cards", 
             "Blue car is not a card",
             "Magic The Gathering Cards are fun to play", 
             "Magic The Gathering Cards are collectible",
             "Magic The Gathering Cards are expensive",
             "Magic The Gathering Cards are popular",
             "Magic The Gathering Cards are played all over the world",
             "Magic The Gathering Cards are sold in many stores",
             "Magic The Gathering Cards are available online",
             "Magic The Gathering Cards are used in tournaments",
             "Magic The Gathering Cards are made by Wizards of the Coast",
             "Magic The Gathering Cards are released in sets",
             "Magic The Gathering Cards are traded by collectors",
             "Magic The Gathering Cards are valuable"
             "Red Cars are faster than blue cars",
             "Dinosaurs are extinct",
             "Dinosaurs lived millions of years ago",
             "Gathering means collecting",
             "Wizards are magicians",
             "Magicians perform magic tricks",
             "Magicians are skilled performers",]

#
# trick from here: https://stackoverflow.com/questions/65785949/valueerror-need-at-least-one-array-to-concatenate-in-top2vec-error
#
documents = documents * 100

# Train a Top2Vec model
model = Top2Vec(documents, speed="learn", workers=4, min_count=1
                ,umap_args={'n_neighbors': min(15, len(documents) - 1)})

# Get the number of detected topics
num_topics = model.get_num_topics()

# Get the topics
topics = model.get_topics(num_topics)

print(f"Number of topics: {num_topics}")
for i, topic in enumerate(topics, start=1):
    print(f"Topic {i}: {topic}")

#
# Topic to words
# 

# Get the number of detected topics
num_topics = model.get_num_topics()

# Loop over the topics
for i in range(num_topics):
    # Get the information about the topic
    topic_info = model.get_topic_info()

    # Print the topic number
    print(f"Topic {i + 1}:")

    # Loop over the topic words and their corresponding scores
    for word, score in zip(topic_info['words'], topic_info['scores']):
        # Print the word and its score
        print(f"    {word} (score: {score})")