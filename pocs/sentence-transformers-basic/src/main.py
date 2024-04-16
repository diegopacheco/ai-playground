from sentence_transformers import SentenceTransformer
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

sentense_to_embeddings(sentence = [
    "This framework generates embeddings for each input sentence",
    "Sentences are passed as a list of string.",
    "The quick brown fox jumps over the lazy dog."
])

print("Magic Cards")
sentense_to_embeddings(sentence = [
    "Magic Cards"
])

print("Magic The Gathering Cards")
sentense_to_embeddings(sentence = [
    "Magic The Gathering Cards"
])