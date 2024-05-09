from transformers import pipeline

classifier = pipeline("ner")
result = classifier("Hello I'm Omar and I live in Zürich.")
print("Named Entities:", result)