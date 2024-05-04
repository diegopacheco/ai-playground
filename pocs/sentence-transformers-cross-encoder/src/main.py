from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-6", max_length=512)
scores = model.predict([("Query1", "Paragraph1"), ("Query1", "Paragraph2")])

scores = model.predict([
    ("How many people live in Berlin?", "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
    ("How many people live in Berlin?", "Berlin is well known for its museums."),
])
# [0.9315295  0.00124542]
print(scores)