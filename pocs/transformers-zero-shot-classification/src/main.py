from transformers import pipeline

pipe = pipeline(model="facebook/bart-large-mnli")
result = pipe("I have a problem with my iphone that needs to be resolved asap!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
)
print(f"{result}")