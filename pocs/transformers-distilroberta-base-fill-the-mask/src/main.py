from transformers import pipeline

classifier = pipeline("fill-mask")
prompt = "Brazil is the country of <mask>."
result = classifier(prompt)
print(f"{prompt} results: {result}")