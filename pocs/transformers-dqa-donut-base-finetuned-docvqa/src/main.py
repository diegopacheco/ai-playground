from transformers import pipeline
from PIL import Image

pipe = pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa")

question = "What is the TOTAL?"
image = Image.open("po.png")

result = pipe(image=image, question=question)
print(f"{question} answer: {result[0]['answer']}")