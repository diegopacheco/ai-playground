from transformers import pipeline

qa_model = pipeline("question-answering")
question = "Where do I live?"
context = "My name is Merve and I live in İstanbul."
result = qa_model(question = question, context = context)

print(f"question: {question} \ncontext: {context} \nresult: {result}")
