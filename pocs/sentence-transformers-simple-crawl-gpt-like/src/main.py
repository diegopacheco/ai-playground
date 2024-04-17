import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Define a function to crawl a website and get its content
def get_context_from_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

# Get the context from a website
context = get_context_from_website("https://diego-pacheco.medium.com/ignoring-culture-cf3efdd2886c")

# Initialize a question-answering pipeline
nlp = pipeline('question-answering')

# Define a question
question = "what is the netflix documentary?"

# Use the pipeline to answer the question given the context
answer = nlp(question=question, context=context)

# Print the answer
print("Answer:", answer['answer'])
print("Answer:", nlp(question="What is not safe?", context=context)['answer'])
print("Answer:", nlp(question="Since when there is drama?", context=context)['answer'])
print("Answer:", nlp(question="what is ignoring culture?", context=context)['answer'])