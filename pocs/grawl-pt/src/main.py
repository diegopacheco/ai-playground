import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import gradio as gr
from requests_html import HTMLSession

def get_context_from_website(url):
    session = HTMLSession()
    response = session.get(url)
    return response.text

nlp = pipeline('question-answering')

def answer_question(question,site):
    post = get_context_from_website(site)
    answer = nlp(question=question, context=post)
    return answer['answer']

examples = [
    ["What is Distributed Monolith?","http://diego-pacheco.blogspot.com/2023/07/distributed-monolith.html"],
    ["What is Technical Debt?","https://en.wikipedia.org/wiki/Technical_debt"],
    ["What is love?","https://genius.com/Haddaway-what-is-love-lyrics"]
]

ui = gr.Interface(fn=answer_question, 
                  title="Diego's CrawlPT - LLM chat feed from posts",
                  description="Ask a question about a post",
                  examples=examples,
                  inputs=["text", "text"], 
                  outputs="text")
ui.launch(server_port=8080, share=False)