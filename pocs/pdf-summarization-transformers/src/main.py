import io
import requests
from PyPDF2 import PdfReader
from transformers import pipeline

def read_pdf_from_url(url):
    response = requests.get(url)
    with io.BytesIO(response.content) as f:
        pdf = PdfReader(f)
        text = ""
        for page in range(len(pdf.pages)):
            text += pdf.pages[page].extract_text()
    return text

def summarize_text(text):
    summarizer = pipeline("summarization")
    text_chunks = text.split('\n\n')  # split text into chunks at blank lines
    summary = ""
    print(f"Number of chunks: {len(text_chunks)}")
    for chunk in text_chunks:
        if len(chunk) > 1024:  # if chunk is too long, split it further
            sub_chunks = [chunk[i:i+1024] for i in range(0, len(chunk), 1024)]
            print(f"Number of sub-chunks: {len(sub_chunks)}")
            for sub_chunk in sub_chunks:
                summary += summarizer(sub_chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                print(f"done summarizing sub-chunk {sub_chunks.index(sub_chunk)+1}/{len(sub_chunks)}")
        else:
            summary += summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        print(f"done summarizing chunk {text_chunks.index(chunk)+1}/{len(text_chunks)}")            
    return summary

def summarize_pdf(url):
    text = read_pdf_from_url(url)
    summary = summarize_text(text)
    return summary

# Example usage: Learning Transferable Visual Models From Natural Language Supervision
url = "https://arxiv.org/pdf/2103.00020v1.pdf"
print(summarize_pdf(url))