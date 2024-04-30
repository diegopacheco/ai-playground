import os
import cohere  
from beeprint import pp
from cohere import ClassifyExample

def __str__(self):
    return pp(self, output=False)

co = cohere.Client(str(os.environ.get('COHERE_TOKEN')))
examples = [
    ClassifyExample(text="Dermatologists don't like her!", label="Spam"),
    ClassifyExample(text="'Hello, open to this?'", label="Spam"),
    ClassifyExample(text="I need help please wire me $1000 right now", label="Spam"),
    ClassifyExample(text="Nice to know you ;)", label="Spam"),
    ClassifyExample(text="Please help me?", label="Spam"),
    ClassifyExample(text="Your parcel will be delivered today", label="Not spam"),
    ClassifyExample(
        text="Review changes to our Terms and Conditions", label="Not spam"
    ),
    ClassifyExample(text="Weekly sync notes", label="Not spam"),
    ClassifyExample(text="'Re: Follow up from today's meeting'", label="Not spam"),
    ClassifyExample(text="Pre-read for tomorrow", label="Not spam"),
]
inputs = [
    "Confirm your email address",
    "hey i need u to send some $",
]
response = co.classify(
    inputs=inputs,
    examples=examples,
)
print(__str__(response))