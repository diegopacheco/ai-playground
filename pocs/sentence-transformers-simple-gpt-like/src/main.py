from transformers import pipeline

# Initialize a question-answering pipeline
nlp = pipeline('question-answering')

# Define a context
context = "Gremio is the team who most win campeonato gaucho in Rio Grande do Sul."

# Define a question
question = "What is the biggest winner of the campeonato gaucho?"

# Use the pipeline to answer the question given the context
answer = nlp(question=question, context=context)

# Print the answer
print("Answer:", answer['answer'])