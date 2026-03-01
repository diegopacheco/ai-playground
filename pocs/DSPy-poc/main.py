import dspy

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

qa = dspy.ChainOfThought('question -> answer')

question = "What are the main benefits of using Python for data science?"
result = qa(question=question)
print(f"Question: {question}")
print(f"Reasoning: {result.reasoning}")
print(f"Answer: {result.answer}")

print("\n---\n")

summarize = dspy.ChainOfThought('text -> summary')
text = """
DSPy is a framework for algorithmically optimizing LM prompts and weights.
It unifies techniques for prompting and fine-tuning LMs as well as
approaches for reasoning, self-refinement, and augmentation with
retrieval and tools. DSPy provides composable and declarative modules
for instructing LMs in a familiar Pythonic syntax.
"""
result2 = summarize(text=text)
print(f"Summary: {result2.summary}")

print("\n---\n")

classify = dspy.Predict('sentence -> sentiment')
sentences = [
    "I love sunny days at the park!",
    "The server crashed and we lost all data.",
    "The meeting was rescheduled to tomorrow."
]
for s in sentences:
    result3 = classify(sentence=s)
    print(f"Sentence: {s}")
    print(f"Sentiment: {result3.sentiment}\n")
