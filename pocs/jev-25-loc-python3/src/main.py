import sys

import numpy

from jev import classify, load_model

CHOICES = ["Legitimate", "Spam", "Phishing"]
EMAILS = [
    "Payroll asks for your password on a non-company sign-in page.",
    "Congratulations! You won a free cruise, click here to claim your prize now!!!",
    "Hi team, the sprint retro moved to Thursday 3pm in room 4B. Agenda attached.",
]


def show(email, scores):
    print(f"Email: {email}")
    for name, values in scores.items():
        rounded = numpy.round(values.astype(float), 3).tolist()
        print(f"  {name}:", dict(zip(CHOICES, rounded, strict=True)))
    winner = CHOICES[int(numpy.argmax(scores["Probabilities"]))]
    print(f"  Decision: {winner}\n")


def main():
    model = load_model()
    for email in sys.argv[1:] or EMAILS:
        show(email, classify(model, email, CHOICES))


if __name__ == "__main__":
    main()
