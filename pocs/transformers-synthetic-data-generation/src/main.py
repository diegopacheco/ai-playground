import json
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

def generate_synthetic_field_data(prompt, max_length=100):
    # Initialize the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate a sequence of text
    outputs = model.generate(inputs, max_length=max_length, temperature=0.7, do_sample=True)

    # Decode the output
    synthetic_data = tokenizer.decode(outputs[0])

    return synthetic_data

def generate_ssn():
    """Generate a random, but valid, US Social Security number."""
    while True:
        area_number = random.randint(1, 899)
        if area_number == 666:
            continue
        group_number = random.randint(1, 99)
        serial_number = random.randint(1, 9999)
        ssn = "{:03d}-{:02d}-{:04d}".format(area_number, group_number, serial_number)
        return ssn

def generate():
    # Generate synthetic test data
    synthetic_test_data = {
        "name": generate_synthetic_field_data("Name", max_length=10),
        "phone": generate_synthetic_field_data("US Phone Number", max_length=10),
        "email": generate_synthetic_field_data("Email Address", max_length=30),
        "ssn": generate_ssn()
    }
    return synthetic_test_data

print(json.dumps(generate(), indent=4))
print(json.dumps(generate(), indent=4))
print(json.dumps(generate(), indent=4))