from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

print("Welcome to the AI Adventure Game! You can type actions or dialogue. For example, 'look around' or 'say Hello'.")
print("Let's start your adventure!")

game_state = "You are standing in a small village. There's a shop to your right and an old man sitting on a bench in front of you."

while True:
    action = input("> ")

    # Concatenate the game state and the user's action and generate the AI's response
    input_ids = tokenizer.encode(game_state + action, return_tensors='pt')
    output = model.generate(input_ids, max_length=1000, temperature=0.7, pad_token_id=tokenizer.eos_token_id)

    # Decode the output and update the game state
    game_state = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(game_state)
