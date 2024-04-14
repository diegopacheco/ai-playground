import openai 
import os  

openai.my_api_key = str(os.getenv("OPENAI_API_KEY"))
messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]

while True: 
    message = input("User : ") 
    if message: 
        messages.append( 
            {"role": "user", "content": message}, 
        ) 
        chat = openai.ChatCompletion.create( 
            model="gpt-3.5-turbo", messages=messages 
        ) 
      
    reply = chat.choices[0].message.content 
    print(f"ChatGPT: {reply}") 
    messages.append({"role": "assistant", "content": reply})