import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

client = Groq(api_key=my_api_key)

model="llama-3.3-70b-versatile"
role="user"
prompt="Hello, LLM! How are you today?"
message={
    "role": role,
    "content": prompt
}
messages= [message] #we can have multiple messages in the list, but here we are just sending one message other can be sstem role msg, assistant role msg etc
response=client.chat.completions.create(model=model, messages=messages)
print(response) #this gives whole response with many things 

answer=response.choices[0].message.content #this give proper answer that we want by selecting choice[0] and then message.content
print(f"Answer from LLM: {answer}")

