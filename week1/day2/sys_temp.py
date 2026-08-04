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
prompt="Hello, suggest a name for my food brand "

message_system={
    "role": "system",
    "content": "You are a professional brand manager who suggests name for my food brand, in one word names "  #system role used to customize our response from llm
}

message={
    "role": role,
    "content": prompt
}
messages= [message_system, message] #Here we are sending both system and user messages
response=client.chat.completions.create(model=model, messages=messages, temperature=1.9) #By default temp=0
#print(response) #this gives whole response with many things 

answer=response.choices[0].message.content #this give proper answer that we want by selecting choice[0] and then message.content
print(f"Answer from LLM: {answer}")

