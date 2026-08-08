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


# For structure output we can use pydantic to define the structure of the output we want from the model.
from pydantic import BaseModel
class Ticket(BaseModel):
    name: str
    email:str | None = None #added none cuz if no email it becomes null and in python it is none so we need to handle that case
    issue:str

schema=Ticket.model_json_schema()
response_format ={
    "type": "json_object"
}

system_prompt=f'''
      Extract the personal information from the ticket somplain strictly based on this schema and give a json output
     {schema}
      '''

message_system={
    "role": "system",
    "content": system_prompt
}


text = "Hello, I am krish. I have an iphone and my ticket purchased is failed and my father is MLA and I want to get my ticket purchased. Can you help me with this?" #there can be anything in a complaint literallyy anything but our output will only fetch what we reuqired
prompt=f'''
  This is a user complaint only fetch the personal information from this {text}
'''
prompts=[prompt]


message={
        "role": role,
        "content": prompt
    }
messages= [message_system, message] 
response=client.chat.completions.create(model=model, messages=messages, response_format=response_format) #we pass our own response format to the model so that it can return the output in the format we want.

answer  = response.choices[0].message.content
print(answer)


#How to read JSon outpu
import json
raw_json=answer
data_file=json.loads(raw_json)
ticket=Ticket(**data_file)

print(ticket.name)
print(ticket.email)
print(ticket.issue)