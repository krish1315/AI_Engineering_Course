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

# 3 prompts for testing limit tokens 
ptompt1="Hello, LLM! How are you today?"
prompt2="What is the weather like in New York City today?"
prompt3="Can you provide a brief summary of the latest news in technology? in 1000 words"
prompts=[ptompt1, prompt2, prompt3]

for prompt in prompts:
    message={
        "role": role,
        "content": prompt
    }
    messages= [message] #we can have multiple messages in the list, but here we are just sending one message other can be sstem role msg, assistant role msg etc
    response=client.chat.completions.create(model=model, messages=messages, max_tokens=100) #Max tokens is the limit of tokens that can be used for the completion. If the completion exceeds this limit, it will be truncated.
    usage=response.usage
    #TO calculate the total tokens used, we can access the usage attribute of the response object. The usage attribute contains information about the number of tokens used for the prompt, completion, and total tokens.
    print(f"Prompt: {prompt}--> Completion Tokens Used: {usage.completion_tokens}, Prompt Tokens Used: {usage.prompt_tokens}, Total Tokens Used: {usage.total_tokens} Finish reason: {response.choices[0].finish_reason}")
    #finsih reason gives the reason for the completion to stop. It can be "stop" if the model stopped generating text because it reached a stopping point, "length" if it stopped because it reached the maximum token limit, or "error" if there was an error during generation. 
