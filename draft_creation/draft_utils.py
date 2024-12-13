import tiktoken
import time
from openai import OpenAI

def clean_text(text):
    # Remove or replace surrogate characters
    return text.encode('utf-8', 'ignore').decode('utf-8')

def count_tokens(text, max_chunk_size=10000):
    
    if not text:
        return 0

    encoding = tiktoken.get_encoding("o200k_base")
    total_tokens = 0
        
    # Process text in chunks
    for i in range(0, len(text), max_chunk_size):
        chunk = text[i:i + max_chunk_size]
        try:
            tokens = encoding.encode(clean_text(chunk), disallowed_special=())
            total_tokens += len(tokens)
        except Exception as e:
            print(f"Warning: Error processing chunk {i}: {e}")
            # Optional: use a simpler fallback method
            total_tokens += len(chunk.split())  # rough approximation
            
    return total_tokens

def wait_for_server(model):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Loop until the server is up
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'Hello'"}],
                max_tokens=10
            )
            print("Server is up and running!")
            print("Response:", response.choices[0].message.content)
            break  # Exit the loop if the server is up
        except Exception as e:
            print("Server is not responding, retrying in 5 seconds...")
            time.sleep(20)  # Wait for 5 seconds before retrying