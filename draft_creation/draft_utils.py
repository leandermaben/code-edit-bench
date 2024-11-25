import tiktoken

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
