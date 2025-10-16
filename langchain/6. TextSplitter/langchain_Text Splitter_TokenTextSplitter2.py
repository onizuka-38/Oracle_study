import tiktoken

text = "This splitter is essential for managing LLM context windows and optimizing API costs by counting tokens."

# --- cl100k_base ---
encoding_cl100k = tiktoken.get_encoding("cl100k_base")
tokens_cl100k = encoding_cl100k.encode(text)

print(f"--- cl100k_base (총 {len(tokens_cl100k)}개 토큰) ---")
for token_id in tokens_cl100k:
    print(f"'{encoding_cl100k.decode([token_id])}'", end=" | ")

print("\n\n" + "="*30 + "\n")

# --- o200k_base  ---
encoding_o200k = tiktoken.get_encoding("o200k_base")
tokens_o200k = encoding_o200k.encode(text)

print(f"--- o200k_base (총 {len(tokens_o200k)}개 토큰) ---")
for token_id in tokens_o200k:
    print(f"'{encoding_o200k.decode([token_id])}'", end=" | ")