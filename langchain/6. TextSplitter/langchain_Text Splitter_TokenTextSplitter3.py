import tiktoken

# gpt-4o 등 최신 모델용 인코딩
encoding = tiktoken.get_encoding("o200k_base")

# --- 영어 문장 토큰화 ---
english_text = "I love LangChain"
english_tokens = encoding.encode(english_text)
print(f"--- 영어 ---")
print(f"문장: '{english_text}'")
print(f"토큰 수: {len(english_tokens)}")
print(f"토큰화 결과: {english_tokens}\n")

# --- 한글 문장 토큰화 ---
korean_text = "나는 랭체인을 좋아합니다"
korean_tokens = encoding.encode(korean_text)
print(f"--- 한글 ---")
print(f"문장: '{korean_text}'")
print(f"토큰 수: {len(korean_tokens)}")
print(f"토큰화 결과: {korean_tokens}")

# 한글 토큰이 어떻게 나뉘었는지 확인
print("\n--- 한글 토큰 분해 ---")
for token in korean_tokens:
    print(f"'{encoding.decode([token])}' ({token})")