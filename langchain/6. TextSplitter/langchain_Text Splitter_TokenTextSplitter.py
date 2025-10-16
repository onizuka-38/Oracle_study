import tiktoken
from langchain_text_splitters import TokenTextSplitter

# --- 토큰 수 계산을 위한 헬퍼 함수 ---
# 최신 모델의 토큰 계산을 위한 인코딩 방식
encoding = tiktoken.get_encoding("o200k_base")

def num_tokens_from_string(string: str) -> int:
    """문자열의 토큰 수를 계산하여 반환합니다."""
    return len(encoding.encode(string))
# ------------------------------------

# 분할할 샘플 텍스트
sample_text = "This splitter is essential for managing LLM context windows and optimizing API costs by counting tokens."

# 원본 텍스트의 총 토큰 수 확인
total_tokens = num_tokens_from_string(sample_text)
print(f"원본 텍스트의 총 토큰 수: {total_tokens}\n")


# 1. TokenTextSplitter 초기화
text_splitter = TokenTextSplitter(
    chunk_size=15,       # 청크의 최대 크기를 15개 '토큰'으로 지정
    chunk_overlap=5,     # 청크 간 5개 '토큰'을 겹치게 설정
    encoding_name="o200k_base" # 토큰 계산에 사용할 인코딩 이름
)

# 2. 텍스트 분할
chunks = text_splitter.split_text(sample_text)

# 3. 결과 확인
print(f"--- TokenTextSplitter 결과 (총 {len(chunks)}개) ---")
for i, chunk in enumerate(chunks):
    # 각 청크의 토큰 수를 계산
    chunk_tokens = num_tokens_from_string(chunk)
    print(f"[{i+1}] (토큰: {chunk_tokens}) {chunk}")