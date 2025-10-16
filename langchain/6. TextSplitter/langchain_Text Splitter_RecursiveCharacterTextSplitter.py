from langchain_text_splitters import RecursiveCharacterTextSplitter

# 분할할 샘플 텍스트
sample_text = """
LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다.

이 프레임워크의 핵심은 모듈화입니다. 이를 통해 개발자는 다양한 컴포넌트를 쉽게 조립할 수 있습니다.

주요 기능으로는 모델 I/O, Chains, RAG 등이 있습니다.
"""

# 1. 분할기 초기화
# chunk_size: 청크의 최대 크기
# chunk_overlap: 청크 간의 중복되는 문자 수
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

# 2. 텍스트 분할
chunks = text_splitter.split_text(sample_text)

# 3. 결과 확인
print(f"--- RecursiveCharacterTextSplitter 결과 (총 {len(chunks)}개) ---")
for i, chunk in enumerate(chunks):
    print(f"{i+1}번 청크:\n{chunk}\n")