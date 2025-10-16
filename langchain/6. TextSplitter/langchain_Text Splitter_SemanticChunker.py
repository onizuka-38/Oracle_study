# pip install langchain-experimental 설치가 필요
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

sample_text = "태양은 별입니다. 달은 지구 주위를 돕니다. 화성은 태양계의 네 번째 행성입니다. 목성은 가장 큰 행성입니다."

# embeddings: 의미적 유사도를 계산하는 데 사용할 임베딩 모델
text_splitter = SemanticChunker(OpenAIEmbeddings())

chunks = text_splitter.split_text(sample_text)

print(f"--- SemanticChunker (총 {len(chunks)}개) ---")
for i, chunk in enumerate(chunks):
    print(f"[{i+1}] {chunk}")