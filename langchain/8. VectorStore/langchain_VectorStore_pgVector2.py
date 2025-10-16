from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

# 1. 제공된 정보를 바탕으로 표준 연결 문자열을 직접 작성
CONNECTION_STRING = "postgresql+psycopg://test:5748@localhost:5432/testdb"

COLLECTION_NAME = "vector_content"

# --- 임베딩 모델 및 샘플 문서 준비 ---
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
documents = [Document(page_content="테스트 데이터로 생성된 문서입니다.")]

# --- 벡터 저장소 생성 및 검색 ---
db = PGVector.from_documents(
    embedding=embeddings_model,
    documents=documents,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
)

retrieved_docs = db.similarity_search("테스트")

print("\n--- 검색 결과 ---")
print(retrieved_docs[0].page_content)