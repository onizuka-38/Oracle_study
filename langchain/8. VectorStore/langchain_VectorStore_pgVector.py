from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

# 1. 연결 정보를 딕셔너리로 정의
db_config = {
    'host': '127.0.0.1',
    'port': 5432,
    'database': 'testdb', 
    'user': 'test',          
    'password': '5748'   
}

# 2. LangChain용 연결 문자열을 다시 생성합니다.
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg",
    host=db_config["host"],
    port=db_config["port"],
    database=db_config["database"],
    user=db_config["user"],
    password=db_config["password"],
)

# 'vector_content'라는 이름의 테이블을 사용합니다.
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

print("문서 추가 및 벡터 저장소 생성을 완료했습니다.")

retrieved_docs = db.similarity_search("테스트")

print("\n--- 검색 결과 ---")
print(retrieved_docs[0].page_content)