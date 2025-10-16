import psycopg
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. 환경 변수 및 DB 설정 ---
load_dotenv()

db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'testdb',
    'user': 'test',
    'password': '5748'
}

COLLECTION_NAME = "test_db"
FILE_PATH = "8. VectorStore/data/SPRi AI Brief_10월호_산업동향_1002_F.pdf"

# --- 2. psycopg를 사용하여 vector extension 설치 ---
conn = None

try:
    conn = psycopg.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    print("vector extension을 성공적으로 확인/생성했습니다.")
    cursor.close()
    
except Exception as e:
    print(f"Extension 생성 중 에러 발생: {e}")
    
finally:
    if conn is not None:
        conn.close()

# --- 3. PDF 로드 및 텍스트 분할 ---
loader = PyPDFLoader(FILE_PATH)
pages = loader.load()
print(f"PDF 파일을 총 {len(pages)} 페이지로 불러왔습니다.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents_to_add = text_splitter.split_documents(pages)
print(f"문서를 총 {len(documents_to_add)}개의 청크로 분할했습니다.")

# --- 4. LangChain PGVector 준비 ---
CONNECTION_STRING = PGVector.connection_string_from_db_params(driver="psycopg", **db_config)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 5. PGVector로 문서 추가 ---
db = PGVector.from_documents(
    documents=documents_to_add,
    embedding=embeddings_model,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    pre_delete_collection=True,
)

print(f"'{COLLECTION_NAME}' 컬렉션에 문서를 추가 완료했습니다.")

# --- 6. 유사도 검색 ---
retrieved_docs = db.similarity_search("생성형 AI의 기술 동향에 대해 알려줘", k=3) # k는 반환할 문서 수(상위 k개) 

print("\n--- 검색 결과 ---")

if retrieved_docs:
    # 인덱스 번호 출력
    for idx, doc in enumerate(retrieved_docs, 1):
        print(f"\n[결과 {idx}]")
        print(f"내용: {doc.page_content}...") 
        print(f"출처: {doc.metadata.get('source', '알 수 없음')}")
        print(f"페이지: {doc.metadata.get('page', '알 수 없음')}")
        print("-" * 80) # 구분선
    
else:
    print("관련 문서를 찾지 못했습니다.")