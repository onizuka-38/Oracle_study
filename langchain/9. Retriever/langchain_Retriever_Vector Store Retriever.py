from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# DB 설정
db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'testdb',
    'user': 'test',
    'password': '5748'
}

FILE_PATH = "9. Retriever/data/SPRi AI Brief_10월호_산업동향_1002_F.pdf"

# PDF 로드 및 분할
loader = PyPDFLoader(FILE_PATH)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(pages)

# Vector Store 생성
CONNECTION_STRING = PGVector.connection_string_from_db_params(driver="psycopg", **db_config)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PGVector.from_documents(
    documents=documents,
    embedding=embeddings_model,
    collection_name="vector_store_example",
    connection=CONNECTION_STRING,
    pre_delete_collection=True,
)

print("=" * 80)
print("Vector Store Retriever (Dense Retriever)")
print("- 의미적 유사도 기반 검색")
print("=" * 80)

# Vector Store Retriever 생성
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 검색 실행
results = vector_retriever.invoke("생성형 AI의 기술 동향 알려줘")

# 결과 출력
for idx, doc in enumerate(results, 1):
    print(f"\n[결과 {idx}]")
    print(f"내용: {doc.page_content[:200]}...")
    print(f"출처: {doc.metadata.get('source', '알 수 없음')}")
    print(f"페이지: {doc.metadata.get('page', '알 수 없음')}")
    print("-" * 80)