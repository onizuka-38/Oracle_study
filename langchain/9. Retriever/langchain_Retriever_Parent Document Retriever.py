from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore, LocalFileStore
from langchain.storage import create_kv_docstore

load_dotenv()

# DB 설정
db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'testdb',
    'user': 'test',
    'password': '5748'
}

FILE_PATH = "../data/SPRi AI Brief_10월호_산업동향_1002_F.pdf"

# PDF 로드
loader = PyPDFLoader(FILE_PATH)
pages = loader.load()

# 부모 문서용 큰 청크
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 자식 문서용 작은 청크
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# Vector Store 생성
CONNECTION_STRING = PGVector.connection_string_from_db_params(driver="psycopg", **db_config)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 빈 vectorstore 생성
vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name="parent_doc_example",
    connection=CONNECTION_STRING,
)

# 부모 문서 저장소(휘발성, 종료 시 사라짐)
store = InMemoryStore()
# 파일로 저장(보존 가능)
# fs = LocalFileStore("9. Retriever/data/parent_data")
# store = create_kv_docstore(fs)

print("=" * 80)
print("Parent Document Retriever")
print("- 작은 청크(자식)로 검색하여 정확도 향상")
print("- 큰 문서(부모) 반환하여 맥락 제공")
print("=" * 80)

# Parent Document Retriever 생성
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 문서 추가
parent_retriever.add_documents(pages)
print("문서 추가 완료\n")

# 검색 실행
results = parent_retriever.invoke("생성형 AI")

# 결과 출력
for idx, doc in enumerate(results[:2], 1):
    print(f"\n[결과 {idx}] - 부모 문서 (전체 맥락)")
    print(f"내용: {doc.page_content[:300]}...")
    print(f"출처: {doc.metadata.get('source', '알 수 없음')}")
    print(f"페이지: {doc.metadata.get('page', '알 수 없음')}")
    print("-" * 80)