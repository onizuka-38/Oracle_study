from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

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
    collection_name="ensemble_example",
    connection=CONNECTION_STRING,
    pre_delete_collection=True,
)

# Vector Retriever 생성
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# BM25 Retriever 생성
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

print("=" * 80)
print("Ensemble Retriever (Hybrid Search)")
print("- Sparse(BM25) + Dense(Vector) 결합")
print("- 키워드와 의미 모두 고려")
print("=" * 80)

# Ensemble Retriever 생성
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # BM25 50%, Vector 50%
)

# 검색 실행
results = ensemble_retriever.invoke("생성형 AI 기술 동향")

# 결과 출력
for idx, doc in enumerate(results, 1):
    print(f"\n[결과 {idx}]")
    print(f"내용: {doc.page_content[:200]}...")
    print(f"출처: {doc.metadata.get('source', '알 수 없음')}")
    print(f"페이지: {doc.metadata.get('page', '알 수 없음')}")
    print("-" * 80)