from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import MultiQueryRetriever

load_dotenv()

# DB 설정
db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'testdb',
    'user': 'test',
    'password': '5748'
}

FILE_PATH = "8. VectorStore/data/SPRi AI Brief_10월호_산업동향_1002_F.pdf"

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
    collection_name="multi_query_example",
    connection=CONNECTION_STRING,
    pre_delete_collection=True,
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("=" * 80)
print("Multi-Query Retriever")
print("- LLM이 원본 질문을 여러 관점으로 변환")
print("- 각 질문으로 검색 후 결과 통합")
print("=" * 80)

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Multi-Query Retriever 생성
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_retriever,
    llm=llm
)

# 검색 실행
print("\n원본 질문: 생성형 AI의 최신 동향은?")
print("(LLM이 자동으로 여러 질문 생성 후 검색)\n")

results = multi_query_retriever.invoke("생성형 AI의 최신 동향은?")

# 결과 출력 (중복 제거된 상위 결과)
for idx, doc in enumerate(results[:3], 1):
    print(f"\n[결과 {idx}]")
    print(f"내용: {doc.page_content[:200]}...")
    print(f"출처: {doc.metadata.get('source', '알 수 없음')}")
    print(f"페이지: {doc.metadata.get('page', '알 수 없음')}")
    print("-" * 80)