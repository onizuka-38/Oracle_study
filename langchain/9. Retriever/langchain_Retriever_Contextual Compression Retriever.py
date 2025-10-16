from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

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
    collection_name="compression_example",
    connection=CONNECTION_STRING,
    pre_delete_collection=True,
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("=" * 80)
print("Contextual Compression Retriever")
print("- LLM이 검색 결과에서 질문 관련 부분만 추출")
print("- 불필요한 내용 제거하여 효율성 향상")
print("=" * 80)

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Compressor 생성
compressor = LLMChainExtractor.from_llm(llm)

# Contextual Compression Retriever 생성
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_retriever
)

# 검색 실행
results = compression_retriever.invoke("생성형 AI의 기술 동향")

# 결과 출력 (압축된 내용)
for idx, doc in enumerate(results, 1):
    print(f"\n[결과 {idx}] - 압축된 내용")
    print(f"내용: {doc.page_content}")
    print(f"출처: {doc.metadata.get('source', '알 수 없음')}")
    print(f"페이지: {doc.metadata.get('page', '알 수 없음')}")
    print("-" * 80)