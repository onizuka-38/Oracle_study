from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

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
    collection_name="self_query_example",
    connection=CONNECTION_STRING,
    pre_delete_collection=True,
)

print("=" * 80)
print("Self-Query Retriever")
print("- 자연어 질문을 자동으로 필터 조건으로 변환")
print("- 메타데이터 기반 검색")
print("=" * 80)

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 메타데이터 필드 정의
metadata_field_info = [
    AttributeInfo(
        name="page",
        description="문서의 페이지 번호",
        type="integer",
    ),
    AttributeInfo(
        name="source",
        description="문서의 출처 파일명",
        type="string",
    ),
]

document_content_description = "AI 기술 동향 보고서"

# Self-Query Retriever 생성
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True
)

# 검색 실행 (메타데이터 필터링 포함)
print("\n질문: 3페이지의 생성형 AI 관련 내용을 찾아줘")
print("(자동으로 page=3 필터 적용)\n")

results = self_query_retriever.invoke("3페이지의 생성형 AI 관련 내용을 찾아줘")

# 결과 출력
for idx, doc in enumerate(results, 1):
    print(f"\n[결과 {idx}]")
    print(f"내용: {doc.page_content[:200]}...")
    print(f"출처: {doc.metadata.get('source', '알 수 없음')}")
    print(f"페이지: {doc.metadata.get('page', '알 수 없음')}")
    print("-" * 80)