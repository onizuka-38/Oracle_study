from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

load_dotenv()

FILE_PATH = "../data/SPRi AI Brief_10월호_산업동향_1002_F.pdf"

# PDF 로드 및 분할
loader = PyPDFLoader(FILE_PATH)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(pages)

print("=" * 80)
print("BM25 Retriever (Sparse Retriever)")
print("- 키워드 기반 검색 (TF-IDF 방식)")
print("=" * 80)

# BM25 Retriever 생성
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3  # 상위 3개 결과 반환

# 검색 실행
results = bm25_retriever.invoke("생성형 AI")

# 결과 출력
for idx, doc in enumerate(results, 1):
    print(f"\n[결과 {idx}]")
    print(f"내용: {doc.page_content[:200]}...")
    print(f"출처: {doc.metadata.get('source', '알 수 없음')}")
    print(f"페이지: {doc.metadata.get('page', '알 수 없음')}")
    print("-" * 80)