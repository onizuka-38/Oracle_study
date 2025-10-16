from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. PDF 문서 로드
# PyPDFLoader는 PDF를 페이지별로 나누어 Document 객체 리스트로 만듭니다.
loader = PyPDFLoader(file_path="6. TextSplitter/data/SPRi AI Brief_10월호_산업동향_1002_F.pdf")
pages = loader.load()
print(f"PDF를 총 {len(pages)}개의 페이지로 불러왔습니다.\n")

# 2. 텍스트 분할기 초기화
# RecursiveCharacterTextSplitter를 사용하여 텍스트를 청크로 나눕니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # 청크의 최대 크기 (문자 수)
    chunk_overlap=50,   # 청크 간 겹치는 문자 수
)

# 3. 문서 분할
# split_documents() 메서드는 Document 객체 리스트를 받아 분할합니다.
chunks = text_splitter.split_documents(pages)

# 4. 결과 확인
print(f"총 {len(chunks)}개의 청크로 분할되었습니다.\n")
print("--- 첫 3개 청크 미리보기 ---")
for i, chunk in enumerate(chunks[:3]):
    print(f"--- 청크 {i+1} ---")
    print(f"메타데이터: {chunk.metadata}")
    print(f"내용: {chunk.page_content}\n")