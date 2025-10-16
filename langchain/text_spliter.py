from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data/SPRi AI Brief_10월호_산업동향_1002_F.pdf")
pages = loader.load()
print(f"PDF를 총 {len(pages)} 페이지로 분할하였습니다.\n")

# 2. 텍스트 분할기 초기화
# RecursiveCharacterTextSplitter를 사용하여 텍스트를 청크로 분할합니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
)

# 3. 문서 분할
# split_documents() 메서드는 텍스트 분할기와 문서 리스트를 입력받아 문서 분할을 수행합니다.
chunks = text_splitter.split_documents(pages)

# 4. 결과 확인
print(f"총 {len(chunks)} 개의 청크로 분할되었습니다.")
print("--- 첫 3개의 청크 미리보기 ---")
for i, chunk in enumerate(chunks[:3]):
    print(f"청크 {i+1}번째 : {chunk}")
    print(f"메타데이터 : {chunk.metadata}\n")
    print(f"문서 내용 : {chunk.page_content}\n")