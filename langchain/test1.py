from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/SPRi AI Brief_10월호_산업동향_1002_F.pdf")
rows = loader.load()

print("---PDFLoader 결과 (첫 번째 행)---")
print(f"메타데이터 : {rows[0].metadata}\n")
print(f"문서 내용 : {rows[0].page_content}\n")
