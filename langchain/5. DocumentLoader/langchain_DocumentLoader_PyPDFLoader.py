from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("5. DocumentLoader/data/SPRi AI Brief_10월호_산업동향_1002_F.pdf")
pages = loader.load()

print("--- PyPDFLoader 결과 (첫 페이지) ---")
print(f"메타데이터: {pages[0].metadata}\n")
print("--- 내용 ---")
print(pages[10].page_content)