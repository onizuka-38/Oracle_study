from langchain_community.document_loaders import TextLoader

loader = TextLoader("5. DocumentLoader/data/Sample_Text.txt", encoding="utf-8")
docs = loader.load()

print("--- TextLoader 결과 ---")
print(f"메타데이터: {docs[0].metadata}\n")
print("--- 내용 ---")
print(docs[0].page_content)