from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="5. DocumentLoader/data/Sample_CSV.csv", encoding="utf-8")
rows = loader.load()

print("--- CSVLoader 결과 (첫 번째 행) ---")
print(f"메타데이터: {rows[0].metadata}\n")
print("--- 내용 ---")
print(rows[0].page_content)