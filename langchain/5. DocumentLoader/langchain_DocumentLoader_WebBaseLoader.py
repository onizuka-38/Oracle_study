from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://n.news.naver.com/article/094/0000012995")
docs = loader.load()

print("--- WebBaseLoader 결과 ---")
print(f"메타데이터: {docs[0].metadata}\n")
print("--- 내용 ---")
print(docs[0].page_content[:1000])