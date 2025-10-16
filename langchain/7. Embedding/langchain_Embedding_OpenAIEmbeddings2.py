from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# 1. OpenAI API 키 로드
load_dotenv()

# 2. PDF 파일 로드
file_path = "7. Embedding/data/SPRi AI Brief_10월호_산업동향_1002_F.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()

print(f"--- PDF 문서 로드 완료 ---")
print(f"총 {len(documents)}개의 페이지가 로드되었습니다.\n")

# 3. 임베딩 모델 초기화
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 4. 문서 내용만 추출 (각 페이지의 텍스트)
document_texts = [doc.page_content for doc in documents]

# 5. 여러 문서 임베딩
document_embeddings = embeddings_model.embed_documents(document_texts)

print("--- 여러 문서 임베딩 (embed_documents) 결과 ---")
print(f"총 {len(document_embeddings)}개의 페이지가 임베딩되었습니다.")
print(f"첫 번째 페이지의 벡터 차원(크기): {len(document_embeddings[0])}")
print(f"첫 번째 페이지의 벡터 (앞 5개 값): {document_embeddings[0][:5]}\n")

# 6. 첫 번째 페이지 내용 미리보기
print("--- 첫 번째 페이지 내용 미리보기 ---")
print(document_texts[0][:500])  # 앞 500자만 출력
print("...\n")

# 7. 단일 텍스트(질의) 임베딩
query = "AI 산업 동향에 대해 알려주세요"
query_embedding = embeddings_model.embed_query(query)

print("--- 단일 질의 임베딩 (embed_query) 결과 ---")
print(f"질의 벡터 차원(크기): {len(query_embedding)}")
print(f"질의 벡터 (앞 5개 값): {query_embedding[:5]}")