from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# 1. OpenAI API 키 로드
load_dotenv()

# 2. 임베딩 모델 초기화
# OpenAI의 text-embedding-3-small 모델을 사용합니다.
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. 여러 문서 임베딩 (embed_documents)
# RAG에서 문서를 벡터 DB에 저장할 때 사용됩니다.
documents = [
    "저는 강아지를 좋아합니다.",
    "저는 고양이를 좋아합니다.",
    "저는 축구를 좋아합니다.",
]
document_embeddings = embeddings_model.embed_documents(documents)

print("--- 여러 문서 임베딩 (embed_documents) 결과 ---")
print(f"총 {len(document_embeddings)}개의 문서가 임베딩되었습니다.")
print(f"첫 번째 문서의 벡터 차원(크기): {len(document_embeddings[0])}")
print(f"첫 번째 문서의 벡터 (앞 5개 값): {document_embeddings}\n")


# 4. 단일 텍스트(질의) 임베딩 (embed_query)
# RAG에서 사용자 질문을 벡터로 변환할 때 사용됩니다.
query = "제가 좋아하는 동물은 무엇인가요?"
query_embedding = embeddings_model.embed_query(query)

print("--- 단일 질의 임베딩 (embed_query) 결과 ---")
print(f"질의 벡터 차원(크기): {len(query_embedding)}")
print(f"질의 벡터 (앞 5개 값): {query_embedding}")