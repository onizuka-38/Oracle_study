from langchain_huggingface import HuggingFaceEmbeddings

# 1. 임베딩 모델 초기화
# model_name: Hugging Face Hub에 있는 모델의 이름을 지정합니다.
# model_kwargs: 모델을 로드할 장치('cpu' 또는 'cuda')를 지정합니다.
# encode_kwargs: 임베딩 시 정규화(normalize) 여부를 설정합니다.
embeddings_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

# 2. 여러 문서 임베딩 (embed_documents)
documents = [
    "저는 강아지를 좋아합니다.",
    "저는 고양이를 좋아합니다.",
    "저는 축구를 좋아합니다.",
]
document_embeddings = embeddings_model.embed_documents(documents)

print("--- 여러 문서 임베딩 (embed_documents) 결과 ---")
print(f"총 {len(document_embeddings)}개의 문서가 임베딩되었습니다.")
print(f"첫 번째 문서의 벡터 차원(크기): {len(document_embeddings[0])}")
print(f"첫 번째 문서의 벡터 (앞 5개 값): {document_embeddings[0][:5]}\n")

# 3. 단일 텍스트(질의) 임베딩 (embed_query)
query = "제가 좋아하는 동물은 무엇인가요?"
query_embedding = embeddings_model.embed_query(query)

print("--- 단일 질의 임베딩 (embed_query) 결과 ---")
print(f"질의 벡터 차원(크기): {len(query_embedding)}")
print(f"질의 벡터 (앞 5개 값): {query_embedding[:5]}")