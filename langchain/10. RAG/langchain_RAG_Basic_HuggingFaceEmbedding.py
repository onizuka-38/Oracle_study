import psycopg
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
print("=" * 80)
print("Langchain RAG 8단계 프로세스 (HuggingFace 임베딩)")
print("=" * 80)

# ============================================================
# 1단계: 문서 로드 (Document Loader)
# - 외부 데이터 소스에서 문서를 로드
# ============================================================
print("\n[1단계] 문서 로드 (Document Loader)")
print("-" * 80)

FILE_PATH = "../data/SPRi AI Brief_10월호_산업동향_1002_F.pdf"
loader = PyPDFLoader(FILE_PATH)

# PDF 파일을 페이지별로 로드
documents = loader.load()

print(f"총 {len(documents)} 페이지 로드 완료")
print(f"첫 페이지 미리보기: {documents[0].page_content[:100]}...")

# ============================================================
# 2단계: 텍스트 분할 (Text Splitter)
# - 로드된 문서를 처리 가능한 작은 단위(청크)로 분할
# ============================================================
print("\n[2단계] 텍스트 분할 (Text Splitter)")
print("-" * 80)

# 1000자 단위로 분할, 100자 중복(오버랩)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 각 청크의 크기
    chunk_overlap=100,    # 청크 간 중복되는 부분
    length_function=len,  # 길이 측정 함수
)

# 문서를 청크로 분할
splits = text_splitter.split_documents(documents)

print(f"총 {len(splits)}개의 청크로 분할 완료")
print(f"첫 번째 청크: {splits[0].page_content[:150]}...")

# ============================================================
# 3단계: 임베딩 (Embedding)
# - 각 텍스트 청크를 벡터로 변환
# ============================================================
print("\n[3단계] 임베딩 (Embedding)")
print("-" * 80)

# HuggingFace 한국어 임베딩 모델 초기화
embeddings_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

# 테스트: 샘플 텍스트를 벡터로 변환
sample_text = "생성형 AI 기술 동향"
sample_vector = embeddings_model.embed_query(sample_text)

print(f"임베딩 모델 준비 완료")
print(f"모델: jhgan/ko-sroberta-multitask (한국어 특화)")
print(f"벡터 차원: {len(sample_vector)}차원")
print(f"샘플 벡터 (처음 5개): {sample_vector[:5]}")

# ============================================================
# 4단계: 벡터스토어 저장 (Vector Store)
# - 임베딩된 벡터들을 데이터베이스에 저장
# ============================================================
print("\n[4단계] 벡터스토어 저장 (Vector Store)")
print("-" * 80)

# PostgreSQL 연결 설정
db_config = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'testdb',  # psycopg는 dbname 사용
    'user': 'test',
    'password': '5748'
}

# pgvector extension 확인
# PostgreSQL에서 벡터 연산을 가능하게 하는 확장 기능
conn = None
try:
    conn = psycopg.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    cursor.close()
    print("pgvector extension 확인 완료")
except Exception as e:
    print(f"Extension 생성 중 에러: {e}")
finally:
    if conn:
        conn.close()

# PGVector 연결 문자열 직접 작성
CONNECTION_STRING = f"postgresql+psycopg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"

# PGVector로 벡터 저장
vectorstore = PGVector.from_documents(
    documents=splits,                          # 분할된 문서들
    embedding=embeddings_model,                # HuggingFace 임베딩 모델
    collection_name="rag_example_huggingface", # 새로운 컬렉션 이름
    connection=CONNECTION_STRING,              # DB 연결
    pre_delete_collection=True,                # 기존 데이터 삭제
)

print(f"벡터스토어에 {len(splits)}개 청크 저장 완료")
print(f"컬렉션 이름: rag_example_huggingface")
print(f"벡터 차원: {len(sample_vector)}차원 (OpenAI 1536차원과 다름)")

# ============================================================
# 5단계: 검색기 (Retriever)
# - 질문과 관련된 문서를 벡터 데이터베이스에서 검색
# ============================================================
print("\n[5단계] 검색기 (Retriever)")
print("-" * 80)

# Retriever 생성
# search_type="similarity": 유사도 기반 검색
# search_kwargs={"k": 3}: 상위 3개 결과 반환
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 테스트 검색
test_query = "생성형 AI의 최신 기술 동향은?"
retrieved_docs = retriever.invoke(test_query)

print(f"검색기 생성 완료")
print(f"테스트 쿼리: '{test_query}'")
print(f"검색된 문서: {len(retrieved_docs)}개")
print(f"첫 번째 결과: {retrieved_docs[0].page_content[:100]}...")

# ============================================================
# 6단계: 프롬프트 (Prompt)
# - 검색된 정보를 바탕으로 LLM을 위한 질문 구성
# ============================================================
print("\n[6단계] 프롬프트 (Prompt)")
print("-" * 80)

# RAG 프롬프트 템플릿 구성 (개선된 버전)
# {context}: 검색된 문서들
# {question}: 사용자 질문
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """당신은 AI 기술 전문가입니다.
아래 제공된 문맥(context)을 **반드시 참고**하여 질문에 답변해주세요.

중요:
1. 문맥에 관련 내용이 있으면 그것을 바탕으로 답변하세요.
2. 문맥에 없는 내용만 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.
3. 답변 시 문맥의 어느 부분을 참고했는지 명시하세요.

문맥:
{context}
"""),
    ("human", "{question}")
])

print("프롬프트 템플릿 생성 완료")
print(f"시스템 역할: AI 기술 전문가")
print(f"프롬프트 구조: 문맥(context) + 질문(question)")

# ============================================================
# 7단계: LLM (Large Language Model)
# - 구성된 프롬프트를 사용하여 답변 생성
# ============================================================
print("\n[7단계] LLM (Large Language Model)")
print("-" * 80)

# ChatGPT 모델 초기화
llm = ChatOpenAI(
    model="gpt-4o-mini",  # 모델 선택
    temperature=0,         # 창의성 조절 (0=결정적, 1=창의적)
)

print("LLM 모델 준비 완료")
print(f"모델: gpt-4o-mini")

# ============================================================
# 8단계: 체인(Chain) 생성
# - 모든 과정을 하나의 파이프라인으로 연결
# ============================================================
print("\n[8단계] 체인(Chain) 생성")
print("-" * 80)

# 문서를 문맥으로 포맷팅하는 함수 (개선된 버전)
def format_docs(docs):
    """
    검색된 문서들을 번호와 구분선으로 포맷팅
    - [문서 1], [문서 2] 형식으로 번호 부여
    - --- 구분선으로 명확하게 분리
    """
    formatted = []
    for idx, doc in enumerate(docs, 1):
        formatted.append(f"[문서 {idx}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# RAG Chain 구성
# RunnablePassthrough(): 입력을 그대로 전달
# retriever: 질문으로 문서 검색
# format_docs: 검색된 문서를 문자열로 변환
# prompt_template: 프롬프트 생성
# llm: 답변 생성
# StrOutputParser(): 출력을 문자열로 파싱
rag_chain = (
    {
        "context": retriever | format_docs,  # 검색 → 포맷팅
        "question": RunnablePassthrough()     # 질문 그대로 전달
    }
    | prompt_template   # 프롬프트 생성
    | llm               # LLM 실행
    | StrOutputParser() # 출력 파싱
)

print("RAG Chain 생성 완료")
print("""
Chain 구조:
   질문 입력
      ↓
   [Retriever] 관련 문서 검색 (5단계)
      ↓
   [Format] 문서를 문맥으로 변환
      ↓
   [Prompt] 프롬프트 생성 (6단계)
      ↓
   [LLM] 답변 생성 (7단계)
      ↓
   최종 답변 출력
""")

# ============================================================
# RAG 실행 테스트
# ============================================================
print("\n" + "=" * 80)
print("RAG 시스템 실행 테스트")
print("=" * 80)

# 질문 목록
questions = [
    "중국 국무원이 발표한 'AI 플러스' 정책의 3단계 중장기 목표는 무엇이며, 6대 핵심 영역은 어디인가요?",
    "구글이 공개한 이미지 편집 모델 '제미나이 2.5 플래시 이미지'의 가장 큰 특징은 무엇인가요?",
    "스탠포드 대학의 연구 결과, 생성 AI의 확산이 경력 초기 근로자의 고용에 어떤 영향을 미치고 있나요? "
]

for idx, question in enumerate(questions, 1):
    print(f"\n[질문 {idx}] {question}")
    print("-" * 80)
    
    # RAG Chain 실행
    answer = rag_chain.invoke(question)
    
    print(f"답변:\n{answer}")
    print("-" * 80)

# ============================================================
# 대화형 RAG 시스템
# ============================================================
print("\n대화형 RAG 시스템 시작")
print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
print("=" * 80)

while True:
    user_question = input("\n질문을 입력하세요: ").strip()
    
    if user_question.lower() in ['quit', 'exit', '종료']:
        print("\nRAG 시스템을 종료합니다.")
        break
    
    if not user_question:
        print("질문을 입력해주세요.")
        continue
    
    try:
        print("\n문서 검색 중...")
        answer = rag_chain.invoke(user_question)
        print(f"\n답변:\n{answer}")
        print("-" * 80)
    except Exception as e:
        print(f"\n에러 발생: {e}")