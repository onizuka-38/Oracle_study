import psycopg
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from operator import itemgetter

load_dotenv()

print("=" * 80)
print("Langchain RAG 8단계 프로세스 - 멀티턴 대화")
print("=" * 80)

# ============================================================
# 1단계: 문서 로드 (Document Loader)
# - 외부 데이터 소스에서 문서를 로드
# ============================================================
print("\n[1단계] 문서 로드 (Document Loader)")
print("-" * 80)

FILE_PATH = "9. Retriever/data/SPRi AI Brief_10월호_산업동향_1002_F.pdf"
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

# OpenAI 임베딩 모델 초기화
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 테스트: 샘플 텍스트를 벡터로 변환
sample_text = "생성형 AI 기술 동향"
sample_vector = embeddings_model.embed_query(sample_text)

print(f"임베딩 모델 준비 완료")
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
    'dbname': 'testdb',
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
    documents=splits,                    # 분할된 문서들
    embedding=embeddings_model,          # 임베딩 모델
    collection_name="rag_example",       # 컬렉션 이름
    connection=CONNECTION_STRING,        # DB 연결
    pre_delete_collection=True,          # 기존 데이터 삭제
)

print(f"벡터스토어에 {len(splits)}개 청크 저장 완료")
print(f"컬렉션 이름: rag_example")

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
# 6단계: 프롬프트 (Prompt) - 멀티턴 대화용
# - 검색된 정보와 대화 히스토리를 바탕으로 LLM을 위한 질문 구성
# ============================================================
print("\n[6단계] 프롬프트 (Prompt) - 멀티턴")
print("-" * 80)

# 멀티턴 RAG 프롬프트 템플릿 구성
# {context}: 검색된 문서들
# {chat_history}: 이전 대화 내역
# {question}: 현재 사용자 질문
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """당신은 AI 기술 전문가입니다.
아래 제공된 문맥(context)과 이전 대화 내역을 **반드시 참고**하여 질문에 답변해주세요.

중요:
1. 이전 대화 내용을 기억하고 연관지어 답변하세요.
2. 문맥에 관련 내용이 있으면 그것을 바탕으로 답변하세요.
3. 문맥에 없는 내용만 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.
4. 답변 시 문맥의 어느 부분을 참고했는지 명시하세요.
5. "그것", "그거", "저것" 등의 대명사는 이전 대화를 참고하여 구체적으로 해석하세요.

문맥:
{context}
"""),
    MessagesPlaceholder(variable_name="chat_history"),  # 대화 히스토리
    ("human", "{question}")
])

print("멀티턴 프롬프트 템플릿 생성 완료")
print(f"시스템 역할: AI 기술 전문가 (대화 기억 기능)")
print(f"프롬프트 구조: 문맥(context) + 대화 히스토리(chat_history) + 질문(question)")

# ============================================================
# 7단계: LLM (Large Language Model)
# - 구성된 프롬프트를 사용하여 답변 생성
# ============================================================
print("\n[7단계] LLM (Large Language Model)")
print("-" * 80)

# ChatGPT 모델 초기화
llm = ChatOpenAI(
    model="gpt-4o-mini",  # 모델 선택
    temperature=0.5,      # 창의성 조절 (0=결정적, 1=창의적)
)

print("LLM 모델 준비 완료")
print(f"모델: gpt-4o-mini")

# ============================================================
# 8단계: 체인(Chain) 생성 - 멀티턴 대화용
# - 모든 과정을 하나의 파이프라인으로 연결
# - 대화 히스토리 관리 기능 추가
# ============================================================
print("\n[8단계] 체인(Chain) 생성 - 멀티턴")
print("-" * 80)

# 문서를 문맥으로 포맷팅하는 함수
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

# RAG Chain 구성 (멀티턴 지원)
# itemgetter = 딕셔너리나 리스트에서 특정 키/인덱스의 값을 추출하는 함수
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history")  # 대화 히스토리 전달
    }
    | prompt_template   # 프롬프트 생성
    | llm               # LLM 실행
    | StrOutputParser() # 출력 파싱
)

# ============================================================
# 대화 히스토리 관리
# - 세션별로 대화 내역을 저장하고 관리
# ============================================================

# 세션별 대화 히스토리 저장소
# 딕셔너리로 여러 사용자/세션의 대화를 분리하여 관리
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    세션 ID로 대화 히스토리를 가져오거나 새로 생성
    
    Args:
        session_id: 세션 구분 ID (예: "user123")
    
    Returns:
        해당 세션의 대화 히스토리 객체
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# RunnableWithMessageHistory로 대화 히스토리 관리 기능 추가
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,                          # 기본 RAG 체인
    get_session_history,                # 세션 히스토리 조회 함수
    input_messages_key="question",      # 사용자 입력 키
    history_messages_key="chat_history" # 히스토리 키 (프롬프트의 MessagesPlaceholder와 일치)
)

print("멀티턴 RAG Chain 생성 완료")
print("""
📊 Chain 구조 (멀티턴):
   질문 입력
      ↓
   [대화 히스토리 로드] 이전 대화 내용 불러오기
      ↓
   [Retriever] 관련 문서 검색 (5단계)
      ↓
   [Format] 문서를 문맥으로 변환
      ↓
   [Prompt] 프롬프트 생성 (문맥 + 히스토리 + 질문)
      ↓
   [LLM] 답변 생성 (7단계)
      ↓
   [대화 히스토리 저장] 질문과 답변 저장
      ↓
   최종 답변 출력
""")

# ============================================================
# RAG 실행 테스트 - 멀티턴 대화
# ============================================================
print("\n" + "=" * 80)
print("RAG 시스템 실행 테스트 - 멀티턴 대화")
print("=" * 80)

# 세션 ID (사용자 구분용)
session_id = "user_test_001"

# 연속된 질문들 (문맥 유지 테스트)
conversation_questions = [
    "중국 국무원이 발표한 'AI 플러스' 정책의 3단계 중장기 목표는 무엇이며, 6대 핵심 영역은 어디인가요?",
    "그 정책에서 가장 중요하게 다루는 산업 분야는 무엇인가요?",
    "해당 정책의 예상 경제 효과나 목표 수치가 있나요?",
    "구글이 공개한 이미지 편집 모델 '제미나이 2.5 플래시 이미지'의 가장 큰 특징은 무엇인가요?",
    "이 모델은 어떤 기술을 기반으로 만들어졌나요?",
    "기존 이미지 편집 모델과 비교했을 때 어떤 장점이 있나요?",
    "스탠포드 대학의 연구 결과, 생성 AI의 확산이 경력 초기 근로자의 고용에 어떤 영향을 미치고 있나요?",
    "구체적으로 어떤 직종이나 직무에서 가장 큰 영향을 받고 있나요?",
    "이러한 영향에 대한 해결책이나 대응 방안이 제시되었나요?",
]

print(f"\n세션 ID: {session_id}")
print(f"연속된 대화를 통해 문맥 유지 테스트\n")

for idx, question in enumerate(conversation_questions, 1):
    print(f"\n{'=' * 80}")
    print(f"[턴 {idx}] 사용자: {question}")
    print("-" * 80)
    
    # 멀티턴 RAG Chain 실행
    # config에 session_id를 전달하여 대화 히스토리 관리
    answer = conversational_rag_chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    
    print(f"AI: {answer}")

# ============================================================
# 대화 히스토리 확인
# ============================================================
print("\n" + "=" * 80)
print("대화 히스토리 확인")
print("=" * 80)

# 저장된 대화 내역 출력
history = store[session_id]
print(f"\n총 메시지 수: {len(history.messages)}개")
print("\n대화 내역:")
for idx, message in enumerate(history.messages, 1):
    role = "사용자" if message.type == "human" else "AI"
    print(f"\n[{idx}] {role}:")
    print(message.content[:200] + "..." if len(message.content) > 200 else message.content)


# ============================================================
# 대화형 RAG 시스템 - 멀티턴
# ============================================================
print("\n대화형 RAG 시스템 시작 (멀티턴)")
print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
print("대화 기록 초기화: 'clear' 또는 '초기화'")
print("=" * 80)

# 초기화 시 새로운 세션 시작
user_session_id = "interactive_user"

while True:
    user_question = input("\n질문: ").strip()
    
    # 종료 명령
    if user_question.lower() in ['quit', 'exit', '종료']:
        print("\nRAG 시스템을 종료합니다.")
        break
    
    # 대화 초기화 명령
    if user_question.lower() in ['clear', '초기화']:
        if user_session_id in store:
            store[user_session_id] = InMemoryChatMessageHistory()
        print("대화 기록이 초기화되었습니다.")
        continue
    
    # 빈 입력 체크
    if not user_question:
        print("질문을 입력해주세요.")
        continue
    
    try:
        print("\n문서 검색 중...")
        
        # 멀티턴 RAG Chain 실행
        answer = conversational_rag_chain.invoke(
            {"question": user_question},
            config={"configurable": {"session_id": user_session_id}}
        )
        
        print(f"\nAI: {answer}")
        print("-" * 80)
        
    except Exception as e:
        print(f"\n에러 발생: {e}")