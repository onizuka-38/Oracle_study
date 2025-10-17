from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_postgres import PGVector
from langchain_core.messages import AIMessage, HumanMessage
import asyncio

load_dotenv()

# --- 1. LLM 설정 ---
# llm = ChatOpenAI(
#     model="gpt-4o-mini", 
#     temperature=0.5,
#     streaming=True
# )

# Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    streaming=True
)

# --- 2. Embedding 모델 설정 ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 3. PGVector 설정 ---
CONNECTION_STRING = "postgresql+psycopg://test:5748@localhost:5432/testdb"
COLLECTION_NAME = "rag_example"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

# --- 4. 벡터 검색 도구 정의 ---
@tool
def search_documents(query: str) -> str:
    """
    데이터베이스에서 벡터 유사도 검색을 수행합니다.
    사용자의 질문과 의미적으로 유사한 문서를 찾아 반환합니다.
    
    예시:
    - "2024년 영국 AI 산업 현황은?" 
    - "중국의 AI 플러스 정책은?"
    - "최근 공개된 AI 모델은?"
    """
    try:
        results = vector_store.similarity_search(query, k=5)
        
        if not results:
            return "검색 결과가 없습니다."
        
        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append(f"[문서 {i}]\n{doc.page_content}\n")
        
        return "\n".join(formatted_results)
    
    except Exception as e:
        return f"검색 중 오류 발생: {str(e)}"

@tool
def search_documents_with_score(query: str) -> str:
    """
    데이터베이스에서 벡터 유사도 검색을 수행하고 유사도 점수도 함께 반환합니다.
    검색 결과의 신뢰도를 확인하고 싶을 때 사용하세요.
    """
    try:
        results = vector_store.similarity_search_with_score(query, k=5)
        
        if not results:
            return "검색 결과가 없습니다."
        
        formatted_results = []
        for i, (doc, score) in enumerate(results, 1):
            formatted_results.append(
                f"[문서 {i}] (유사도: {score:.4f})\n{doc.page_content[:500]}...\n"
            )
        
        return "\n".join(formatted_results)
    
    except Exception as e:
        return f"검색 중 오류 발생: {str(e)}"

# --- 5. 도구 리스트 ---
tools = [search_documents, search_documents_with_score]

# --- 6. 멀티턴 대화를 위한 프롬프트 ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 AI 산업 동향 분석 전문가입니다.

    **역할:**
    1. 사용자의 질문을 분석하여 벡터 검색이 필요한지 판단합니다.
    2. 이전 대화 내용을 참고하여 맥락을 이해합니다.
    3. 벡터 유사도 검색을 통해 관련 문서를 찾습니다.
    4. 검색된 정보를 바탕으로 정확하고 자연스러운 답변을 생성합니다.

    **데이터베이스 내용:**
    - SPRi AI Brief 2025년 10월호
    - AI 산업의 최신 동향 (정책, 기업, 기술, 인력/교육)
    - 각국의 AI 정책 및 규제
    - 주요 기업의 AI 모델 발표
    - AI 기술 연구 동향

    **작업 순서:**
    1. 이전 대화 맥락 파악
    2. 현재 질문 분석 (대명사나 축약 표현을 맥락에 맞게 해석)
    3. search_documents 도구로 관련 문서 검색
    4. 검색 결과를 바탕으로 정확한 답변 작성

    **중요:**
    - 이전 대화를 기억하고 연속적인 질문에 답변하세요
    - "그것", "그 회사", "더 자세히" 등의 표현은 이전 대화를 참고하세요
    - 검색된 문서 내용만을 기반으로 답변하세요
    - 답변 시 출처를 간단히 언급하세요
    """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

# --- 7. 에이전트 생성 ---
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# --- 8. 멀티턴 대화 함수 (실제 토큰 스트리밍) ---
async def run_multiturn_conversation():
    """멀티턴 대화를 지원하는 대화형 루프 (토큰 단위 스트리밍)"""
    chat_history = []
    
    print("=" * 70)
    print("Agentic RAG 시스템 시작 (멀티턴 대화 + 토큰 스트리밍)")
    print("=" * 70)
    print("\n종료: 'quit', 'exit', '종료' 입력")
    print("대화 초기화: 'reset', 'clear', '초기화' 입력")
    print("=" * 70)
    
    while True:
        # asyncio에서 input 사용
        user_input = await asyncio.to_thread(input, "\n질문: ")
        user_input = user_input.strip()
        
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("시스템을 종료합니다.")
            break
        
        if user_input.lower() in ['reset', 'clear', '초기화']:
            chat_history = []
            print("대화 기록이 초기화되었습니다.")
            continue
        
        if not user_input:
            continue
        
        try:
            print("\n답변: ", end="", flush=True)
            
            full_response = ""
            
            # astream_events로 실제 토큰 단위 스트리밍(비동기)
            async for event in agent_executor.astream_events(
                {
                    "input": user_input,
                    "chat_history": chat_history
                },
                version="v2"
            ):
                kind = event["event"]
                
                # 도구 시작
                if kind == "on_tool_start":
                    tool_name = event["name"]
                    print(f"\n[{tool_name}] 검색 중...", end="", flush=True)
                
                # 도구 종료
                elif kind == "on_tool_end":
                    print(" 완료\n답변: ", end="", flush=True)
                
                # LLM 토큰 스트리밍 (최종 답변)
                elif kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        print(content, end="", flush=True)
                        full_response += content
            
            print()  # 줄바꿈
            
            # 대화 기록에 추가
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=full_response))
            
        except Exception as e:
            print(f"\n오류 발생: {str(e)}")

# --- 9. 실행 ---
if __name__ == "__main__":
    asyncio.run(run_multiturn_conversation())