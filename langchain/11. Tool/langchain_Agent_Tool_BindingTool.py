from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
import os

# SERPER_API_KEY 설치 필요 -> https://serper.dev
# 무료 발급 가능 / 2500회 무료

load_dotenv()

# Google Search 도구 정의 (Serper API 사용)
@tool
def google_search(query: str) -> str:
    """인터넷에서 정보를 검색합니다. 최신 정보나 실시간 데이터가 필요할 때 사용하세요."""
    search = GoogleSerperAPIWrapper()
    result = search.run(query)
    return result

# 계산기 도구 정의
@tool
def calculator(expression: str) -> str:
    """수학 계산을 수행합니다. 예: '2 + 2' 또는 '10 * 5'"""
    try:
        result = eval(expression)
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"

# LLM 설정
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 도구 리스트
tools = [google_search, calculator]

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 유능한 AI 어시스턴트입니다. 사용자의 질문에 답하기 위해 필요한 도구를 사용하세요."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 에이전트 생성
agent = create_tool_calling_agent(llm, tools, prompt)

# 에이전트 실행기 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# 실행 예제
if __name__ == "__main__":
    # 예제 1: 구글 검색 사용
    print("=" * 50)
    print("질문 1: 2024년 노벨 물리학상 수상자는?")
    print("=" * 50)
    response1 = agent_executor.invoke({
        "input": "2024년 노벨 물리학상 수상자는 누구인가요?"
    })
    print(f"\n답변: {response1['output']}\n")
    
    # 예제 2: 계산기 사용
    print("=" * 50)
    print("질문 2: 123 * 456은 얼마인가요?")
    print("=" * 50)
    response2 = agent_executor.invoke({
        "input": "123 * 456을 계산해주세요"
    })
    print(f"\n답변: {response2['output']}\n")
    
    # 예제 3: 검색과 계산 조합
    print("=" * 50)
    print("질문 3: 현재 비트코인 가격을 알려주고 10개 구매 시 비용은?")
    print("=" * 50)
    response3 = agent_executor.invoke({
        "input": "현재 비트코인 가격을 검색하고, 10개를 구매하면 얼마인지 계산해주세요"
    })
    print(f"\n답변: {response3['output']}\n")