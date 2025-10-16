from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
import os

# SERPER_API_KEY 설치 필요 -> https://serper.dev/api-key
# 무료 발급 가능 / 2500회 무료

load_dotenv()

# Google Search 도구 정의 (Serper API 사용)
@tool
def goolge_search(query: str) -> str:
    """주어진 질문에 대한 검색 결과를 찾아 반환합니다."""
    search = GoogleSerperAPIWrapper()
    result = search.run(query)
    return result

# 계산기 도구 정의
@tool
def calculate(expression: str) -> str:
    """주어진 수식을 계산하여 반환합니다."""
    try:
        result = eval(expression)
        return f"수식을 계산하면 {result} 가 됩니다."
    except Exception as e:
        return f"수식을 계산하는 데 실패했습니다. 오류: {e}"


llm = ChatOpenAI(model="gpt-4", temperature=0)

tools = [goolge_search, calculate]

prompt = ChatPromptTemplate.from_messages([
    ("system", "주어진 질문에 대한 검색 결과를 찾아 반환합니다."),
    ("human", "{input}"),
    ("placeholder","{agent_scratchpad}")
])

# 에이전트 생성
agent = create_tool_calling_agent(llm, tools, prompt=prompt)

# 에이전트 실행
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 실행 예제
if __name__ == "__main__":
    # 예제 1: 구글 검색 사용
    print("=" * 50)
    print("질문1: 2024년 노벨 화학상 수상자는?")
    print("=" * 50)
    response1 = agent_executor.invoke({
        "input": "질문1: 2024년 노벨 화학상 수상자는?"
    })
    print(f"\n답변 : {response1['output']}")
    
    # 예제 2: 계산기 사용
    print("=" * 50)
    print("질문2: 123*456 을 계산하여 반환합니다.")
    print("=" * 50)
    response2 = agent_executor.invoke({
        "input": "질문2: 123*456 을 계산하여 반환합니다."
    })
    print(f"\n답변 : {response2['output']}")
    
    # 예제 3 : 검색과 계산 조합
    print("=" * 50)
    print("질문3: 현재 비트코인 가격을 알려주고 10개 구매시 비용은?")
    print("=" * 50)
    response3 = agent_executor.invoke({
        "input": "질문3: 현재 비트코인 가격을 알려주고 10개 구매시 비용은?"
    })
    print(f"\n답변 : {response3['output']}")