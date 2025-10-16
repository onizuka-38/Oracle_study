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
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)