from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

load_dotenv()

# --- 1. LLM 설정 ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 2. PostgreSQL 연결 ---
db = SQLDatabase.from_uri("postgresql+psycopg://test:5748@localhost:5432/testdb")

# --- 3. SQL 실행 도구 정의 ---
@tool
def run_sql_query(query: str) -> str:
    """SQL 쿼리를 실행하고 결과를 반환합니다."""
    db_tool = QuerySQLDataBaseTool(db=db)
    return db_tool.invoke({"query": query})

tools = [run_sql_query]

# --- 4. 에이전트 생성 ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 데이터베이스 전문가입니다. 사용자의 질문을 SQL 쿼리로 변환해 실행하세요."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}") # 이전 도구 호출 기록
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 5. 실행 예시 ---
agent_executor.invoke({"input": "langchain_pg_embedding 테이블의 모든 이름을 보여줘"})
