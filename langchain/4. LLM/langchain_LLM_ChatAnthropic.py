from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 1. API 키 로드
load_dotenv()

# 2. 모델 초기화 (Claude 4 Sonnet 사용)
model = ChatAnthropic(model="claude-sonnet-4-20250514")

# 3. 체인 구성 및 실행
prompt = ChatPromptTemplate.from_template("세상에서 가장 높은 산은 무엇인가요?")
chain = prompt | model | StrOutputParser()

print(chain.invoke({}))