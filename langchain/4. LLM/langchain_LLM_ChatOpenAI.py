from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 1. API 키 로드
load_dotenv()

# 2. 모델 초기화 (API 사용)
model = ChatOpenAI(model="gpt-4o-mini")

# 3. 체인 구성 및 실행
prompt = ChatPromptTemplate.from_template("세상에서 가장 높은 산은 무엇인가요?")
chain = prompt | model | StrOutputParser()

print(chain.invoke({}))