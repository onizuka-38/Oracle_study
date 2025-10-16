from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# API 키 설정
load_dotenv()

# 1. 체인 구성 (이 체인 자체가 하나의 Runnable)
prompt = ChatPromptTemplate.from_template("{framework}는 어떤 특징을 가지고 있어?")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | model | parser

# 'stream'을 사용하여 체인 실행
print("--- stream() 결과 ---")
for chunk in chain.stream({"framework": "Vue.js"}):
    print(chunk, end="", flush=True)
print("\n") # 마지막에 줄바꿈 추가