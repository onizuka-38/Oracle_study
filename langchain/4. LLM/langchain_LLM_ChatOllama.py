from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 모델 초기화 (로컬에서 실행 중인 gemma3:12b 모델 지정)
# OpenSource 모델이므로 API 키는 필요 없습니다.
model = ChatOllama(model="gemma3:12b")

# 2. 체인 구성 및 실행
prompt = ChatPromptTemplate.from_template("세상에서 가장 높은 산은 무엇인가요?")
chain = prompt | model | StrOutputParser()

print("--- Gemma3 ---")
print(chain.invoke({}))