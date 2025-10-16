from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("하늘이 파란 이유를 설명해줘.")

# 전: 파서 미적용 체인
chain_before = prompt | model
result_before = chain_before.invoke({})

# 후: StrOutputParser 적용 체인
chain_after = prompt | model | StrOutputParser()
result_after = chain_after.invoke({})

print("--- [전] 파서 미적용 ---")
print(result_before)
print(f"타입: {type(result_before)}\n")

print("--- [후] StrOutputParser 적용 ---")
print(result_after)
print(f"타입: {type(result_after)}")