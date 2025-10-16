from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template('사용자 요청을 JSON으로 분석해줘. 요청: 콜라 2개 주문할게\n\nJSON: {{"item": "콜라", "quantity": 2}}')
parser = JsonOutputParser()

# 전: 파서 미적용 (모델의 원본 출력)
chain_before = prompt | model | StrOutputParser()
result_before = chain_before.invoke({})

# 후: JsonOutputParser 적용
chain_after = prompt | model | parser
result_after = chain_after.invoke({})

print("--- [전] 파서 미적용 (LLM 원본 출력) ---")
print(result_before)
print(f"타입: {type(result_before)}\n")

print("--- [후] JsonOutputParser 적용 ---")
print(result_after)
print(f"타입: {type(result_after)}")