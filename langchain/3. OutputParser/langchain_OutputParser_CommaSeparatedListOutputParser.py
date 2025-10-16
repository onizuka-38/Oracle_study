from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("인공지능과 관련된 키워드 5개를 쉼표로 구분하여 나열해줘.")
parser = CommaSeparatedListOutputParser()

# 전: 파서 미적용 (모델의 원본 출력)
chain_before = prompt | model | StrOutputParser() # 문자열만 확인하기 위해 StrOutputParser 사용
result_before = chain_before.invoke({})

# 후: CommaSeparatedListOutputParser 적용
chain_after = prompt | model | parser
result_after = chain_after.invoke({})

print("--- [전] 파서 미적용 (LLM 원본 출력) ---")
print(result_before)
print(f"타입: {type(result_before)}\n")

print("--- [후] CommaSeparatedListOutputParser 적용 ---")
print(result_after)
print(f"타입: {type(result_after)}")