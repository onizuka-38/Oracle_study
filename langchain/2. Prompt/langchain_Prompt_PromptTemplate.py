from langchain.prompts import PromptTemplate

# 하나의 문자열로 템플릿을 정의합니다.
template = "{country}의 수도는 어디인가요?"
prompt_template = PromptTemplate.from_template(template)

# 변수를 채워 프롬프트를 완성합니다.
prompt = prompt_template.format(country="프랑스")

print(prompt)
print(type(prompt))