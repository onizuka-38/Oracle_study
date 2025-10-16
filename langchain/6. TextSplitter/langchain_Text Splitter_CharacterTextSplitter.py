from langchain_text_splitters import CharacterTextSplitter

sample_text = "안녕하세요. 반갑습니다.\n\nLangChain 강의에 오신 것을 환영합니다.\n\n텍스트 분할을 알아봅시다."

# separator: 분할 기준으로 사용할 문자
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=50,
    chunk_overlap=10
)

chunks = text_splitter.split_text(sample_text)

print(f"--- CharacterTextSplitter (총 {len(chunks)}개) ---")
for i, chunk in enumerate(chunks):
    print(f"[{i+1}] {chunk}")