from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """
# LangChain

## Chains
Chains는 LangChain의 핵심입니다.

## Agents
Agents는 LLM이 환경과 상호작용하게 합니다.
"""

# headers_to_split_on: 분할 기준으로 사용할 (헤더, 별칭) 튜플 리스트
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]

text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

chunks = text_splitter.split_text(markdown_text)

print(f"--- MarkdownHeaderTextSplitter (총 {len(chunks)}개) ---")
for chunk in chunks:
    print(chunk)