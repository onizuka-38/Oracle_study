import requests
from bs4 import BeautifulSoup
import re

######## HTML 문자열에서 파싱 ##################
html_doc = "<html><body><h1>Hello World</h1></body></html>"
soup = BeautifulSoup(html_doc, 'html.parser')

# 파일에서 파싱 (없으면 건너뜀)
try:
    with open('example.html', 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
except FileNotFoundError:
    pass

# 웹 페이지에서 파싱
url = 'https://finance.naver.com'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}
try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    response.encoding = 'euc-kr'  # 네이버는 euc-kr 인코딩 사용
    soup = BeautifulSoup(response.text, 'html.parser')
except requests.RequestException as e:
    print("HTTP 요청 오류:", e)
    soup = BeautifulSoup("", "html.parser")

######## 요소찾기 : 태그 이름으로 찾기 ##################
# 첫 번째 태그 찾기
title = soup.find('title')
print(title.text)

# 모든 태그 찾기
all_links = soup.find_all('a')
for link in all_links:
    href = link.get('href')
    if href:
        print(href)

######## 요소찾기 : css 선택자로 찾기 ##################
# 클래스로 찾기
elements = soup.select('.class-name')

# ID로 찾기
element = soup.select_one('#id-name')

# 복합 선택자
items = soup.select('div.container > p.description')

######## 요소찾기 : 속성으로 찾기 ##################
# 특정 속성이 있는 요소 찾기
images = soup.find_all('img', src=True)

# 특정 속성값으로 찾기
logo = soup.find('img', {'src': 'logo.png'})

# 정규표현식 사용
images = soup.find_all('img', src=re.compile(r'\.png$', re.I))
