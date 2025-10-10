######## naver_stock.py ########
import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook

# ✅ 1. 요청 보내기
url = 'https://finance.naver.com/'
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
response = requests.get(url, headers=headers)
response.raise_for_status()

# ✅ 2. HTML 파싱
soup = BeautifulSoup(response.text, 'html.parser')

# ✅ 3. 인기 종목 테이블 선택
tbody = soup.select_one(
    '#container > div.aside > div.group_aside > '
    'div.aside_area.aside_popular > table > tbody'
)

if not tbody:
    print("❌ 테이블을 찾지 못했습니다. 페이지 구조가 바뀌었을 수 있습니다.")
    exit()

# ✅ 4. 행별 데이터 추출
datas = []
trs = tbody.select('tr')

for tr in trs:
    name_tag = tr.select_one('th > a')
    price_tag = tr.select_one('td')
    change_tag = tr.select_one('td > span')

    if not name_tag or not price_tag:
        continue

    name = name_tag.get_text(strip=True)
    current_price = price_tag.get_text(strip=True)
    change_price = change_tag.get_text(strip=True) if change_tag else ''
    # 종목 상승/하락 여부는 tr의 class로 구분됨 (없으면 'none')
    change_direction = tr.get('class', ['none'])[0]

    datas.append([name, current_price, change_direction, change_price])

# ✅ 5. 엑셀로 저장
wb = Workbook()
# 기본 시트 제거 후 새 시트 생성
default_sheet = wb.active
wb.remove(default_sheet)
ws = wb.create_sheet('결과')

# 헤더 추가
ws.append(['종목명', '현재가', '등락방향', '변동폭'])
for row in datas:
    ws.append(row)

save_path = 'naver_fin01.xlsx'
wb.save(save_path)
print(f"✅ 저장 완료: {save_path}")
