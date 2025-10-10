# COPYRIGHT 2025 (C) WYHIL. ALL RIGHTS RESERVED
# crux:PLATFORM | CONFIDENTIAL crux:ACADEMY

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

BASE = "https://finance.naver.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def get_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    # 네이버는 주로 euc-kr. 탐지 실패 대비 fallback
    r.encoding = r.apparent_encoding or "euc-kr"
    if "euc" not in (r.encoding or "").lower():
        r.encoding = "euc-kr"
    return BeautifulSoup(r.text, "html.parser")

# ------------------ 1) 뉴스: href 패턴 기반 ------------------
def extract_news(soup: BeautifulSoup, limit=10):
    items = []
    # 우선: href에 news_read 포함된 링크(네이버 금융 뉴스 본문)
    for a in soup.select('a[href*="news_read"]'):
        title = a.get_text(strip=True)
        href = urljoin(BASE, a.get("href", ""))
        if title and href not in {h for _, h in items}:
            items.append((title, href))
            if len(items) >= limit:
                break
    # 후보가 없으면 넓게: "news"가 클래스/아이디에 포함된 컨테이너 내부 링크
    if not items:
        for a in soup.select('[class*="news"] a, [id*="news"] a'):
            title = a.get_text(strip=True)
            href = urljoin(BASE, a.get("href", ""))
            if title and "news" in href and href not in {h for _, h in items}:
                items.append((title, href))
                if len(items) >= limit:
                    break
    return items

# ------------------ 2) 인기 종목: 종목 상세 URL 패턴 ------------------
def extract_popular_stocks(soup: BeautifulSoup, limit=10):
    names = []
    for a in soup.select('a[href*="/item/main.naver?code="]'):
        t = a.get_text(strip=True)
        if t and t not in names:
            names.append(t)
        if len(names) >= limit:
            break
    # 없으면 aside 영역 추정 fallback (클래스명 변화 대비)
    if not names:
        for a in soup.select('[class*="popular"] a, [id*="popular"] a'):
            t = a.get_text(strip=True)
            if t and t not in names:
                names.append(t)
            if len(names) >= limit:
                break
    return names

# ------------------ 3) 환율: 메인 실패 시 marketindex 페이지로 대체 ------------------
def extract_exchange(limit=6):
    # 메인에서 시도
    soup = get_soup(BASE)
    rows = []
    for tr in soup.select('[class*="exchange"] tbody tr, [id*="exchange"] tbody tr'):
        tds = [td.get_text(strip=True) for td in tr.select('td')]
        if len(tds) >= 2:
            rows.append((tds[0], tds[1]))
        if len(rows) >= limit:
            break
    if rows:
        return rows

    # fallback: 전용 페이지 (구조가 훨씬 안정적)
    mkt = get_soup("https://finance.naver.com/marketindex/exchangeList.naver")
    for tr in mkt.select("table tbody tr"):
        name = tr.select_one("td.tit a")
        price = tr.select_one("td.sale")
        if name and price:
            rows.append((name.get_text(strip=True), price.get_text(strip=True)))
        if len(rows) >= limit:
            break
    return rows

if __name__ == "__main__":
    soup = get_soup(BASE)

    print("📢 [뉴스 상위]")
    for title, href in extract_news(soup, limit=8):
        print("-", title, "->", href)

    print("\n📈 [인기 종목 추정]")
    for n in extract_popular_stocks(soup, limit=10):
        print("-", n)

    print("\n💱 [환율]")
    for cur, price in extract_exchange(limit=8):
        print(f"{cur}: {price}")
