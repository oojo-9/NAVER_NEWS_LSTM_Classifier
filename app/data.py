from typing import List, Tuple
from datetime import datetime

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


# IT 뉴스: HTML 크롤링
def get_news_it() -> List[Tuple[str, str]]:
    urls = [
        "https://news.naver.com/breakingnews/section/105/230",
        "https://news.naver.com/breakingnews/section/105/226",
        "https://news.naver.com/breakingnews/section/105/227",
        "https://news.naver.com/breakingnews/section/105/228",
        "https://news.naver.com/breakingnews/section/105/229",
    ]

    result = []

    for url in urls:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        title_tags = soup.select(".sa_text_strong")

        for tag in title_tags:
            title = tag.get_text(strip=True)
            if title:
                result.append((title, "IT"))

    result = list(dict.fromkeys(result))

    print("=" * 50)
    print("IT")
    print("count:", len(result))

    return result


# 스포츠 뉴스: API
from datetime import datetime

def get_news_sports() -> List[Tuple[str, str]]:

    today = datetime.now().strftime("%Y%m%d")

    urls = [
        f"https://api-gw.sports.naver.com/news/articles/general?sort=latest&date={today}&page=1&pageSize=100&isPhoto=N",
        f"https://api-gw.sports.naver.com/news/articles/kfootball?sort=latest&date={today}&page=1&pageSize=100&isPhoto=N",
    ]

    result = []

    for url in urls:

        response = requests.get(url, headers=HEADERS, timeout=10)
        data = response.json()

        news_list = data.get("result", {}).get("newsList", [])

        for news in news_list:

            title = news.get("title")

            if title:
                result.append((title, "SPORTS"))

    result = list(dict.fromkeys(result))

    print("=" * 50)
    print("SPORTS")
    print("count:", len(result))

    return result


# 연예 뉴스: API
def get_news_entertainment() -> List[Tuple[str, str]]:
    today = datetime.now().strftime("%Y%m%d")

    url = (
        "https://api-gw.entertain.naver.com/news/articles"
        f"?date={today}&page=1&pageSize=100"
    )

    response = requests.get(url, headers=HEADERS, timeout=10)
    data = response.json()

    news_list = data.get("result", {}).get("newsList", [])

    result = []

    for news in news_list:
        title = news.get("title")
        if title:
            result.append((title, "ENTERTAINMENT"))

    result = list(dict.fromkeys(result))

    print("=" * 50)
    print("ENTERTAINMENT")
    print("count:", len(result))

    return result


# 전체 데이터 로드
def load_sample_data() -> Tuple[List[str], List[str]]:
    news_data = []

    news_data.extend(get_news_it())
    news_data.extend(get_news_sports())
    news_data.extend(get_news_entertainment())

    news_data = list(dict.fromkeys(news_data))

    texts = [text for text, _ in news_data]
    labels = [label for _, label in news_data]

    print("=" * 50)
    print("전체 데이터:", len(texts))
    print("IT:", labels.count("IT"))
    print("SPORTS:", labels.count("SPORTS"))
    print("ENTERTAINMENT:", labels.count("ENTERTAINMENT"))

    if labels.count("IT") == 0 or labels.count("SPORTS") == 0 or labels.count("ENTERTAINMENT") == 0:
        raise ValueError("크롤링 실패: 0개인 카테고리가 있습니다.")

    return texts, labels


if __name__ == "__main__":
    texts, labels = load_sample_data()