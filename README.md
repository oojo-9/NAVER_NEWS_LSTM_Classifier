# NAVER_NEWS_LSTM_Classifier
 - NAVER NEWS : IT , SPORTS , ENTERTAINMENT
 - 3가지 카테고리 뉴스를 기반으로 뉴스 제목 입력 시 카테고리 분류 

   - 사용 URL : 
[IT]<br>
https://news.naver.com/breakingnews/section/105/230<br>
https://news.naver.com/breakingnews/section/105/226<br>
https://news.naver.com/breakingnews/section/105/227<br>
https://news.naver.com/breakingnews/section/105/228<br>
https://news.naver.com/breakingnews/section/105/229<br>
<br>
   - [SPORTS]<br>
   https://m.sports.naver.com/general/news<br>
   https://m.sports.naver.com/kfootball/news<br>
   <br>
   - [ENTERTAINMENT]<br>
   https://m.entertain.naver.com/now<br>



## 프로젝트 구조
```
NAVER_NEWS_LSTM_Classifier/
├─ app/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ preprocess.py
│  ├─ model.py
│  ├─ train.py
│  ├─ predict.py
│  └─ visualize.py
│
├─ models/
├─ data/
│  ├─ confusion_matrix.png
│  ├─ data_count_graph.png
│  └─ loss_graph.png
│
├─ main.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```


## 이슈
- 스포츠, 엔터의 경우 URL을 입력하여 크롤링 시 지속 제목을 인식하지 못함
└ API 주소를 통해 해결 완료

- 뉴스 종류가 많지 않아서 데이터 수치가 좋지않고, 예측카테고리가 좋지 않음
└ 여러 링크를 통해 데이터를 수집함