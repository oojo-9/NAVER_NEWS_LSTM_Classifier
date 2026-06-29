from dataclasses import dataclass

@dataclass
class Config:
    max_vocab: int = 5000  # 토큰화에 사용할 최대 단어 수이다.
    max_len: int = 80  # 모든 기사 문장의 길이를 동일하게 맞추기 위한 최대 토큰 길이이다.
    embed_dim: int = 64  # 각 단어 정수를 몇 차원의 임베딩 벡터로 바꿀지 정한다.
    hidden_dim: int = 64  # LSTM 내부 은닉 상태의 차원 수이다.
    batch_size: int = 8  # 한 번의 학습 단계에서 모델에 넣을 샘플 개수이다.
    epochs: int = 8  # 전체 학습 데이터를 몇 번 반복해서 학습할지 정한다.
    learning_rate: float = 0.001  # Adam 최적화 알고리즘의 학습률이다.
    test_size: float = 0.25  # 전체 데이터 중 평가 데이터로 사용할 비율이다.
    random_state: int = 42  # 실험 결과를 재현하기 위한 난수 고정값이다.
    model_path: str = "../models/naver_lstm_model.pt"  # 학습된 모델을 저장할 경로이다.
