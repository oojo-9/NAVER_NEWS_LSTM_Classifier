from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
from konlpy.tag import Okt

okt = Okt()

STOP_WORDS = {
    "은", "는", "이", "가", "을", "를", "에", "의", "와", "과",
    "도", "로", "으로", "에서", "하다", "했다", "한다", "그리고",
}

def clean_text(text: str, remove_stopwords:bool = True) -> str:
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s]", " ", text)  # 한글, 영어, 숫자, 공백만 남긴다.
    text = re.sub(r"\s+", " ", text).strip()
    tokens = okt.morphs(text)

    if remove_stopwords:
        tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = [w for w in tokens if len(w) > 1]
    return " ".join(tokens)

def build_vocab(texts: Sequence[str], max_vocab:int) -> Dict[str, int]:

    counter : Counter[str] = Counter()
    for text in texts:
        counter.update(text.split())
    most_common = counter.most_common(max_vocab -2)
    vocab = {"<PAD>":0, "<OOV>":1}
    for index, (word, _) in enumerate(most_common, start=2):
        vocab[word] = index
    return vocab

def texts_to_sequences(texts: Sequence[str], vocab: Dict[str, int]) -> List[List[int]]:

    sequences : List[List[int]] = []
    for text in texts:
        seq = [vocab.get(word, vocab["<OOV>"]) for word in text.split()]
        sequences.append(seq)
    return sequences

def pad_sequences(sequences: Sequence[Sequence[int]], max_len: int) -> np.ndarray:

    padded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        truncated = list(seq)[-max_len:]

        if truncated:
            padded[i, -len(truncated):] = truncated
    return padded

def encode_labels(labels: Sequence[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:

    label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    encoded = np.array([label_to_id[label] for label in labels], dtype=np.int64)
    return encoded, label_to_id, id_to_label