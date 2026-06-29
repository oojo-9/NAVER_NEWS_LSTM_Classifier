from __future__ import annotations

from sklearn.metrics import classification_report, accuracy_score
from app.visualize import save_loss_graph, save_confusion_matrix, save_data_count_graph

import os
import pickle
import random
from logging import critical
from typing import Dict, Tuple

import numpy as np
import torch
from ipykernel import embed
from torch._dynamo.variables import optimizer

from app import config

torch.set_num_threads(1)  # 작은 실습 데이터에서는 CPU 스레드를 1개로 제한하여 실행 환경별 지연을 줄인다.
torch.backends.mkldnn.enabled = False  # 일부 CPU 환경에서 LSTM 연산이 오래 멈추는 문제를 피하기 위해 MKLDNN을 비활성화한다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import Config
from app.data import load_sample_data
from app.model import TextLSTMClassifier
from app.preprocess import build_vocab, clean_text, encode_labels, pad_sequences, texts_to_sequences

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_model(config: Config) -> Tuple[TextLSTMClassifier, Dict[str, object]]:
    set_seed(config.random_state)
    raw_texts, labels = load_sample_data()
    cleaned_texts = [clean_text(text) for text in raw_texts]
    vocab = build_vocab(cleaned_texts, config.max_vocab)
    sequences = texts_to_sequences(cleaned_texts, vocab)
    x = pad_sequences(sequences, config.max_len)
    y, label_to_id, id_to_label = encode_labels(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.test_size,
        random_state=config.random_state,
        stratify=y)


    train_dataset = TensorDataset(torch.tensor(x_train),torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(x_test),torch.tensor(y_test))
    train_loader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=config.batch_size)

    model = TextLSTMClassifier(vocab_size=len(vocab),embed_dim=config.embed_dim,hidden_dim=config.hidden_dim,num_classes=len(label_to_id))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    train_losses = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch:02d}/{config.epochs} - loss: {avg_loss:.4f}")

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.tolist())
            all_targets.extend(batch_y.tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    print(f"평가 정확도: {accuracy:.4f}")
    print(classification_report(all_targets, all_preds, target_names=[id_to_label[i] for i in range(len(id_to_label))], zero_division=0))

    # -----------------------------
    # 시각화
    # -----------------------------
    save_loss_graph(train_losses)

    save_confusion_matrix(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=[id_to_label[i] for i in range(len(id_to_label))]
    )

    save_data_count_graph(labels)

    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    torch.save(model.state_dict(), config.model_path)
    with open(config.model_path.replace(".pt", "_meta.pkl"), "wb") as f:
        pickle.dump({"vocab": vocab, "label_to_id": label_to_id, "id_to_label": id_to_label, "config": config}, f)

    metadata = {"vocab": vocab, "label_to_id": label_to_id, "id_to_label": id_to_label, "accuracy": accuracy}
    return model, metadata


if __name__ == "__main__":
    config = Config()
    train_model(config)
