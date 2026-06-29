# 시각화
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_loss_graph(train_losses, save_path="data/loss_graph.png"):
    os.makedirs("data", exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    print(f"Loss 그래프 저장 완료: {save_path}")


def save_confusion_matrix(y_true, y_pred, class_names, save_path="data/confusion_matrix.png"):
    os.makedirs("data", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    plt.figure(figsize=(7, 6))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

    print(f"Confusion Matrix 저장 완료: {save_path}")


def save_data_count_graph(labels, save_path="data/data_count_graph.png"):
    os.makedirs("data", exist_ok=True)

    label_names = sorted(set(labels))
    counts = [labels.count(label) for label in label_names]

    plt.figure(figsize=(8, 5))
    plt.bar(label_names, counts)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("News Data Count by Category")
    plt.grid(axis="y")
    plt.savefig(save_path)
    plt.close()

    print(f"데이터 개수 그래프 저장 완료: {save_path}")