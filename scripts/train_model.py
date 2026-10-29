from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "landmarks"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CLASSES = [
    "fist", "like", "dislike", "one",
    "peace", "palm", "no_gesture"
]

FEATURE_COLUMNS = [f"feature_{i}" for i in range(63)]

SEED = 42
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 15

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class GestureMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(CLASSES)),
        )

    def forward(self, x):
        return self.network(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(split):
    path = DATA_DIR / f"{split}.csv"
    df = pd.read_csv(path)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["label"].map(
        {name: index for index, name in enumerate(CLASSES)}
    ).to_numpy()

    if np.isnan(X).any() or pd.isna(y).any():
        raise ValueError(f"Invalid data found in {path}")

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)

            predictions.extend(
                logits.argmax(dim=1).cpu().numpy()
            )
            targets.extend(y.cpu().numpy())

    loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(
        targets, predictions,
        labels=list(range(len(CLASSES))),
        average="macro",
        zero_division=0,
    )

    return loss, accuracy, macro_f1, targets, predictions


def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_data("train")
    X_val, y_val = load_data("val")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_counts = torch.bincount(
        y_train, minlength=len(CLASSES)
    ).float()

    class_weights = (
        class_counts.sum()
        / (len(CLASSES) * class_counts.clamp(min=1))
    ).to(DEVICE)

    model = GestureMLP().to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_f1 = -1.0
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []

    print(f"Device: {DEVICE}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)

        train_loss /= len(train_loader.dataset)

        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        improved = (
            val_f1 > best_f1 + 1e-5
            or (
                abs(val_f1 - best_f1) <= 1e-5
                and val_loss < best_loss
            )
        )

        if improved:
            best_f1 = val_f1
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    model.load_state_dict(best_state)

    model_path = MODEL_DIR / "gesture_mlp.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "input_size": 63,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
    }, model_path)

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / "training_history.csv",
        index=False,
    )

    _, val_acc, val_f1, targets, predictions = evaluate(
        model, val_loader, criterion
    )

    print("\n" + "=" * 50)
    print("BEST MODEL RESULTS")
    print("=" * 50)
    print(f"Best epoch: {best_epoch}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation macro F1: {val_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        targets,
        predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
    ))

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(CLASSES))),
    )

    pd.DataFrame(
        matrix,
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(RESULTS_DIR / "val_confusion_matrix.csv")

    print(f"\nModel saved: {model_path}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
