from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "gesture_mlp.pth"
TEST_PATH = ROOT / "dataset" / "landmarks" / "test.csv"
OUTPUT_DIR = ROOT / "results"

FEATURE_COLUMNS = [f"feature_{i}" for i in range(63)]


class GestureMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    classes = checkpoint["classes"]

    model = GestureMLP(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_df = pd.read_csv(TEST_PATH)

    features = test_df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    labels = test_df["label"].map({
        name: index
        for index, name in enumerate(classes)
    })

    if labels.isna().any() or not np.isfinite(features).all():
        raise ValueError("Invalid test data detected.")

    y_true = labels.to_numpy(dtype=np.int64)

    X = torch.from_numpy(features)

    with torch.inference_mode():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        y_pred = probabilities.argmax(dim=1).numpy()

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        average="macro",
        zero_division=0,
    )

    print("=" * 50)
    print("INDEPENDENT TEST RESULTS")
    print("=" * 50)
    print(f"Test samples: {len(y_true)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    ))

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
    )

    pd.DataFrame(
        matrix,
        index=classes,
        columns=classes,
    ).to_csv(
        OUTPUT_DIR / "test_confusion_matrix.csv"
    )

    results = test_df[["image", "label"]].copy()
    results["predicted"] = [
        classes[index] for index in y_pred
    ]
    results["confidence"] = (
        probabilities.max(dim=1).values.numpy()
    )

    results.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    print("\nSaved evaluation files to results/")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
