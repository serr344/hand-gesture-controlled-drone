from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent

PYTORCH_MODEL = ROOT / "models" / "gesture_mlp.pth"
ONNX_MODEL = ROOT / "models" / "gesture_mlp.onnx"
TEST_DATA = ROOT / "dataset" / "landmarks" / "test.csv"


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
    checkpoint = torch.load(
        PYTORCH_MODEL,
        map_location="cpu",
        weights_only=True,
    )

    classes = checkpoint["classes"]

    model = GestureMLP(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Export the model with a dynamic batch dimension.
    example_input = torch.randn(1, 63)

    exported = torch.onnx.export(
        model,
        (example_input,),
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: "batch_size"}},
        opset_version=18,
        dynamo=True,
    )

    exported.save(str(ONNX_MODEL))

    # Validate the ONNX model structure.
    onnx_model = onnx.load(str(ONNX_MODEL))
    onnx.checker.check_model(onnx_model)

    # Compare PyTorch and ONNX predictions.
    df = pd.read_csv(TEST_DATA)

    features = df[
        [f"feature_{i}" for i in range(63)]
    ].to_numpy(dtype=np.float32)

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )

    with torch.inference_mode():
        pytorch_logits = model(
            torch.from_numpy(features)
        ).numpy()

    onnx_logits = session.run(
        ["logits"],
        {"landmarks": features},
    )[0]

    max_difference = np.max(
        np.abs(pytorch_logits - onnx_logits)
    )

    pytorch_predictions = np.argmax(
        pytorch_logits, axis=1
    )

    onnx_predictions = np.argmax(
        onnx_logits, axis=1
    )

    agreement = np.mean(
        pytorch_predictions == onnx_predictions
    )

    print("=" * 45)
    print("ONNX EXPORT RESULTS")
    print("=" * 45)

    print(f"ONNX model: {ONNX_MODEL.name}")
    print(f"Classes: {len(classes)}")
    print(f"Test samples: {len(features)}")
    print(f"Max logit difference: {max_difference:.8f}")
    print(f"Prediction agreement: {agreement:.2%}")

    if not np.allclose(
        pytorch_logits,
        onnx_logits,
        rtol=1e-4,
        atol=1e-5,
    ):
        raise RuntimeError(
            "ONNX output differs from PyTorch output."
        )

    if agreement != 1.0:
        raise RuntimeError(
            "ONNX predictions do not match PyTorch."
        )

    print("\nONNX export verified successfully!")


if __name__ == "__main__":
    main()
