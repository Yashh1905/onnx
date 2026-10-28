
import os
import sys
import traceback
from pathlib import Path

import torch
import onnxruntime as ort

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from optimum.onnxruntime import ORTModelForSequenceClassification


def download_onnx_model(
    model_name="microsoft/deberta-v3-small",
    output_dir="./models/onnx_models/deberta-v3-small",
    num_labels=2,
):
    """
    Download a tokenizer and PyTorch sequence-classification model,
    export the model to ONNX, and test ONNX inference.

    Args:
        model_name: Hugging Face model identifier.
        output_dir: Root directory for saved model files.
        num_labels: Number of classification labels.
    """

    output_dir = Path(output_dir).resolve()

    tokenizer_dir = output_dir / "tokenizer"
    pytorch_dir = output_dir / "pytorch_base"
    onnx_dir = output_dir / "model"

    for directory in (tokenizer_dir, pytorch_dir, onnx_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ONNX MODEL DOWNLOAD AND EXPORT")
    print("=" * 65)
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"ONNX Runtime version: {ort.__version__}")
    print()

    # ---------------------------------------------------------
    # 1. Download and save the tokenizer
    # ---------------------------------------------------------
    print("[1/5] Downloading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
    )

    tokenizer.save_pretrained(str(tokenizer_dir))

    print(f"Tokenizer saved: {tokenizer_dir}")

    # ---------------------------------------------------------
    # 2. Download the PyTorch classification model
    # ---------------------------------------------------------
    print("\n[2/5] Loading PyTorch classification model...")

    pytorch_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    pytorch_model.eval()
    pytorch_model.save_pretrained(str(pytorch_dir))

    print(f"PyTorch model saved: {pytorch_dir}")

    # ---------------------------------------------------------
    # 3. Export to ONNX
    # ---------------------------------------------------------
    print("\n[3/5] Exporting model to ONNX...")

    try:
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            str(pytorch_dir),
            export=True,
        )

        ort_model.save_pretrained(str(onnx_dir))
        tokenizer.save_pretrained(str(onnx_dir))

        print(f"ONNX model saved: {onnx_dir}")

    except Exception as exc:
        print("\nONNX export failed.")
        print(f"Error: {exc}")
        print(
            "\nCheck your Optimum and optimum-onnx versions. "
            "If the error mentions 'from_transformers', "
            "it likely indicates an incompatible package installation."
        )
        raise

    # ---------------------------------------------------------
    # 4. Test the exported model
    # ---------------------------------------------------------
    
    print("\n[4/5] Testing ONNX inference...")

    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(onnx_dir),
    )

    test_text = "This is a sample sentence for testing the model."

    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    # ONNX Runtime inference does not require model.eval()
    with torch.no_grad():
        outputs = ort_model(**inputs)

    print(f"Test input: {test_text}")
    print(f"Logits shape: {tuple(outputs.logits.shape)}")
    print(f"Predicted label ID: {outputs.logits.argmax(dim=-1).item()}")

    # ---------------------------------------------------------
    # 5. Verify saved ONNX files
    # ---------------------------------------------------------
    print("\n[5/5] Checking saved files...")

    onnx_files = list(onnx_dir.glob("*.onnx"))

    if not onnx_files:
        raise FileNotFoundError(
            f"No ONNX files were found in {onnx_dir}"
        )

    for file_path in onnx_files:
        print(f"ONNX file: {file_path.name}")
        print(f"File size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")

    print("\n" + "=" * 65)
    print("MODEL DOWNLOAD AND EXPORT COMPLETED SUCCESSFULLY")
    print("=" * 65)

    return {
        "tokenizer_dir": str(tokenizer_dir),
        "pytorch_dir": str(pytorch_dir),
        "onnx_dir": str(onnx_dir),
        "onnx_files": [str(path) for path in onnx_files],
    }


if __name__ == "__main__":
    try:
        result = download_onnx_model(
            model_name="microsoft/deberta-v3-small",
            output_dir="./models/onnx_models/deberta-v3-small",
            num_labels=2,
        )

        print("\nSaved model locations:")
        for name, path in result.items():
            print(f"{name}: {path}")

    except Exception:
        print("\nThe download/export process failed:")
        traceback.print_exc()
        sys.exit(1)
