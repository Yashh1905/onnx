import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
import logging

logger = logging.getLogger(__name__)

def export_to_onnx(
    pytorch_model_dir="./models/pytorch_models/deberta-round2",
    output_dir="./models/exported_onnx/deberta-round2",
    quantize=True,
    optimize=True
):
    """
    Convert fine-tuned PyTorch model to ONNX format
    
    Args:
        pytorch_model_dir: Directory containing fine-tuned PyTorch model
        output_dir: Directory to save ONNX model
        quantize: Whether to apply quantization for smaller model size
        optimize: Whether to optimize the ONNX model
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Loading PyTorch model from {pytorch_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(pytorch_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(pytorch_model_dir)
    
    # Export directly with PyTorch using the tokenizer inputs supported by the model.
    logger.info("Exporting to ONNX format...")
    export_pytorch_to_onnx_manual(model, tokenizer, pytorch_model_dir, output_dir)

    # Quantization (optional but recommended)
    if quantize:
        logger.info("Quantizing ONNX model...")
        quantize_onnx_model(
            model_path=os.path.join(output_dir, "model.onnx"),
            quantized_output_path=os.path.join(output_dir, "model_quantized.onnx")
        )
    
    # Verify the exported model
    logger.info("Verifying exported ONNX model...")
    verify_onnx_model(output_dir, tokenizer)
    
    return output_dir

def export_pytorch_to_onnx_manual(model, tokenizer, model_dir, output_dir):
    """Export a sequence-classification model with dynamic batch and sequence axes."""
    import json

    class LogitsWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, input_ids, attention_mask):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogitsWrapper(model.to(device).eval())
    sample = tokenizer(
        "What is the weather today?",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )
    input_ids = sample["input_ids"].to(device)
    attention_mask = sample["attention_mask"].to(device)
    onnx_model_path = os.path.join(output_dir, "model.onnx")

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(model.base_model.config.to_dict(), file, indent=2)
    tokenizer.save_pretrained(output_dir)
    logger.info("ONNX model exported to %s", onnx_model_path)

def quantize_onnx_model(model_path, quantized_output_path):
    """
    Apply dynamic quantization to reduce model size and latency
    """
    try:
        quantize_dynamic(
            model_path,
            quantized_output_path,
            weight_type=QuantType.QInt8
        )
        logger.info(f"✓ Quantized model saved to {quantized_output_path}")
        
        # Compare sizes
        original_size = os.path.getsize(model_path) / 1024 / 1024
        quantized_size = os.path.getsize(quantized_output_path) / 1024 / 1024
        reduction = (1 - quantized_size / original_size) * 100
        
        logger.info(f"Model size: {original_size:.2f}MB → {quantized_size:.2f}MB ({reduction:.1f}% reduction)")
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")

def verify_onnx_model(model_dir, tokenizer):
    """
    Test the exported ONNX model with sample inference
    """
    import onnxruntime as rt
    import numpy as np
    
    onnx_model_path = os.path.join(model_dir, "model.onnx")
    
    # Load ONNX model
    sess = rt.InferenceSession(onnx_model_path)
    
    # Test inference
    test_text = "What is the weather today?"
    inputs = tokenizer(
        test_text,
        return_tensors="np",
        padding="max_length",
        max_length=128,
        truncation=True
    )
    
    ort_inputs = {
        name: inputs[name].astype(np.int64)
        for name in (input.name for input in sess.get_inputs())
        if name in inputs
    }
    
    ort_outs = sess.run(None, ort_inputs)
    logits = ort_outs[0]
    
    label_names = ["FAST", "CAPABLE", "EXPERT"]
    pred_label = label_names[np.argmax(logits)]
    confidence = np.max(np.exp(logits[0]) / np.sum(np.exp(logits[0])))
    
    logger.info(f"✓ ONNX inference successful!")
    logger.info(f"  Test input: '{test_text}'")
    logger.info(f"  Predicted: {pred_label} (confidence: {confidence:.4f})")

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    export_to_onnx(
        pytorch_model_dir=os.path.join(project_dir, "models", "pytorch_models", "deberta-round2"),
        output_dir=os.path.join(project_dir, "models", "exported_onnx", "deberta-round2"),
        quantize=True,
        optimize=True
    )
