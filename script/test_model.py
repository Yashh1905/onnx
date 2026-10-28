import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

model_path = "models/exported_onnx/deberta-round2/model.onnx"
tokenizer_path = "models/exported_onnx/deberta-round2"

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
labels = ["FAST", "CAPABLE", "EXPERT"]

user_text = input("Enter your query: ")

inputs = tokenizer(
    user_text,
    return_tensors="np",
    padding="max_length",
    truncation=True,
    max_length=128
)

ort_inputs = {
    input_info.name: inputs[input_info.name].astype(np.int64)
    for input_info in session.get_inputs()
    if input_info.name in inputs
}

outputs = session.run(None, ort_inputs)
logits = outputs[0][0]

probabilities = np.exp(logits - np.max(logits))
probabilities = probabilities / probabilities.sum()

prediction = int(np.argmax(probabilities))

print(f"\nInput: {user_text}")
print(f"Prediction: {labels[prediction]}")
print(f"Confidence: {probabilities[prediction]:.2%}")
print("Scores:", dict(zip(labels, probabilities.round(4))))