import os
import json
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from transformers import DataCollatorWithPadding

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentClassificationDataset(Dataset):
    """Custom dataset for intent classification"""
    
    def __init__(self, jsonl_file, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
        'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
        'f1': f1_score(labels, predictions, average='weighted', zero_division=0)
    }

def train_model(
    base_model_dir="./models/pytorch_models/deberta-round2",
    train_data="./data/processed/round2_train.jsonl",
    val_data="./data/processed/round2_validation.jsonl",
    test_data="./data/processed/round2_test.jsonl",
    output_dir="./models/pytorch_models/deberta-fine-tuned",
    num_labels=3,
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    max_length=128,
    warmup_steps=500
):
    """
    Fine-tune the model on labeled data
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model from {base_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_dir,
        num_labels=num_labels
    )
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = IntentClassificationDataset(train_data, tokenizer, max_length)
    val_dataset = IntentClassificationDataset(val_data, tokenizer, max_length)
    test_dataset = IntentClassificationDataset(test_data, tokenizer, max_length)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir='./logs/training_logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        gradient_accumulation_steps=1,
        fp16=False,  # Use mixed precision on GPU
        dataloader_num_workers=0,
        group_by_length=True,

        use_cpu=True,
         
        bf16=False,

       
       report_to="none",
       seed=42,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.001
            )
        ]
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate on validation and held-out test sets
    logger.info("Evaluating on validation set...")
    metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="validation")
    logger.info(f"Validation metrics: {metrics}")

    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    logger.info(f"Test metrics: {test_metrics}")
    metrics.update(test_metrics)

    return model, tokenizer, metrics

if __name__ == "__main__":
    model, tokenizer, metrics = train_model(
        base_model_dir="microsoft/deberta-v3-small",
        train_data="./data/processed/round2_train.jsonl",
        val_data="./data/processed/round2_validation.jsonl",
        test_data="./data/processed/round2_test.jsonl",
        output_dir="./models/pytorch_models/deberta-round2",
        num_labels=3,
        num_epochs=3,
        batch_size=16,
        learning_rate=1e-5,
        warmup_steps=800
    )
    
    print("\n=== Training Complete ===")
    print(f"Model saved to: ./models/pytorch_models/deberta-round2")
    print(f"Validation F1 Score: {metrics.get('validation_f1', 'N/A'):.4f}")
