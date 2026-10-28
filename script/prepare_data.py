import json
import os
from typing import List, Dict
from sklearn.model_selection import train_test_split
import pandas as pd

class DataGenerator:
    def __init__(self, output_dir="./data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.label_map = {
            "FAST": 0,
            "CAPABLE": 1,
            "EXPERT": 2
        }
        self.label_inverse = {v: k for k, v in self.label_map.items()}
    
    def load_from_csv(self, csv_path: str) -> List[Dict]:
        """
        Load training data from CSV with columns: text, label
        label can be numeric (0,1,2) or string (FAST, CAPABLE, EXPERT)
        """
        df = pd.read_csv(csv_path)
        data = []
        
        for _, row in df.iterrows():
            text = row['text'].strip()
            label = row['label']
            
            # Convert string labels to numbers if needed
            if isinstance(label, str):
                label_id = self.label_map.get(label.upper())
                label_name = label.upper()
            else:
                label_id = int(label)
                label_name = self.label_inverse.get(label_id, "UNKNOWN")
            
            if label_id is not None:
                data.append({
                    "text": text,
                    "label": label_id,
                    "label_name": label_name
                })
        
        return data
    
    def load_from_jsonl(self, jsonl_path: str) -> List[Dict]:
        """Load and normalize JSONL records."""
        data = []
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number} of {jsonl_path}") from error
                data.append(self._normalize_record(item))
        return data

    def load_from_json(self, json_path: str) -> List[Dict]:
        """Load grouped JSON files containing FAST, CAPABLE, and EXPERT arrays."""
        with open(json_path, "r", encoding="utf-8-sig") as file:
            grouped_data = json.load(file)

        if not isinstance(grouped_data, dict):
            raise ValueError(f"Expected a label-grouped object in {json_path}")

        data = []
        for label_name, records in grouped_data.items():
            if label_name not in self.label_map:
                continue
            if not isinstance(records, list):
                raise ValueError(f"Expected a list for label {label_name} in {json_path}")
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object record for label {label_name} in {json_path}")
                item = dict(record)
                item.setdefault("label", label_name)
                data.append(self._normalize_record(item))
        return data

    def _normalize_record(self, item: Dict) -> Dict:
        """Convert supported records to the format expected by training."""
        text = item.get("text", item.get("prompt"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every record must contain a non-empty text or prompt")

        raw_label = item.get("label")
        if isinstance(raw_label, str):
            label_name = raw_label.upper()
            label_id = self.label_map.get(label_name)
        else:
            label_id = raw_label
            label_name = self.label_inverse.get(label_id)

        if label_id not in self.label_inverse:
            raise ValueError(f"Unsupported label: {raw_label!r}")

        return {"text": text.strip(), "label": label_id, "label_name": label_name}
    
    def merge_datasets(self, file_paths: List[str], output_name: str = "merged"):
        """
        Merge multiple dataset files into one
        Supports CSV and JSONL formats
        """
        all_data = []
        
        for file_path in file_paths:
            print(f"Loading {file_path}...")
            extension = os.path.splitext(file_path)[1].lower()
            if extension == ".csv":
                all_data.extend(self.load_from_csv(file_path))
            elif extension == ".jsonl":
                all_data.extend(self.load_from_jsonl(file_path))
            elif extension == ".json":
                all_data.extend(self.load_from_json(file_path))
            else:
                raise ValueError(f"Unsupported dataset format: {file_path}")
        
        # Remove duplicates based on text
        unique_texts = set()
        deduplicated = []
        for item in all_data:
            text = item["text"]
            if text not in unique_texts:
                unique_texts.add(text)
                deduplicated.append(item)
        
        print(f"Merged {len(all_data)} samples, deduplicated to {len(deduplicated)}")
        
        # Save merged dataset
        output_path = os.path.join(self.output_dir, f"{output_name}.jsonl")
        with open(output_path, 'w') as f:
            for item in deduplicated:
                f.write(json.dumps(item) + '\n')
        
        return deduplicated
    
    def create_train_val_test_split(self, data: List[Dict], 
                                    train_ratio=0.7, 
                                    val_ratio=0.15,
                                    test_ratio=0.15,
                                    output_prefix="dataset"):
        """
        Split data into train/val/test sets
        Ensures balanced class distribution
        """
        # First split: train and temp (val + test)
        train, temp = train_test_split(
            data, 
            test_size=(1-train_ratio),
            random_state=42,
            stratify=[d['label'] for d in data]
        )
        
        # Second split: val and test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val, test = train_test_split(
            temp,
            test_size=1-val_ratio_adjusted,
            random_state=42,
            stratify=[d['label'] for d in temp]
        )
        
        # Save splits
        splits = {
            'train': train,
            'validation': val,
            'test': test
        }
        
        for split_name, split_data in splits.items():
            path = os.path.join(self.output_dir, f"{output_prefix}_{split_name}.jsonl")
            with open(path, 'w') as f:
                for item in split_data:
                    f.write(json.dumps(item) + '\n')
            
            # Class distribution
            label_counts = {}
            for item in split_data:
                label = item['label']
                label_counts[self.label_inverse[label]] = label_counts.get(self.label_inverse[label], 0) + 1
            
            print(f"[OK] {split_name.capitalize()}: {len(split_data)} samples")
            print(f"  Distribution: {label_counts}")
        
        return splits

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    output_dir = os.path.join(data_dir, "processed")
    generator = DataGenerator(output_dir=output_dir)
    
    # Example 1: Load and split single CSV
    # data = generator.load_from_csv("./data/raw/training_data.csv")
    # generator.create_train_val_test_split(data)
    
    # Example 2: Merge multiple datasets
    merged_data = generator.merge_datasets(
        file_paths=[
            os.path.join(data_dir, "training_data_calculations.json"),
            os.path.join(data_dir, "training_data_general_topics.json"),
            os.path.join(data_dir, "training_dataset_improved.json"),
        ],
        output_name="round2_merged"
    )
    
    generator.create_train_val_test_split(
        merged_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        output_prefix="round2"
    )