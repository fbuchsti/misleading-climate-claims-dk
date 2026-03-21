import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, precision_score, recall_score
import json

# -------- CONFIG --------
MODEL_NAME = "Maltehb/danish-bert-botxo"
TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH = "data/processed/val.parquet"

MAX_LEN = 385
BATCH_SIZE = 4
EPOCHS = 5

# -------- LOAD DATA --------
train_df = pd.read_parquet(TRAIN_PATH)
val_df = pd.read_parquet(VAL_PATH)

train_df = train_df.sample(n=20, random_state=42)
val_df   = val_df.sample(n=10, random_state=42)

# define target column
label_cols = sorted([c for c in train_df.columns if c.startswith("label_")])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# -------- DATASET --------
class ClimateDataset(Dataset):
    def __init__(self, df, tokenizer, label_cols, mode="head"):
        self.texts = df["ArticleText"].tolist()
        self.labels = df[label_cols].values
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        tokens = self.tokenizer(
            text,
            truncation=False,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"][0]

        # slicing
        if len(input_ids) > MAX_LEN:
            if self.mode == "head":
                input_ids = input_ids[:MAX_LEN]
            elif self.mode == "tail":
                input_ids = input_ids[-MAX_LEN:]
            elif self.mode == "mid":
                start = (len(input_ids) - MAX_LEN) // 2
                input_ids = input_ids[start:start + MAX_LEN]

        attention_mask = torch.ones(len(input_ids))

        # padding
        pad_len = MAX_LEN - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len)])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }


# -------- TRAINING METRICS --------
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    accuracy = (preds == labels).mean()

    return {"accuracy": accuracy}


# -------- FINAL EVALUATION --------
def evaluate_model(trainer, dataset):
    output = trainer.predict(dataset)

    logits = output.predictions
    labels = output.label_ids

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    return {
        "loss": output.metrics["test_loss"],
        "accuracy": (preds == labels).mean(),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0),
    }


# -------- RUN --------
all_results = {}

for mode in ["head", "mid", "tail"]:

    print(f"\n--- Training {mode} ---")

    train_dataset = ClimateDataset(train_df, tokenizer, label_cols, mode)
    val_dataset   = ClimateDataset(val_df, tokenizer, label_cols, mode)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_cols),
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=f"models/bert_{mode}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # train
    trainer.train()

    # final evaluation
    results = evaluate_model(trainer, val_dataset)

    print(f"\nFinal results ({mode}):")
    print(results)

    # save results
    with open(f"results_{mode}.json", "w") as f:
        json.dump(results, f, indent=2)

    # save model
    model.save_pretrained(f"models/bert_{mode}")
    tokenizer.save_pretrained(f"models/bert_{mode}")

    all_results[mode] = results


print("\n=== SUMMARY ===")
for k, v in all_results.items():
    print(k, v)