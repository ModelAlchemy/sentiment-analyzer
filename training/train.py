"""
Fine-tune DistilBERT on IMDB dataset.

Usage:
    python training/train.py [--push_to_hub] [--hub_model_id YOUR_USERNAME/sentiment-distilbert]

Requirements:
    pip install transformers datasets scikit-learn torch accelerate huggingface_hub

This script:
  1. Loads IMDB 50k movie reviews from HuggingFace datasets
  2. Fine-tunes distilbert-base-uncased for binary sentiment
  3. Evaluates on test set (target AUC > 0.93)
  4. Saves model locally + optionally pushes to HuggingFace Hub
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "training" / "checkpoints"
METRICS_FILE = ROOT / "training" / "metrics.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",    default="distilbert-base-uncased")
    p.add_argument("--dataset",       default="imdb")
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--max_length",    type=int,   default=256)
    p.add_argument("--train_samples", type=int,   default=None, help="Subset for quick testing")
    p.add_argument("--push_to_hub",   action="store_true")
    p.add_argument("--hub_model_id",  default=None)
    return p.parse_args()


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="binary"),
    }


def train(args):
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer, EarlyStoppingCallback,
        )
        from datasets import load_dataset
        import torch
    except ImportError:
        print("❌ Install requirements: pip install transformers datasets torch accelerate")
        return

    print(f"=== Fine-tuning {args.model_name} on {args.dataset} ===\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = load_dataset(args.dataset)
    if args.train_samples:
        dataset["train"] = dataset["train"].select(range(args.train_samples))
        dataset["test"]  = dataset["test"].select(range(min(args.train_samples // 5, 2000)))
    print(f"  Train: {len(dataset['train']):,} | Test: {len(dataset['test']):,}")

    # ── Tokenize ──────────────────────────────────────────────────────────────
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    # ── Model ────────────────────────────────────────────────────────────────
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    # ── Training arguments ───────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy="end",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\nTraining for up to {args.epochs} epochs...")
    trainer.train()

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n=== Final Evaluation ===")
    predictions = trainer.predict(tokenized["test"])
    logits = predictions.predictions
    labels = predictions.label_ids
    proba  = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    preds  = np.argmax(logits, axis=-1)

    metrics = {
        "accuracy":  round(float(accuracy_score(labels, preds)), 4),
        "f1":        round(float(f1_score(labels, preds)), 4),
        "auc_roc":   round(float(roc_auc_score(labels, proba[:, 1])), 4),
        "model":     args.model_name,
        "dataset":   args.dataset,
        "epochs":    args.epochs,
        "train_size": len(dataset["train"]),
        "test_size":  len(dataset["test"]),
    }

    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["NEGATIVE", "POSITIVE"]))

    # Save metrics
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to {METRICS_FILE}")

    # Save model locally
    local_path = ROOT / "training" / "saved_model"
    trainer.save_model(str(local_path))
    tokenizer.save_pretrained(str(local_path))
    print(f"✅ Model saved to {local_path}")

    if args.push_to_hub:
        print(f"✅ Model pushed to HuggingFace Hub: {args.hub_model_id}")

    print("\n🎉 Done! To use your model in the Streamlit app:")
    print(f"   Change model= parameter to '{args.hub_model_id or str(local_path)}'")

    return metrics


if __name__ == "__main__":
    args = parse_args()
    train(args)
