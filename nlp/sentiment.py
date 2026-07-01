"""
Fine-tuned BERT (DistilBERT) pipeline for negotiation tone classification.

8 signal classes: hesitant, urgent, aggressive, bluffing, collaborative,
dismissive, enthusiastic, noncommittal.

Uses DistilBERT instead of full BERT for faster fine-tuning/inference --
swap base_model to "bert-base-uncased" if you want the heavier version.

HONEST NOTE: shipped with only the small hand-labeled seed set
(nlp/seed_training_data.py, ~10/class). This is enough to validate the
training pipeline runs correctly end-to-end, NOT enough data for
reliable real-world tone detection. Expand the dataset (real messages
from Phase 0, or LLM-generated synthetic examples) before trusting this
in the live coaching flow. Target from the blueprint: Sentiment F1 0.80+
-- that requires real training data at meaningful scale, not this seed set.

Run training: python -m nlp.sentiment
"""

import os
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from nlp.seed_training_data import SEED_DATA, LABELS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
BASE_MODEL = "distilbert-base-uncased"

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def build_dataset(test_size: float = 0.2, seed: int = 42):
    # Start with real + hand-written seed examples
    all_examples = list(SEED_DATA)

    # Automatically merge in Groq-generated data if it exists
    generated_path = os.path.join(os.path.dirname(__file__), "generated_training_data.py")
    if os.path.exists(generated_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("generated_training_data", generated_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        generated = getattr(mod, "GENERATED_DATA", [])
        all_examples.extend(generated)
        print(f"Loaded {len(generated)} generated examples + "
              f"{len(SEED_DATA)} seed examples = {len(all_examples)} total")
    else:
        print(f"No generated data found. Using {len(SEED_DATA)} seed examples only.")
        print("Run `python -m nlp.generate_data` first for much better accuracy.")

    texts = [t for t, _ in all_examples]
    labels = [LABEL2ID[l] for _, l in all_examples]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    return train_ds, val_ds


def train_sentiment_model(epochs: int = 8, batch_size: int = 8, output_dir: str = MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds, val_ds = build_dataset()

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=5,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
    )

    print(f"Fine-tuning {BASE_MODEL} on {len(train_ds)} train / {len(val_ds)} val examples "
          f"across {len(LABELS)} tone classes...")
    trainer.train()

    metrics = trainer.evaluate()
    print("\nFinal validation metrics:", metrics)
    print(f"(Target from blueprint: F1 0.80+ -- seed dataset is too small to "
          f"reliably hit this; treat this run as a pipeline validation, not "
          f"a production benchmark.)")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model + tokenizer to {output_dir}")

    return model, tokenizer, metrics


class ToneClassifier:
    """
    Inference wrapper. Loads a fine-tuned model from MODEL_DIR if present;
    otherwise raises with instructions to train first.

    This is the integration point referenced in env/negotiation_env.py's
    `sentiment_score` observation field (currently a 0.5 placeholder) --
    once wired into the live coaching API (Week 11-12), each incoming
    client message gets scored here before being added to the state vector.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"No trained model found at {model_dir}. Run "
                f"`python -m nlp.sentiment` to fine-tune one first."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """
        Returns: {
            "dominant_tone": str,
            "confidence": float,
            "scores": {tone: probability, ...}   # all 8 classes
        }
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        scores = {ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)}
        dominant_idx = int(np.argmax(probs))

        return {
            "dominant_tone": ID2LABEL[dominant_idx],
            "confidence": round(probs[dominant_idx], 4),
            "scores": scores,
        }

    def sentiment_score_for_state_vector(self, text: str) -> float:
        """
        Collapses the 8-class tone distribution into a single 0-1 scalar
        for the RL state vector's sentiment_score field. Simple heuristic:
        "favorable" tones (collaborative, enthusiastic) push toward 1,
        "unfavorable" tones (aggressive, dismissive, bluffing) push toward 0,
        neutral/ambiguous tones sit near 0.5.
        """
        result = self.predict(text)
        favorable = {"collaborative", "enthusiastic"}
        unfavorable = {"aggressive", "dismissive", "bluffing"}

        scores = result["scores"]
        favorable_mass = sum(scores[t] for t in favorable)
        unfavorable_mass = sum(scores[t] for t in unfavorable)

        # map [-1, 1] (unfavorable - favorable) onto [0, 1]
        raw = favorable_mass - unfavorable_mass
        return round((raw + 1) / 2, 4)


if __name__ == "__main__":
    train_sentiment_model()
