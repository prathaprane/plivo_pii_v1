# src/train.py

import os
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")  # real data
    ap.add_argument("--train_synthetic", default="data/train_synthetic.jsonl")  # synthetic data
    ap.add_argument("--dev", default="data/dev.jsonl")  # kept for compatibility
    ap.add_argument("--out_dir", default="out")

    # TRAINING HYPERPARAMS (for decent-sized data)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # how many times to repeat the tiny real set in the mix
    ap.add_argument("--oversample_real", type=int, default=100)

    return ap.parse_args()


class CombinedDataset(Dataset):
    """
    Simple wrapper over a list of items (dicts) produced by PIIDataset.
    """

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Using device: {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 1. Load real + synthetic datasets
    real_ds = PIIDataset(
        args.train,
        tokenizer,
        LABELS,
        max_length=args.max_length,
        is_train=True,
    )
    synth_ds = PIIDataset(
        args.train_synthetic,
        tokenizer,
        LABELS,
        max_length=args.max_length,
        is_train=True,
    )

    print(f"Real train samples: {len(real_ds)}")
    print(f"Synthetic train samples: {len(synth_ds)}")

    # 2. Combine: synthetic + oversampled real
    items = []

    # all synthetic
    items.extend(synth_ds.items)

    # repeat real data many times so it isn't ignored
    for _ in range(args.oversample_real):
        items.extend(real_ds.items)

    train_ds = CombinedDataset(items)
    print(f"Total training samples after combining: {len(train_ds)}")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    # 3. Model
    model = create_model(args.model_name)
    model.to(args.device)
    model.train()

    # 4. Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_dl) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 5. Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # 6. Save model and tokenizer
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved model + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
