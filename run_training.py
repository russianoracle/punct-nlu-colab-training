#!/usr/bin/env python3
"""
Standalone training script for PunctNLU in Colab.
Run: python run_training.py
"""
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_ID      = "cointegrated/rubert-tiny2"
SEQ_LEN       = 64
NUM_LABELS    = 12
BATCH_SIZE    = 32
NUM_EPOCHS    = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR    = Path("./checkpoints/punct_nlu")

# Find corpus - ensure it's Path object
corpus_local = Path("/content/sentences.csv")
corpus_cwd = Path("./sentences.csv")

if corpus_local.exists():
    CORPUS_PATH = corpus_local
elif corpus_cwd.exists():
    CORPUS_PATH = corpus_cwd
else:
    CORPUS_PATH = corpus_local  # Default for error message

PUNCT_CHARS = {",": 1, ".": 2, "!": 2, "…": 2, ";": 2, "?": 3}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── VERIFY CORPUS ───────────────────────────────────────────────────────────
log.info("=" * 80)
log.info("Checking corpus...")
log.info("=" * 80)

if CORPUS_PATH.exists():
    size_mb = CORPUS_PATH.stat().st_size / 1e6
    log.info(f"✓ Corpus found: {CORPUS_PATH} ({size_mb:.0f} MB)")
else:
    log.error(f"✗ Corpus not found at {CORPUS_PATH}")
    log.error("Checked paths:")
    log.error("  - /content/sentences.csv (Colab /content)")
    log.error("  - ./sentences.csv (current directory)")
    sys.exit(1)

# ── LABELS ──────────────────────────────────────────────────────────────────

def cap_mode_vec(words: np.ndarray) -> np.ndarray:
    modes = np.zeros(len(words), dtype=np.int32)
    for i, word in enumerate(words):
        clean = re.sub(r"[^\w]", "", word, flags=re.UNICODE)
        if clean:
            alpha = [c for c in clean if c.isalpha()]
            if alpha:
                if all(c.isupper() for c in alpha):
                    modes[i] = 2
                elif alpha[0].isupper():
                    modes[i] = 1
    return modes


def punct_class_vec(words: np.ndarray) -> np.ndarray:
    classes = np.zeros(len(words), dtype=np.int32)
    for i, word in enumerate(words):
        if word:
            last = word[-1]
            classes[i] = PUNCT_CHARS.get(last, 0)
    return classes


def word_labels_vec(words: list[str]) -> np.ndarray:
    words_arr = np.array(words, dtype=object)
    punct_classes = punct_class_vec(words_arr)
    cap_modes = cap_mode_vec(words_arr)
    return punct_classes * 3 + cap_modes


def ortho_to_norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word, flags=re.UNICODE).lower()


# ── DATA LOADER ─────────────────────────────────────────────────────────────

def iter_tatoeba(path: Path) -> Iterator[tuple[list[str], list[int]]]:
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3 or parts[1] != "rus":
                continue

            sentence = parts[2].strip()
            if len(sentence) < 5:
                continue

            words = sentence.split()
            if len(words) < 2:
                continue

            labels = word_labels_vec(words).tolist()
            norms = [ortho_to_norm(w) for w in words]

            if not any(norms):
                continue

            yield norms, labels
            count += 1
            if count % 100000 == 0:
                log.info(f"  Loaded {count} sentences...")


@dataclass
class EncodedSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def encode(norm_words: list[str], word_labels: list[int], tokenizer) -> Optional[EncodedSample]:
    ids = np.zeros(SEQ_LEN, dtype=np.int32)
    mask = np.zeros(SEQ_LEN, dtype=np.int32)
    labels = np.full(SEQ_LEN, -100, dtype=np.int32)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    ids[0] = cls_id
    mask[0] = 1
    pos = 1

    for word, lbl in zip(norm_words, word_labels):
        if pos >= SEQ_LEN - 1:
            break

        sub_ids = tokenizer.encode(word, add_special_tokens=False)
        if not sub_ids:
            sub_ids = [tokenizer.unk_token_id]

        if pos + len(sub_ids) >= SEQ_LEN:
            break

        ids[pos] = sub_ids[0]
        mask[pos] = 1
        labels[pos] = lbl
        pos += 1

        for sub_id in sub_ids[1:]:
            if pos >= SEQ_LEN:
                break
            ids[pos] = sub_id
            mask[pos] = 1
            labels[pos] = -100
            pos += 1

    if pos < SEQ_LEN:
        ids[pos] = sep_id
        mask[pos] = 1
        pos += 1

    return EncodedSample(
        input_ids=torch.from_numpy(ids).long(),
        attention_mask=torch.from_numpy(mask).long(),
        labels=torch.from_numpy(labels).long(),
    )


class PunctDataset(Dataset):
    def __init__(self, samples: list[EncodedSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids": s.input_ids,
            "attention_mask": s.attention_mask,
            "labels": s.labels,
        }


# ── MODEL ───────────────────────────────────────────────────────────────────

class PunctNLUModel(nn.Module):
    def __init__(self, model_id: str, num_labels: int = 12):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_id)
        self.dropout = nn.Dropout(0.1)
        self.punct_head = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.classify_head = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        hidden = self.dropout(hidden)

        punct_logits = self.punct_head(hidden)
        classify_logits = self.classify_head(hidden)

        return punct_logits, classify_logits


# ── TRAINING ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        punct_logits, classify_logits = model(input_ids, attention_mask)

        loss_punct = ce_loss_fn(punct_logits.view(-1, 12), labels.view(-1))

        # Binary classification: has punctuation (label > 0) or not (label == 0)
        binary_labels = (labels > 0).long()
        loss_classify = ce_loss_fn(
            classify_logits.view(-1, 2),
            binary_labels.view(-1)
        )

        loss = loss_punct + 0.5 * loss_classify
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            log.info(f"  Batch {batch_idx + 1}/{len(loader)}: loss={loss.item():.4f}")

    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            punct_logits, classify_logits = model(input_ids, attention_mask)

            loss_punct = ce_loss_fn(punct_logits.view(-1, 12), labels.view(-1))

            # Binary classification: has punctuation (label > 0) or not (label == 0)
            binary_labels = (labels > 0).long()
            loss_classify = ce_loss_fn(
                classify_logits.view(-1, 2),
                binary_labels.view(-1)
            )

            loss = loss_punct + 0.5 * loss_classify
            total_loss += loss.item()

            pred = torch.argmax(punct_logits, dim=-1)
            mask = labels >= 0
            correct += (pred[mask] == labels[mask]).sum()
            total += mask.sum()

    return total_loss / len(loader), (correct / max(total, 1)).item()


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 80)
    log.info(f"Device: {DEVICE}")
    log.info(f"Model: {MODEL_ID}")
    log.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log.info("=" * 80)

    # Load corpus
    log.info(f"\n1️⃣  Loading corpus from {CORPUS_PATH}...")
    corpus_data = list(iter_tatoeba(CORPUS_PATH))
    log.info(f"Loaded {len(corpus_data)} samples")

    # Load tokenizer
    log.info(f"\n2️⃣  Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Encode all samples
    log.info("\n3️⃣  Encoding samples...")
    encoded_samples = []
    for norm_words, labels in corpus_data:
        sample = encode(norm_words, labels, tokenizer)
        if sample:
            encoded_samples.append(sample)

    log.info(f"Encoded {len(encoded_samples)} samples")

    # Split train/val
    log.info("\n4️⃣  Splitting train/val...")
    np.random.seed(42)
    indices = np.random.permutation(len(encoded_samples))
    split = int(0.95 * len(encoded_samples))

    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = PunctDataset([encoded_samples[i] for i in train_indices])
    val_dataset = PunctDataset([encoded_samples[i] for i in val_indices])

    log.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    # Initialize model
    log.info("\n5️⃣  Building model...")
    model = PunctNLUModel(MODEL_ID).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {params:,}")

    # Training loop
    log.info("\n6️⃣  Starting training...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total_steps,
    )

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_loss, val_acc = eval_epoch(model, val_loader, DEVICE)

        elapsed = time.time() - t0
        log.info(f"Epoch {epoch}/{NUM_EPOCHS}  train={train_loss:.4f}  "
                f"val={val_loss:.4f}  acc={val_acc:.3f}  t={elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best.pt")
            log.info(f"  ✓ saved best → {OUTPUT_DIR / 'best.pt'}")

    log.info(f"\n✓ Training done. Best val loss: {best_val_loss:.4f}")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
    }, OUTPUT_DIR / "punct_nlu.pt")
    log.info(f"✓ Saved final checkpoint → {OUTPUT_DIR / 'punct_nlu.pt'}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
