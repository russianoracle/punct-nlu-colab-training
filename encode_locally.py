#!/usr/bin/env python3
"""
Local encoding script for M2 Mac
Encode samples locally, save to file, upload to GitHub
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional, List, Tuple
import re
from transformers import AutoTokenizer
from tqdm import tqdm
import json

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_ID = "cointegrated/rubert-tiny2"
CORPUS_PATH = Path("/Users/artemgusarov/Downloads/sentences.csv")
OUTPUT_DIR = Path("/Users/artemgusarov/Downloads/encoded_samples")
SEQ_LEN = 64
BATCH_SIZE = 32

PUNCT_CHARS = {",": 1, ".": 2, "!": 2, "…": 2, ";": 2, "?": 3}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EncodedSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


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


def word_labels_vec(words: List[str]) -> np.ndarray:
    words_arr = np.array(words, dtype=object)
    punct_classes = punct_class_vec(words_arr)
    cap_modes = cap_mode_vec(words_arr)
    return punct_classes * 3 + cap_modes


def ortho_to_norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word, flags=re.UNICODE).lower()


def iter_tatoeba(path: Path) -> Iterator[Tuple[List[str], List[int]]]:
    """Read Tatoeba corpus"""
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


def encode_sample(norm_words: List[str], word_labels: List[int], tokenizer) -> Optional[EncodedSample]:
    """Encode ONE sample"""
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

        sub_ids = tokenizer.encode(word, add_special_tokens=False) or [tokenizer.unk_token_id]

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

    return EncodedSample(
        input_ids=torch.from_numpy(ids).long(),
        attention_mask=torch.from_numpy(mask).long(),
        labels=torch.from_numpy(labels).long(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 80)
    print("LOCAL ENCODING ON M2 — PunctNLU Samples")
    print("=" * 80)

    if not CORPUS_PATH.exists():
        print(f"✗ Corpus not found at {CORPUS_PATH}")
        print("  Download from: https://downloads.tatoeba.org/exports/sentences.csv")
        return

    # Load corpus
    print(f"\n📂 Loading corpus from {CORPUS_PATH}...")
    corpus_data = list(iter_tatoeba(CORPUS_PATH))
    print(f"✓ Loaded {len(corpus_data):,} sentences")

    # Load tokenizer
    print(f"\n🔤 Loading tokenizer: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Encode
    print(f"\n⚙️  Encoding {len(corpus_data):,} samples...")
    print(f"   (Sequential on M2 CPU: ~30-60 minutes)")
    print(f"   (Saving progress every 100k samples)\n")

    encoded_samples = []
    checkpoint_interval = 100000

    for idx, (norm_words, word_labels) in enumerate(tqdm(corpus_data, desc="Encoding")):
        sample = encode_sample(norm_words, word_labels, tokenizer)
        if sample:
            encoded_samples.append(sample)

        # Save checkpoint
        if (idx + 1) % checkpoint_interval == 0:
            checkpoint_file = OUTPUT_DIR / f"checkpoint_{idx + 1}.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(encoded_samples, f)
            print(f"\n  💾 Checkpoint at {idx + 1}: {checkpoint_file.stat().st_size / 1e9:.2f}GB")

    # Final save
    print(f"\n✅ Encoding complete!")
    output_file = OUTPUT_DIR / "encoded_samples.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(encoded_samples, f)

    size_gb = output_file.stat().st_size / 1e9
    print(f"✓ Saved {len(encoded_samples):,} samples to {output_file}")
    print(f"  Size: {size_gb:.2f}GB")

    # Save metadata
    metadata = {
        "total_samples": len(encoded_samples),
        "file_size_gb": size_gb,
        "seq_len": SEQ_LEN,
        "model_id": MODEL_ID,
    }
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n📋 Next steps:")
    print(f"   1. Upload {output_file} to GitHub releases")
    print(f"   2. Colab notebook will download from GitHub during training")
    print(f"\n" + "=" * 80)


if __name__ == "__main__":
    main()