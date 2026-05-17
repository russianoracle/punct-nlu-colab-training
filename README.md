# PunctNLU CUDA Training on Google Colab

Train Russian punctuation restoration model on **GPU** (much faster than local M2).

## Quick Start

### 1️⃣ Open Colab Notebook

[👉 **CLICK HERE** to open in Colab](https://colab.research.google.com/github/russianoracle/punct-nlu-colab-training/blob/main/PunctNLU_Colab_Training.ipynb)

### 2️⃣ Prepare Training Corpus

The notebook needs `sentences.csv` (710MB, 1.194M Russian sentences from Tatoeba).

**Corpus Detection Priority:**
1. Check `/content/sentences.csv` (if already in Colab)
2. Check Google Drive: `/content/drive/MyDrive/PunctNLU/sentences.csv`
3. Prompt for upload if not found

**Option A: Direct Upload in Colab (Fastest)**
- Run first cell in notebook
- If corpus not found, click "Choose Files" button
- Select `sentences.csv` from your computer
- Colab uploads it automatically (~5-10 min)

**Option B: Pre-upload to Google Drive**
1. Upload `sentences.csv` to Google Drive (your account)
2. Create folder: `PunctNLU` in `MyDrive`
3. Place file there before running notebook
4. Notebook will find it automatically

**Option C: Use Standalone Script**
```bash
# In Colab (Runtime → Run all cells)
# OR locally if you have GPU:
python run_training.py
```
Script auto-detects corpus at `/content/sentences.csv` or `./sentences.csv`

### 3️⃣ Run Training

**Method 1: Interactive Notebook**
Execute cells in order:
1. **Setup & Install** — installs PyTorch, transformers, etc.
2. **Mount Drive + Corpus Check** — auto-detects `/content/sentences.csv` or Google Drive, uploads if needed
3. **Configuration** — CUDA-optimized settings (batch_size=32)
4. **Data Loading** — reads corpus (takes ~1 min)
5. **Training** — 3 epochs on GPU (~2-4 hours depending on GPU type)
6. **Download Results** — download trained `best.pt` model

**Method 2: Standalone Script (Skip Mount Cell)**
If corpus already at `/content/sentences.csv`:
```python
# In Colab cell:
!python run_training.py
```
Or locally with GPU:
```bash
python run_training.py
```

The script auto-verifies corpus and shows status before training.

## Performance

| Device | Time (3 epochs) | Batch Size | GPU Memory |
|--------|-----------------|-----------|-----------|
| **Colab T4** | 4-5h | 32 | 15GB |
| **Colab A100** | 1.5-2h | 32 | 40GB |
| M2 (Local) | 12-14h | 16 | 8GB |

## Model Architecture

- **Base:** rubert-tiny2 (29.2M parameters)
- **Heads:** 
  - Head A1: Punctuation restoration (12 classes: none, comma, period, question)
  - Head A2: Actionable classification (2 classes)
- **Training:** AdamW, LinearLR scheduler, 3 epochs
- **Output:** `best.pt` (checkpoint with best validation loss)

## Files

- `PunctNLU_Colab_Training.ipynb` — Full training notebook
- `sentences.csv` — Training corpus (not in repo, upload yourself)
- `checkpoints/` — Output directory (created during training)

## Next Steps

After training:
1. Download `best.pt` from Colab
2. Export to CoreML for iOS/macOS
3. Integrate into AIssistant app

## Verify Corpus Before Training

Run this in Colab to check where corpus is:
```python
from pathlib import Path

# Check all possible locations
locations = [
    "/content/sentences.csv",
    "/content/drive/MyDrive/PunctNLU/sentences.csv",
    "./sentences.csv",
]

for loc in locations:
    p = Path(loc)
    if p.exists():
        size_mb = p.stat().st_size / 1e6
        print(f"✓ Found: {loc} ({size_mb:.0f} MB)")
    else:
        print(f"✗ Not found: {loc}")
```

## Troubleshooting

**Q: "Corpus not found"**
- Check above locations
- Upload `sentences.csv` via Colab file browser (left sidebar: Files → Upload)
- Or run notebook Cell 1 which auto-detects and prompts for upload

**Q: "CUDA out of memory"**
- Reduce `BATCH_SIZE` from 32 to 16 in configuration cell
- Clear cache: `torch.cuda.empty_cache()`

**Q: "Model not training (loss stays same)"**
- Check HEAD A1 (punct) is being trained, not just HEAD A2 (classify)
- Learning rate: try 1e-5 or 1e-3

## Specs

- **Framework:** PyTorch 2.0+, transformers 4.30+
- **Language:** Python 3.11
- **GPU Required:** Yes (CUDA 11.8+, Colab has this)

---

Created for [AIssistant](https://github.com/russianoracle/AIssistant) punctuation restoration.