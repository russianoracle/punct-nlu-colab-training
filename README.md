# PunctNLU CUDA Training on Google Colab

Train Russian punctuation restoration model on **GPU** (much faster than local M2).

## Quick Start

### 1️⃣ Open Colab Notebook

[👉 **CLICK HERE** to open in Colab](https://colab.research.google.com/github/russianoracle/punct-nlu-colab-training/blob/main/PunctNLU_Colab_Training.ipynb)

### 2️⃣ Prepare Training Corpus

The notebook needs `sentences.csv` (710MB, 1.194M Russian sentences from Tatoeba).

**Option A: Upload in Colab (Recommended)**
- Run first 2 cells in notebook
- Click left sidebar → Upload → select `sentences.csv`
- Wait for upload (~2-3 min for 710MB over slow internet)

**Option B: Use Google Drive**
1. Upload `sentences.csv` to your Google Drive (same account as Colab)
2. In notebook, copy from Drive:
```python
!cp /content/drive/MyDrive/sentences.csv ./sentences.csv
```

**Option C: Download from Release (if available)**
- Check GitHub Releases for corpus artifact

### 3️⃣ Run Training

Execute cells in order:
1. **Setup & Install** — installs PyTorch, transformers, etc.
2. **Mount Drive** — optional, for corpus access
3. **Configuration** — CUDA-optimized settings (batch_size=32)
4. **Data Loading** — reads corpus (takes ~1 min)
5. **Training** — 3 epochs on GPU (~2-4 hours depending on GPU type)
6. **Download Results** — download trained `best.pt` model

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

## Troubleshooting

**Q: "Corpus not found"**
- Upload `sentences.csv` to Colab file browser (left sidebar)
- Run cell to check: `Path("./sentences.csv").exists()`

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