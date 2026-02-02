# Quick Start Guide - UAV IDS

## Step-by-Step Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Install dependencies:**
   ```bash
   cd c:\Users\DELL\btp
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python -c "import torch; import pandas; import sklearn; print('All dependencies installed!')"
   ```

---

## Option 1: Run Complete Pipeline (Recommended)

Run everything in one command:

```bash
cd src
python main.py --all
```

This will:
1. Generate all 3 datasets (~30 seconds)
2. Train the LSTM model (~5-10 minutes on CPU, ~2-3 minutes on GPU)
3. Evaluate on cross-datasets (~1 minute)
4. Save all results to `models/` and `results/`

---

## Option 2: Step-by-Step Execution

### Step 1: Generate Datasets

```bash
cd src
python generate_datasets.py
```

**Expected output:**
```
Dataset-1 (Normal):           10000 samples, 0 attacks
Dataset-2 (Injection/Replay): 10000 samples, 3000 attacks
Dataset-3 (GPS/Sensor):       10000 samples, 3000 attacks
```

**Verify:** Check `data/` folder for 3 CSV files

---

### Step 2: Train the Model

```bash
python train.py
```

**Expected output:**
```
Epoch [1/50] | Train Loss: 0.2456, Train Acc: 0.8912 | Val Loss: 0.1234, Val Acc: 0.9456
...
Training complete!
```

**Verify:** Check `models/` folder for:
- `lstm_ids_best.pth`
- `lstm_ids_final.pth`
- `scaler.pkl`

**Check results:** Open `results/training_history.png`

---

### Step 3: Evaluate on Test Datasets

```bash
python evaluate.py
```

**Expected output:**
```
Dataset-2 (Injection/Replay):
  Accuracy:  0.9123
  Precision: 0.8876
  Recall:    0.9012
  F1-Score:  0.8943
  FPR:       0.0567

Dataset-3 (GPS/Sensor):
  Accuracy:  0.8745
  ...
```

**Verify:** Check `results/` folder for:
- `confusion_matrix_dataset2.png`
- `confusion_matrix_dataset3.png`
- `roc_curves.png`
- `metrics_comparison.png`
- `evaluation_report.json`

---

## Understanding the Results

### Training Plots (`training_history.png`)

- **Left plot (Loss):** Should decrease over epochs
  - If loss is flat → increase learning rate
  - If loss oscillates → decrease learning rate or batch size

- **Right plot (Accuracy):** Should increase over epochs
  - Train accuracy ~95-99% is good
  - Gap between train/val accuracy > 10% → overfitting (increase dropout)

### Confusion Matrices

```
Predicted:     Normal  Attack
Actual Normal: [8500]  [500]   ← 500 false alarms
Actual Attack: [300]   [2700]  ← 300 missed attacks
```

**Good IDS:**
- High TN (top-left): Correctly identified normal behavior
- High TP (bottom-right): Correctly detected attacks
- Low FP (top-right): Few false alarms
- Low FN (bottom-left): Few missed attacks

### ROC Curves

- **Perfect classifier:** Curve hugs top-left corner (AUC = 1.0)
- **Random classifier:** Diagonal line (AUC = 0.5)
- **Good IDS:** AUC > 0.85

### Metrics

| Metric | Good Value | Interpretation |
|--------|-----------|----------------|
| Accuracy | > 0.85 | Overall correctness |
| Precision | > 0.80 | Of predicted attacks, % real |
| Recall | > 0.85 | Of real attacks, % detected |
| F1-Score | > 0.82 | Balanced metric |
| **FPR** | **< 0.10** | **False alarm rate (critical!)** |

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

### Problem: "FileNotFoundError: dataset_1_normal.csv"

**Solution:** Run dataset generation first:
```bash
cd src
python generate_datasets.py
```

### Problem: "RuntimeError: Model file not found"

**Solution:** Train the model first:
```bash
python train.py
```

### Problem: Training is very slow

**Solutions:**
- Reduce batch size in `train.py`: `BATCH_SIZE = 32`
- Reduce window size: `WINDOW_SIZE = 5`
- Reduce epochs: `EPOCHS = 20`
- Use GPU if available (PyTorch will auto-detect CUDA)

### Problem: Low accuracy on test sets

**Solutions:**
- Train longer (increase `EPOCHS`)
- Increase model capacity (`HIDDEN_SIZE = 128`)
- Adjust classification threshold in `evaluate.py`
- Check if datasets were generated correctly

---

## Next Steps

### 1. Experiment with Hyperparameters

Edit `train.py`:
```python
WINDOW_SIZE = 20      # Try longer sequences
HIDDEN_SIZE = 128     # Try bigger model
NUM_LAYERS = 3        # Try deeper model
DROPOUT = 0.4         # Try more regularization
```

### 2. Try Attention Mechanism

Edit `train.py`, line ~90:
```python
model = create_model(
    ...
    use_attention=True  # Enable attention
)
```

### 3. Generate More Diverse Attacks

Edit `generate_datasets.py` to add custom attack patterns.

### 4. Adjust Detection Threshold

Edit `evaluate.py`:
```python
THRESHOLD = 0.3  # More sensitive (more TP, more FP)
THRESHOLD = 0.7  # More conservative (fewer FP, fewer TP)
```

---

## File Locations

After running the complete pipeline:

```
btp/
├── data/
│   ├── dataset_1_normal.csv              ← Training data
│   ├── dataset_2_injection_replay.csv    ← Test data 1
│   └── dataset_3_gps_spoofing.csv        ← Test data 2
│
├── models/
│   ├── lstm_ids_best.pth                 ← Best model (use this!)
│   ├── lstm_ids_final.pth                ← Final model
│   └── scaler.pkl                        ← Feature normalizer
│
└── results/
    ├── training_history.png              ← Training curves
    ├── training_history.json             ← Raw training data
    ├── confusion_matrix_dataset2.png     ← Dataset-2 results
    ├── confusion_matrix_dataset3.png     ← Dataset-3 results
    ├── roc_curves.png                    ← ROC comparison
    ├── metrics_comparison.png            ← Metrics bar charts
    └── evaluation_report.json            ← Detailed metrics
```

---

## Performance Benchmarks

**On a typical laptop (Intel i5, 16GB RAM, no GPU):**

- Dataset generation: ~30 seconds
- Model training: ~8 minutes (50 epochs with early stopping)
- Evaluation: ~1 minute

**With GPU (NVIDIA GTX 1060 or better):**

- Dataset generation: ~30 seconds
- Model training: ~2-3 minutes
- Evaluation: ~30 seconds

---

## Getting Help

1. Check [README.md](../README.md) for detailed documentation
2. Review code comments in each module
3. Check error messages carefully
4. Verify file paths are correct
5. Ensure all dependencies are installed

---

## Research Tips

### For Academic Papers

1. **Report all metrics** (not just accuracy)
2. **Include confusion matrices** in your paper
3. **Report ROC-AUC** for threshold-independent evaluation
4. **Discuss FPR** (critical for real-world IDS deployment)
5. **Mention cross-dataset evaluation** (zero-day detection capability)

### For Presentations

1. Use `roc_curves.png` to show model performance
2. Use `metrics_comparison.png` to compare datasets
3. Use confusion matrices to explain TP/FP/TN/FN
4. Highlight low FPR (practical deployability)

### For Further Research

1. Test on real UAV telemetry data
2. Compare with other ML models (SVM, Random Forest, etc.)
3. Implement online learning for adaptive IDS
4. Add explainability (SHAP values for feature importance)
5. Extend to multi-class classification (attack type classification)

---

**You're ready to go! Start with:** `python main.py --all`
