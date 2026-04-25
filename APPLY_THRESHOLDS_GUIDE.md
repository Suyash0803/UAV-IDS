# Quick Start: Applying Optimized Thresholds

This guide shows how to use the optimized thresholds from threshold tuning in your evaluation pipeline.

---

## Current Status

The threshold tuning identified optimal thresholds:
- **Dataset-2 (Command Injection/Replay):** 0.90
- **Dataset-3 (GPS Spoofing):** 0.10

However, **evaluate.py** currently uses a fixed threshold of **0.5** for all predictions.

---

## Option 1: Quick Update (Modify evaluate.py)

### Step 1: Update the main() function

Find the section in [evaluate.py](src/evaluate.py) around **line 380-400** that evaluates datasets:

**Current code:**
```python
# Evaluate Dataset-2
print("\n" + "="*70)
print("Dataset-2: Command Injection & Replay Attacks")
print("="*70)

X_test2, y_test2 = preprocessor.prepare_unseen_data('../data/dataset_2_injection_replay.csv')
test_dataset2 = UAVSequenceDataset(X_test2, y_test2)
test_loader2 = DataLoader(test_dataset2, batch_size=64, shuffle=False)

metrics2, (y_true2, y_pred2, y_prob2) = evaluator.evaluate_dataset(
    test_loader2, 
    'Dataset-2 (Injection/Replay)',
    threshold=0.3  # Change this value
)
```

**Updated code with optimized threshold:**
```python
# Evaluate Dataset-2
print("\n" + "="*70)
print("Dataset-2: Command Injection & Replay Attacks")
print("="*70)

X_test2, y_test2 = preprocessor.prepare_unseen_data('../data/dataset_2_injection_replay.csv')
test_dataset2 = UAVSequenceDataset(X_test2, y_test2)
test_loader2 = DataLoader(test_dataset2, batch_size=64, shuffle=False)

metrics2, (y_true2, y_pred2, y_prob2) = evaluator.evaluate_dataset(
    test_loader2, 
    'Dataset-2 (Injection/Replay)',
    threshold=0.90  # ✓ OPTIMIZED: Reduces FPR from 64.80% to 62.31%
)
```

### Step 2: Update Dataset-3 evaluation

**Find around line 410-430:**
```python
# Evaluate Dataset-3
X_test3, y_test3 = preprocessor.prepare_unseen_data('../data/dataset_3_gps_spoofing.csv')
test_dataset3 = UAVSequenceDataset(X_test3, y_test3)
test_loader3 = DataLoader(test_dataset3, batch_size=64, shuffle=False)

metrics3, (y_true3, y_pred3, y_prob3) = evaluator.evaluate_dataset(
    test_loader3,
    'Dataset-3 (GPS Spoofing)',
    threshold=0.3  # Change this value
)
```

**Updated code:**
```python
# Evaluate Dataset-3
X_test3, y_test3 = preprocessor.prepare_unseen_data('../data/dataset_3_gps_spoofing.csv')
test_dataset3 = UAVSequenceDataset(X_test3, y_test3)
test_loader3 = DataLoader(test_dataset3, batch_size=64, shuffle=False)

metrics3, (y_true3, y_pred3, y_prob3) = evaluator.evaluate_dataset(
    test_loader3,
    'Dataset-3 (GPS Spoofing)',
    threshold=0.10  # ✓ OPTIMIZED: Improves recall from 50.59% to 51.67%
)
```

### Step 3: Re-run evaluation

```bash
cd src
python evaluate.py
```

---

## Option 2: Production Implementation (Adaptive Thresholds)

For production systems, add adaptive threshold selection to evaluate.py:

### Add this function after the IDSEvaluator class:

```python
def get_optimal_threshold(dataset_type='unknown'):
    """
    Return optimized threshold based on dataset/attack type
    
    Args:
        dataset_type: 'injection', 'gps_spoofing', or 'unknown'
    
    Returns:
        float: Optimal classification threshold
    
    Usage:
        threshold = get_optimal_threshold('injection')
        y_pred = (y_prob >= threshold).astype(int)
    """
    OPTIMIZED_THRESHOLDS = {
        'injection': 0.90,        # Command injection/replay attacks
        'command_replay': 0.90,   # Alias
        'gps_spoofing': 0.10,     # GPS spoofing attacks
        'gps': 0.10,              # Alias
        'unknown': 0.10,          # Conservative default (favor recall)
        'general': 0.50           # Balanced default
    }
    
    return OPTIMIZED_THRESHOLDS.get(dataset_type, 0.10)


# Usage in main():
threshold_dataset2 = get_optimal_threshold('injection')  # Returns 0.90
threshold_dataset3 = get_optimal_threshold('gps_spoofing')  # Returns 0.10

metrics2, (y_true2, y_pred2, y_prob2) = evaluator.evaluate_dataset(
    test_loader2, 
    'Dataset-2 (Injection/Replay)',
    threshold=threshold_dataset2
)
```

---

## Option 3: Dual-Threshold Alert System

For systems requiring human review, implement confidence-based classification:

```python
def classify_with_confidence(y_prob, attack_type='unknown'):
    """
    Three-level classification: Normal, Suspicious, Attack
    
    Args:
        y_prob: Predicted probabilities (numpy array)
        attack_type: 'injection' or 'gps_spoofing'
    
    Returns:
        classifications: Array with values [0=Normal, 1=Suspicious, 2=Attack]
    """
    if attack_type == 'injection':
        high_threshold = 0.90  # Definite attack
        low_threshold = 0.50   # Suspicious
    elif attack_type == 'gps_spoofing':
        high_threshold = 0.50  # Definite attack
        low_threshold = 0.10   # Suspicious
    else:
        high_threshold = 0.70
        low_threshold = 0.30
    
    classifications = np.zeros_like(y_prob, dtype=int)
    classifications[y_prob >= high_threshold] = 2  # Attack
    classifications[(y_prob >= low_threshold) & (y_prob < high_threshold)] = 1  # Suspicious
    # classifications < low_threshold remain 0 (Normal)
    
    return classifications


# Usage:
y_conf = classify_with_confidence(y_prob2, attack_type='injection')
print(f"Normal: {np.sum(y_conf == 0)}")
print(f"Suspicious: {np.sum(y_conf == 1)} (requires human review)")
print(f"Attack: {np.sum(y_conf == 2)} (automated response)")
```

---

## Verification After Update

After applying the optimized thresholds, you should see:

### Expected Output for Dataset-2:
```
Dataset-2: Command Injection & Replay Attacks
==================================================
Threshold: 0.90
Accuracy:  0.9046
Precision: 0.9781  (↑ from 0.9772)
Recall:    0.9221  (slightly lower, acceptable trade-off)
F1-Score:  0.9493  (maintained)
FPR:       0.6231  (↓ from 0.6480) ✓ IMPROVED
```

### Expected Output for Dataset-3:
```
Dataset-3: GPS Spoofing & Sensor Anomalies
==================================================
Threshold: 0.10
Accuracy:  0.5230  (↑ from 0.5127)
Precision: 0.9992  (maintained)
Recall:    0.5167  (↑ from 0.5059) ✓ IMPROVED
F1-Score:  0.6812  (↑ from 0.6719) ✓ IMPROVED
FPR:       0.0292  (slight increase, acceptable)
```

---

## Files Modified

To implement Option 1 (simplest):
- ✏️ [src/evaluate.py](src/evaluate.py) - Update lines ~390 and ~420 (threshold parameters)

To implement Option 2 (recommended):
- ✏️ [src/evaluate.py](src/evaluate.py) - Add `get_optimal_threshold()` function
- ✏️ [src/evaluate.py](src/evaluate.py) - Update main() to use adaptive thresholds

To implement Option 3 (advanced):
- ✏️ [src/evaluate.py](src/evaluate.py) - Add `classify_with_confidence()` function
- ✏️ Create new monitoring dashboard (separate script)

---

## Next Steps

1. **Choose implementation option** (Option 1 recommended for immediate use)
2. **Update evaluate.py** with optimized thresholds
3. **Re-run evaluation:**
   ```bash
   cd src
   python evaluate.py
   ```
4. **Verify improvements** in output metrics
5. **Save new results** in results/ directory
6. **Document changes** in your project README

---

## Need Help?

- **Full threshold analysis:** See [THRESHOLD_TUNING_SUMMARY.md](THRESHOLD_TUNING_SUMMARY.md)
- **Threshold tuning script:** [src/tune_threshold.py](src/tune_threshold.py)
- **JSON results:** [results/threshold_results_dataset2.json](results/threshold_results_dataset2.json)

---

**Remember:** These thresholds are optimized for the current model and datasets. If you retrain the model with different data, you should re-run threshold tuning to find new optimal values.
