# Design Decisions & Technical Architecture

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Dataset Design](#dataset-design)
3. [Model Architecture](#model-architecture)
4. [Training Strategy](#training-strategy)
5. [Evaluation Methodology](#evaluation-methodology)
6. [Key Design Choices](#key-design-choices)

---

## Architecture Overview

### Modular Design Philosophy

The project follows a **modular, research-oriented architecture** with clear separation of concerns:

```
Data Generation → Preprocessing → Model → Training → Evaluation
     ↓                ↓             ↓         ↓          ↓
  datasets.csv    sequences    LSTM IDS   models/   results/
```

**Benefits:**
- Each module can be modified independently
- Easy to experiment with different components
- Reproducible research workflow
- Clean separation between data, model, and evaluation

---

## Dataset Design

### 1. Feature Selection

**Chosen Features (9 total):**
```python
['lat', 'lon', 'altitude', 'velocity', 'pitch', 'roll', 'yaw', 'battery', 'command_id']
```

**Rationale:**
- **GPS (lat, lon, altitude)**: Core telemetry, vulnerable to spoofing
- **Motion (velocity, pitch, roll, yaw)**: Indicates UAV behavior, detects erratic movement
- **Battery**: Can be manipulated to force emergency landing
- **Command ID**: Detects command injection attacks

**Alternative considered:** 
- Adding timestamp deltas → Would make replay detection easier but less generalizable
- Adding acceleration → More features but increases model complexity

**Decision:** Keep features minimal but sufficient to represent UAV state

---

### 2. Attack Pattern Design

#### Dataset-1: Normal Only
**Decision:** Train only on normal data

**Rationale:**
- Simulates real-world scenario (attacks are rare during training)
- Tests model's ability to learn "normal" behavior
- Enables zero-day attack detection
- Avoids overfitting to specific attack patterns

**Alternative considered:** Mix normal + attacks → Would improve detection of seen attacks but hurt generalization

---

#### Dataset-2: Command Injection & Replay
**Attack characteristics:**
1. **Command Injection:**
   - Malicious command IDs (9000-9100 range)
   - May cause erratic velocity/altitude changes
   
2. **Replay Attack:**
   - Exact repetition of previous values
   - Static GPS coordinates (position freeze)

**Rationale:**
- Tests temporal pattern detection (LSTM strength)
- Replay attacks are hard to detect with static ML models
- Command injection is common in IoT attacks

---

#### Dataset-3: GPS Spoofing & Sensor Anomalies
**Attack characteristics:**
1. **GPS Spoofing:**
   - Sudden large jumps (~11km)
   - Persistent spoofing (multiple consecutive samples)
   
2. **Sensor Anomalies:**
   - Physically impossible readings (high velocity + low altitude)
   - Extreme attitude angles
   
3. **Telemetry Manipulation:**
   - Fake battery drops
   - Invalid command sequences
   - Erratic yaw changes

**Rationale:**
- Tests model's understanding of physical constraints
- GPS spoofing is a primary UAV threat
- Sensor anomalies simulate hardware compromise

---

### 3. Attack Ratio: 30%

**Decision:** 30% attack samples in test datasets

**Rationale:**
- Realistic imbalance (attacks are less common than normal traffic)
- Not too imbalanced (still sufficient attack samples for evaluation)
- Tests model's ability to handle class imbalance

**Alternative considered:**
- 50-50 split → Unrealistic
- 5% attacks → Too few attack samples for robust evaluation

---

## Model Architecture

### 1. LSTM vs Other Architectures

| Model | Pros | Cons | Decision |
|-------|------|------|----------|
| **LSTM** | Captures temporal patterns, handles sequences, proven for time-series | Slower than CNNs | ✅ **Chosen** |
| CNN-1D | Fast, parallel processing | Misses long-range dependencies | ❌ |
| Transformer | State-of-the-art NLP, attention | Overkill for short sequences, data-hungry | ❌ |
| Autoencoder | Unsupervised, learns normal manifold | Harder to train, less interpretable | ❌ |
| Random Forest | Fast, interpretable | No temporal modeling, manual feature engineering | ❌ |

**LSTM chosen because:**
1. **Temporal dependencies**: Captures patterns over time windows
2. **Sequence modeling**: Natural fit for telemetry streams
3. **Replay detection**: Can detect repeated patterns
4. **Research standard**: Widely used for IDS tasks

---

### 2. Bidirectional LSTM

**Decision:** Use bidirectional LSTM (forward + backward)

**Rationale:**
- **Forward pass**: Learns patterns from past → present
- **Backward pass**: Learns patterns from future → past
- **Combined**: Better context understanding

**Example:**
```
Normal:  [t-5] → [t-4] → [t-3] → [t-2] → [t-1] → [t]
Attack:  [t-5] → [t-4] → [JUMP] → [t-2] → [t-1] → [t]
                              ↑
                    Detected by looking both ways
```

**Trade-off:** 2x slower than unidirectional, but worth the accuracy gain

---

### 3. Model Size: hidden_size=64, num_layers=2

**Decision:** Moderate model size

**Rationale:**
- **64 hidden units**: Sufficient capacity for 9 features
- **2 layers**: Deep enough for hierarchical patterns, not too deep to overfit
- **~50k parameters**: Small enough to train quickly, large enough to generalize

**Alternatives considered:**
- Larger (128 hidden, 3 layers) → Better capacity but overfits on small datasets
- Smaller (32 hidden, 1 layer) → Underfits, misses complex patterns

---

### 4. Dropout: 0.3

**Decision:** 30% dropout rate

**Rationale:**
- **Prevents overfitting**: Randomly drops 30% of neurons during training
- **Forces redundancy**: Model learns robust features
- **Standard practice**: 0.2-0.5 is common for sequence models

**Applied at:**
- Between LSTM layers
- After fully connected layers

---

### 5. Output: Sigmoid (Probability)

**Decision:** Output attack probability [0, 1] rather than binary classification

**Rationale:**
- **Flexible threshold**: Can adjust sensitivity post-training
- **Confidence score**: Provides confidence in predictions
- **ROC analysis**: Enables threshold-independent evaluation

**Example:**
```
Output: 0.92 → High confidence attack
Output: 0.51 → Borderline (might be false positive)
Output: 0.08 → Likely normal
```

---

## Preprocessing Design

### 1. Sliding Windows: window_size=10

**Decision:** 10-timestep windows

**Rationale:**
- **10 seconds** of telemetry (at 1Hz sampling)
- **Enough context** to detect patterns (replay, GPS jumps)
- **Not too long** to lose temporal resolution

**Visual:**
```
Original: [t1] [t2] [t3] [t4] [t5] [t6] [t7] [t8] [t9] [t10] [t11] ...

Windows:  [t1  t2  t3  t4  t5  t6  t7  t8  t9  t10]  ← Window 1
           [t2  t3  t4  t5  t6  t7  t8  t9  t10 t11] ← Window 2
            [t3  t4  t5  t6  t7  t8  t9  t10 t11 t12] ← Window 3
```

**Alternatives:**
- 5 timesteps → Too short, misses patterns
- 20 timesteps → Too long, less samples, slower training

---

### 2. Window Labeling: ANY timestep attack → Window attack

**Decision:** If any timestep in window is attack, label window as attack

**Rationale:**
- **Conservative**: Flags suspicious windows
- **Early detection**: Detects attacks at window boundaries
- **Practical**: Real IDS raises alert on any anomaly in observation window

**Alternative:** Majority voting → Might miss short-lived attacks

---

### 3. Normalization: MinMaxScaler

**Decision:** MinMaxScaler (scale to [0, 1]) instead of StandardScaler

**Rationale:**
- **Bounded features**: UAV telemetry has natural bounds (altitude: 0-120m, battery: 0-100%)
- **Preserves zero**: Important for command_id=0
- **Neural network friendly**: [0, 1] range works well with sigmoid/tanh activations

**StandardScaler alternative:**
- Uses z-score (mean=0, std=1)
- Better for unbounded features (e.g., sensor noise)
- Can produce negative values (less intuitive for UAV features)

**Critical:** Scaler fitted ONLY on training data to prevent data leakage!

---

## Training Strategy

### 1. Training Set: Dataset-1 (Normal Only)

**Decision:** Train exclusively on normal behavior

**Rationale:**
- **Anomaly detection paradigm**: Learn normal → flag deviations
- **Zero-day capability**: Detects unseen attack types
- **Real-world scenario**: Limited attack samples during deployment

**Trade-off:**
- Lower recall on attacks (some attacks look "normal-ish")
- But better generalization to new attacks

---

### 2. Loss Function: Binary Cross-Entropy

**Formula:**
```
BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Rationale:**
- **Standard for binary classification**
- **Probabilistic interpretation**: Measures KL divergence between distributions
- **Gradient-friendly**: Smooth gradients for optimization

**Alternative considered:**
- Focal Loss → Good for extreme imbalance, but 30% attack rate is manageable with BCE
- Class weights → Can be added if FPR is too high

---

### 3. Optimizer: Adam

**Decision:** Adam with learning_rate=0.001

**Rationale:**
- **Adaptive learning rates**: Per-parameter learning rates
- **Momentum**: Accelerates convergence
- **Robust**: Works well out-of-the-box for most problems

**Alternatives:**
- SGD → Slower convergence, needs manual LR tuning
- AdamW → Better regularization, but Adam sufficient for this task

---

### 4. Learning Rate Scheduling: ReduceLROnPlateau

**Decision:** Reduce LR by 0.5 when validation loss plateaus for 5 epochs

**Rationale:**
- **Fine-tuning**: Initially learns fast (LR=0.001), then fine-tunes (LR=0.0005, 0.00025...)
- **Escape local minima**: Smaller LR helps converge to better solutions
- **Automatic**: No manual intervention needed

**Schedule:**
```
Epochs 1-10:   LR = 0.001   (rapid learning)
Epochs 11-20:  LR = 0.0005  (refinement)
Epochs 21-30:  LR = 0.00025 (fine-tuning)
```

---

### 5. Early Stopping: patience=10

**Decision:** Stop if validation loss doesn't improve for 10 consecutive epochs

**Rationale:**
- **Prevents overfitting**: Stops before model memorizes training data
- **Saves time**: No need to run all 50 epochs if converged
- **Best model selection**: Automatically saves best checkpoint

**Visual:**
```
Validation Loss:
Epoch 1:  0.35 ← Save (best)
Epoch 5:  0.20 ← Save (best)
Epoch 10: 0.15 ← Save (best)
Epoch 15: 0.14 ← Save (best)
Epoch 20: 0.16 ← No improvement (counter: 5)
Epoch 25: 0.18 ← No improvement (counter: 10) → STOP
```

---

## Evaluation Methodology

### 1. Cross-Dataset Evaluation

**Decision:** Train on Dataset-1, test on Datasets 2 & 3

**Rationale:**
- **Zero-day detection**: Tests on unseen attack distributions
- **Generalization**: Proves model isn't overfitting to specific attacks
- **Real-world simulation**: New attacks appear constantly

**Contrast with:**
- Train-test split on same dataset → Easier but less realistic
- K-fold CV → Good for model selection, but doesn't test generalization to new attacks

---

### 2. Metrics: Emphasize FPR

**Decision:** Report accuracy, precision, recall, F1, **FPR**, ROC-AUC

**Why FPR is critical for IDS:**
- **False positives = False alarms**
- High FPR → Security team ignores alerts (alarm fatigue)
- **Goal: FPR < 0.10** (< 10% false alarm rate)

**Metric priorities:**
1. **FPR**: Must be low (< 0.10)
2. **Recall**: Should be high (> 0.85) to catch attacks
3. **F1**: Balances precision and recall
4. **Accuracy**: Useful but can be misleading with imbalanced data

---

### 3. Confusion Matrix Visualization

**Decision:** Plot normalized + raw counts

**Rationale:**
- **Raw counts**: Absolute numbers (e.g., "500 false alarms out of 10,000")
- **Normalized**: Percentages per class (e.g., "5% FPR")
- **Combined**: Full picture of model performance

---

### 4. ROC Curves: Threshold-Independent Evaluation

**Decision:** Plot ROC curves for all test datasets

**Rationale:**
- **Threshold-independent**: Performance across all thresholds
- **AUC metric**: Single number summarizing performance (AUC > 0.85 is good)
- **Comparison**: Easy to compare multiple datasets on one plot

**Interpretation:**
```
AUC = 1.0:  Perfect classifier
AUC = 0.9:  Excellent
AUC = 0.8:  Good
AUC = 0.7:  Fair
AUC = 0.5:  Random guessing
```

---

## Key Design Choices Summary

### 1. Research Quality
- ✅ Modular code structure
- ✅ Comprehensive documentation
- ✅ Reproducible (random seeds)
- ✅ Extensive evaluation metrics
- ✅ Cross-dataset testing

### 2. Extensibility
- ✅ Easy to add new attack types
- ✅ Easy to modify model architecture
- ✅ Easy to experiment with hyperparameters
- ✅ Plug-and-play data pipeline

### 3. Practical Considerations
- ✅ Low false positive rate
- ✅ Interpretable results (confusion matrices)
- ✅ Flexible threshold adjustment
- ✅ Fast inference (LSTM is lightweight)

### 4. Trade-offs Made

| Choice | Benefit | Cost |
|--------|---------|------|
| Synthetic data | Reproducible, controlled | Not real-world validated |
| Train on normal only | Zero-day detection | Lower recall on attacks |
| LSTM | Temporal modeling | Slower than static models |
| Window size=10 | Good context | Some memory overhead |
| 30% attack rate | Realistic imbalance | Could test extreme imbalance |

---

## Future Improvements

### Short-term
1. **Class weights**: Handle imbalance better
2. **Attention mechanism**: Already implemented, needs testing
3. **Ensemble methods**: Combine multiple models
4. **Hyperparameter tuning**: Grid search for optimal parameters

### Medium-term
1. **Real UAV data**: Validate on actual drone telemetry
2. **Online learning**: Update model with new data
3. **Multi-class classification**: Identify specific attack types
4. **Feature importance**: SHAP values for explainability

### Long-term
1. **Federated learning**: Distributed IDS across UAV swarm
2. **Real-time deployment**: Integrate with UAV firmware
3. **Adversarial robustness**: Defense against evasion attacks
4. **Automatic feature engineering**: Learn features from raw sensor data

---

## Conclusion

The design prioritizes:
1. **Research quality**: Modular, well-documented, reproducible
2. **Generalization**: Zero-day attack detection capability
3. **Practicality**: Low false positives, flexible threshold
4. **Extensibility**: Easy to modify and extend

Every design decision balances multiple factors (accuracy, speed, generalization, interpretability) to create a robust, research-oriented IDS.

---

**Questions or suggestions?** Check the code comments for implementation details!
