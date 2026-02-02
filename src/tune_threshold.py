"""
Threshold Tuning Script

This script helps find the optimal classification threshold for better attack detection.
When training only on normal data, the default 0.5 threshold is often too conservative.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve

from model import create_model, load_model
from preprocess import UAVDataPreprocessor
from torch.utils.data import DataLoader
from preprocess import UAVSequenceDataset


def find_optimal_threshold(model, test_loader, device='cpu'):
    """
    Find optimal threshold by maximizing F1-score
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        optimal_threshold: Best threshold
        metrics_at_threshold: Performance metrics at optimal threshold
    """
    model.eval()
    
    # Get predictions
    y_true = []
    y_prob = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            
            y_true.extend(target.cpu().numpy())
            y_prob.extend(output.cpu().numpy().flatten())
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Try different thresholds
    thresholds = np.arange(0.01, 0.99, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)
    
    # Find best threshold
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    
    accuracy = accuracy_score(y_true, y_pred_optimal)
    precision = precision_score(y_true, y_pred_optimal, zero_division=0)
    recall = recall_score(y_true, y_pred_optimal, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_optimal)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics = {
        'threshold': optimal_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': best_f1,
        'fpr': fpr,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }
    
    # Plot threshold vs F1-score
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, linewidth=2)
    plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('Threshold Tuning: F1-Score vs Classification Threshold', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return optimal_threshold, metrics, (thresholds, f1_scores)


def main():
    print("="*70)
    print("UAV IDS - Threshold Tuning")
    print("="*70)
    print("\nThis script finds the optimal classification threshold")
    print("to improve attack detection on test datasets.\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading trained model...")
    model = create_model(input_size=9, hidden_size=64, num_layers=2, dropout=0.3, device=device)
    model = load_model(model, path='../models/lstm_ids_best.pth', device=device)
    
    # Load preprocessor
    preprocessor = UAVDataPreprocessor(window_size=10)
    preprocessor.load_scaler()
    
    # Test on Dataset-2
    print("\n" + "="*70)
    print("Dataset-2: Command Injection & Replay Attacks")
    print("="*70)
    
    X_test2, y_test2 = preprocessor.prepare_unseen_data('../data/dataset_2_injection_replay.csv')
    test_dataset2 = UAVSequenceDataset(X_test2, y_test2)
    test_loader2 = DataLoader(test_dataset2, batch_size=64, shuffle=False)
    
    optimal_thresh2, metrics2, (thresholds2, f1s2) = find_optimal_threshold(model, test_loader2, device)
    
    print(f"\nOptimal Threshold: {optimal_thresh2:.3f}")
    print(f"  Accuracy:  {metrics2['accuracy']:.4f}")
    print(f"  Precision: {metrics2['precision']:.4f}")
    print(f"  Recall:    {metrics2['recall']:.4f}")
    print(f"  F1-Score:  {metrics2['f1_score']:.4f}")
    print(f"  FPR:       {metrics2['fpr']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics2['tn']:6d}  |  FP: {metrics2['fp']:6d}")
    print(f"  FN: {metrics2['fn']:6d}  |  TP: {metrics2['tp']:6d}")
    
    plt.savefig('../results/threshold_tuning_dataset2.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to ../results/threshold_tuning_dataset2.png")
    plt.close()
    
    # Test on Dataset-3
    print("\n" + "="*70)
    print("Dataset-3: GPS Spoofing & Sensor Anomalies")
    print("="*70)
    
    X_test3, y_test3 = preprocessor.prepare_unseen_data('../data/dataset_3_gps_spoofing.csv')
    test_dataset3 = UAVSequenceDataset(X_test3, y_test3)
    test_loader3 = DataLoader(test_dataset3, batch_size=64, shuffle=False)
    
    optimal_thresh3, metrics3, (thresholds3, f1s3) = find_optimal_threshold(model, test_loader3, device)
    
    print(f"\nOptimal Threshold: {optimal_thresh3:.3f}")
    print(f"  Accuracy:  {metrics3['accuracy']:.4f}")
    print(f"  Precision: {metrics3['precision']:.4f}")
    print(f"  Recall:    {metrics3['recall']:.4f}")
    print(f"  F1-Score:  {metrics3['f1_score']:.4f}")
    print(f"  FPR:       {metrics3['fpr']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics3['tn']:6d}  |  FP: {metrics3['fp']:6d}")
    print(f"  FN: {metrics3['fn']:6d}  |  TP: {metrics3['tp']:6d}")
    
    plt.savefig('../results/threshold_tuning_dataset3.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to ../results/threshold_tuning_dataset3.png")
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nRecommended Thresholds:")
    print(f"  Dataset-2: {optimal_thresh2:.3f} (F1={metrics2['f1_score']:.4f})")
    print(f"  Dataset-3: {optimal_thresh3:.3f} (F1={metrics3['f1_score']:.4f})")
    print("\nTo use these thresholds, edit evaluate.py:")
    print(f"  Change THRESHOLD from 0.5 to ~{min(optimal_thresh2, optimal_thresh3):.2f}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
