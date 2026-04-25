"""
LSTM Autoencoder Evaluation Script

This module evaluates the trained autoencoder on test datasets using
reconstruction error as an anomaly score.

Evaluation Strategy:
===================
1. Load trained autoencoder and anomaly threshold
2. Test on Dataset-2 (command injection/replay attacks)
3. Test on Dataset-3 (GPS spoofing attacks - zero-day)
4. Classify: error > threshold → anomaly (attack)
5. Compute metrics: Precision, Recall, F1, FPR

Key Metric: Recall on Dataset-3 (GPS Spoofing)
===============================================
Since the autoencoder was trained ONLY on normal data, it should detect
GPS spoofing attacks (large jumps) as high reconstruction errors.
This is the primary advantage over supervised classifiers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import json
from datetime import datetime

from model import create_autoencoder
from preprocess import UAVDataPreprocessor, UAVSequenceDataset
from torch.utils.data import DataLoader


class AutoencoderEvaluator:
    """Evaluator for LSTM Autoencoder anomaly detection"""
    
    def __init__(self, model, threshold, device='cpu'):
        """
        Initialize evaluator
        
        Args:
            model: Trained LSTM_Autoencoder
            threshold: Anomaly detection threshold (from training)
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.threshold = threshold
        self.device = device
        self.model.eval()
    
    def predict(self, data_loader):
        """
        Predict anomalies based on reconstruction error
        
        Args:
            data_loader: DataLoader for test data
            
        Returns:
            y_true: True labels
            y_pred: Predicted labels (0=normal, 1=anomaly)
            errors: Reconstruction errors
        """
        y_true = []
        errors = []
        
        self.model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                
                # Compute reconstruction error
                batch_errors = self.model.get_reconstruction_error(data)
                
                y_true.extend(target.cpu().numpy())
                errors.extend(batch_errors.cpu().numpy())
        
        y_true = np.array(y_true)
        errors = np.array(errors)
        
        # Classify: error > threshold → anomaly
        y_pred = (errors > self.threshold).astype(int)
        
        return y_true, y_pred, errors
    
    def evaluate_dataset(self, data_loader, dataset_name):
        """
        Evaluate on a dataset
        
        Args:
            data_loader: Test data loader
            dataset_name: Name for display
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        print(f"\nEvaluating: {dataset_name}")
        print("-"*70)
        
        # Get predictions
        y_true, y_pred, errors = self.predict(data_loader)
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Print results
        print(f"Samples: {len(y_true)}")
        print(f"Threshold: {self.threshold:.6f}\n")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"FPR:       {fpr:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {tn:6d}  |  FP: {fp:6d}")
        print(f"  FN: {fn:6d}  |  TP: {tp:6d}")
        
        # Error statistics
        normal_errors = errors[y_true == 0]
        attack_errors = errors[y_true == 1]
        
        print(f"\nReconstruction Error Statistics:")
        print(f"  Normal samples:")
        print(f"    Mean: {np.mean(normal_errors):.6f}")
        print(f"    Std:  {np.std(normal_errors):.6f}")
        print(f"  Attack samples:")
        print(f"    Mean: {np.mean(attack_errors):.6f}")
        print(f"    Std:  {np.std(attack_errors):.6f}")
        print(f"  Separation: {np.mean(attack_errors) / np.mean(normal_errors):.2f}x")
        
        metrics = {
            'dataset': dataset_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'fpr': float(fpr),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            },
            'error_stats': {
                'normal_mean': float(np.mean(normal_errors)),
                'normal_std': float(np.std(normal_errors)),
                'attack_mean': float(np.mean(attack_errors)),
                'attack_std': float(np.std(attack_errors))
            }
        }
        
        return metrics, (y_true, y_pred, errors)
    
    def plot_error_distribution(self, y_true, errors, dataset_name, save_path):
        """
        Plot reconstruction error distribution
        
        Args:
            y_true: True labels
            errors: Reconstruction errors
            dataset_name: Dataset name for title
            save_path: Path to save plot
        """
        normal_errors = errors[y_true == 0]
        attack_errors = errors[y_true == 1]
        
        plt.figure(figsize=(14, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal', color='blue', edgecolor='black')
        plt.hist(attack_errors, bins=50, alpha=0.6, label='Attack', color='red', edgecolor='black')
        plt.axvline(self.threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold: {self.threshold:.6f}')
        plt.xlabel('Reconstruction Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Error Distribution - {dataset_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        data_to_plot = [normal_errors, attack_errors]
        bp = plt.boxplot(data_to_plot, labels=['Normal', 'Attack'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        plt.axhline(self.threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold: {self.threshold:.6f}')
        plt.ylabel('Reconstruction Error', fontsize=12)
        plt.title(f'Error Comparison - {dataset_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, dataset_name, save_path):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            dataset_name: Dataset name for title
            save_path: Path to save plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'],
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(f'Confusion Matrix - {dataset_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, y_true, errors, dataset_name, save_path):
        """
        Plot ROC curve using reconstruction errors as scores
        
        Args:
            y_true: True labels
            errors: Reconstruction errors (higher = more anomalous)
            dataset_name: Dataset name for title
            save_path: Path to save plot
        """
        fpr_list, tpr_list, _ = roc_curve(y_true, errors)
        roc_auc = auc(fpr_list, tpr_list)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_list, tpr_list, linewidth=2, 
                label=f'{dataset_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {dataset_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()


def main():
    """Main evaluation workflow"""
    print("="*70)
    print("UAV IDS - LSTM Autoencoder Evaluation (Anomaly Detection)")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configuration
    WINDOW_SIZE = 10
    BATCH_SIZE = 64
    HIDDEN_SIZE = 32
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Step 1: Load model and threshold
    print("Step 1: Loading autoencoder and threshold")
    print("-"*70)
    
    # Load threshold
    with open('../models/autoencoder_threshold.json', 'r') as f:
        threshold_info = json.load(f)
    
    anomaly_threshold = threshold_info['anomaly_threshold']
    print(f"Anomaly threshold: {anomaly_threshold:.6f} (95th percentile)")
    
    # Load model
    model = create_autoencoder(
        input_size=9,
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        dropout=0.2,
        device=device
    )
    
    model.load_state_dict(torch.load('../models/autoencoder_best.pth', map_location=device))
    model.eval()
    print("✓ Model loaded\n")
    
    # Step 2: Load preprocessor
    preprocessor = UAVDataPreprocessor(window_size=WINDOW_SIZE)
    preprocessor.load_scaler()
    
    # Initialize evaluator
    evaluator = AutoencoderEvaluator(model, anomaly_threshold, device=device)
    
    # Step 3: Evaluate on Dataset-2
    print("\n" + "="*70)
    print("Step 2: Dataset-2 (Command Injection & Replay Attacks)")
    print("="*70)
    
    X_test2, y_test2 = preprocessor.prepare_unseen_data('../data/dataset_2_injection_replay.csv')
    test_dataset2 = UAVSequenceDataset(X_test2, y_test2)
    test_loader2 = DataLoader(test_dataset2, batch_size=BATCH_SIZE, shuffle=False)
    
    metrics2, (y_true2, y_pred2, errors2) = evaluator.evaluate_dataset(
        test_loader2, 'Dataset-2 (Injection/Replay)'
    )
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    evaluator.plot_error_distribution(
        y_true2, errors2, 'Dataset-2',
        '../results/autoencoder_errors_dataset2.png'
    )
    evaluator.plot_confusion_matrix(
        y_true2, y_pred2, 'Dataset-2',
        '../results/autoencoder_cm_dataset2.png'
    )
    evaluator.plot_roc_curve(
        y_true2, errors2, 'Dataset-2',
        '../results/autoencoder_roc_dataset2.png'
    )
    
    # Step 4: Evaluate on Dataset-3 (ZERO-DAY GPS SPOOFING)
    print("\n" + "="*70)
    print("Step 3: Dataset-3 (GPS Spoofing - Zero-Day Attack)")
    print("="*70)
    print("Key Question: Can autoencoder detect attacks it never saw during training?")
    
    X_test3, y_test3 = preprocessor.prepare_unseen_data('../data/dataset_3_gps_spoofing.csv')
    test_dataset3 = UAVSequenceDataset(X_test3, y_test3)
    test_loader3 = DataLoader(test_dataset3, batch_size=BATCH_SIZE, shuffle=False)
    
    metrics3, (y_true3, y_pred3, errors3) = evaluator.evaluate_dataset(
        test_loader3, 'Dataset-3 (GPS Spoofing)'
    )
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    evaluator.plot_error_distribution(
        y_true3, errors3, 'Dataset-3',
        '../results/autoencoder_errors_dataset3.png'
    )
    evaluator.plot_confusion_matrix(
        y_true3, y_pred3, 'Dataset-3',
        '../results/autoencoder_cm_dataset3.png'
    )
    evaluator.plot_roc_curve(
        y_true3, errors3, 'Dataset-3',
        '../results/autoencoder_roc_dataset3.png'
    )
    
    # Step 5: Save results
    print("\n" + "="*70)
    print("Step 4: Saving results")
    print("="*70)
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'LSTM Autoencoder',
        'threshold': anomaly_threshold,
        'dataset_2': metrics2,
        'dataset_3': metrics3
    }
    
    with open('../results/autoencoder_evaluation_report.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("✓ Results saved to autoencoder_evaluation_report.json")
    
    # Step 6: Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print("\n📊 Dataset-2 (Command Injection/Replay):")
    print(f"  Recall:    {metrics2['recall']:.4f}")
    print(f"  Precision: {metrics2['precision']:.4f}")
    print(f"  F1-Score:  {metrics2['f1_score']:.4f}")
    print(f"  FPR:       {metrics2['fpr']:.4f}")
    
    print("\n📊 Dataset-3 (GPS Spoofing - Zero-Day):")
    print(f"  Recall:    {metrics3['recall']:.4f} ← Zero-day detection capability")
    print(f"  Precision: {metrics3['precision']:.4f}")
    print(f"  F1-Score:  {metrics3['f1_score']:.4f}")
    print(f"  FPR:       {metrics3['fpr']:.4f}")
    
    print("\n💡 Key Insights:")
    if metrics3['recall'] > 0.5:
        print("  ✓ Autoencoder successfully detects GPS spoofing (zero-day)")
        print("    → High reconstruction errors for sudden GPS jumps")
    else:
        print("  ⚠️  Limited zero-day detection capability")
        print("    → Consider: larger latent dimension, more training data")
    
    if metrics2['fpr'] > 0.3:
        print(f"  ⚠️  High FPR on Dataset-2: {metrics2['fpr']:.1%}")
        print("    → Autoencoder may be too sensitive")
        print("    → Consider: adjust threshold, ensemble with classifier")
    
    print("\n📁 Generated Files:")
    print("  Models:")
    print("    - autoencoder_best.pth")
    print("    - autoencoder_threshold.json")
    print("  Results:")
    print("    - autoencoder_evaluation_report.json")
    print("    - autoencoder_errors_dataset2.png")
    print("    - autoencoder_errors_dataset3.png")
    print("    - autoencoder_cm_dataset2.png")
    print("    - autoencoder_cm_dataset3.png")
    print("    - autoencoder_roc_dataset2.png")
    print("    - autoencoder_roc_dataset3.png")
    
    print("\n" + "="*70)
    print("Next Step: Run ensemble_evaluate.py")
    print("  → Combine classifier + autoencoder for best performance")
    print("="*70)


if __name__ == "__main__":
    main()
