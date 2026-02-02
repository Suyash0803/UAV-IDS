"""
Evaluation Script for UAV LSTM IDS

This module handles comprehensive evaluation:
1. Cross-dataset evaluation (zero-day attack detection)
2. Detailed metrics: Accuracy, Precision, Recall, F1-score, FPR
3. Confusion matrix visualization
4. ROC curve and AUC analysis
5. Per-attack-type performance analysis

Evaluation strategy:
- Test on Dataset-2 (injection/replay attacks)
- Test on Dataset-3 (GPS spoofing/sensor anomalies)
- Measure generalization: Can model detect unseen attack types?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import json
import os
from datetime import datetime

from model import create_model, load_model
from preprocess import UAVDataPreprocessor, create_dataloaders


class IDSEvaluator:
    """Comprehensive evaluator for IDS model"""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize evaluator
        
        Args:
            model: Trained LSTM IDS model
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.model.eval()  # Set to evaluation mode
        
    def predict(self, data_loader, threshold=0.5):
        """
        Make predictions on dataset
        
        Args:
            data_loader: DataLoader for test data
            threshold: Classification threshold
            
        Returns:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
        """
        y_true = []
        y_pred = []
        y_prob = []
        
        self.model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                
                # Get predictions
                output = self.model(data)
                predictions = (output >= threshold).float()
                
                y_true.extend(target.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy().flatten())
                y_prob.extend(output.cpu().numpy().flatten())
        
        return np.array(y_true), np.array(y_pred), np.array(y_prob)
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # False Positive Rate (FPR)
        # FPR = FP / (FP + TN) = False alarms / Total normal samples
        # Critical metric for IDS: Low FPR means fewer false alarms
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Negative Rate (FNR) - Missed attacks
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # True Negative Rate (Specificity)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC AUC
        if len(np.unique(y_true)) > 1:
            fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr_curve, tpr_curve)
        else:
            roc_auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fpr': fpr,
            'fnr': fnr,
            'specificity': tnr,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
        
        return metrics
    
    def evaluate_dataset(self, data_loader, dataset_name, threshold=0.5):
        """
        Evaluate on a single dataset
        
        Args:
            data_loader: DataLoader for dataset
            dataset_name: Name for reporting
            threshold: Classification threshold
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        print(f"\nEvaluating on {dataset_name}...")
        print("-"*70)
        
        # Get predictions
        y_true, y_pred, y_prob = self.predict(data_loader, threshold)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        metrics['dataset'] = dataset_name
        metrics['threshold'] = threshold
        metrics['n_samples'] = len(y_true)
        
        # Print results
        print(f"\nResults for {dataset_name}:")
        print(f"  Samples: {metrics['n_samples']}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"  FPR:         {metrics['fpr']:.4f} (Lower is better)")
        print(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['tn']:6d}  |  FP: {metrics['fp']:6d}")
        print(f"  FN: {metrics['fn']:6d}  |  TP: {metrics['tp']:6d}")
        
        return metrics, y_true, y_pred, y_prob
    
    def plot_confusion_matrix(self, cm, dataset_name, save_path):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            dataset_name: Dataset name for title
            save_path: Path to save plot
        """
        plt.figure(figsize=(8, 6))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'],
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {dataset_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add normalized values as text
        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.7, f'({cm_normalized[i, j]:.2%})',
                        ha='center', va='center', color='gray', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_roc_curve(self, results_list, save_path):
        """
        Plot ROC curves for multiple datasets
        
        Args:
            results_list: List of (dataset_name, y_true, y_prob) tuples
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        for dataset_name, y_true, y_prob in results_list:
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{dataset_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('ROC Curves - Cross-Dataset Evaluation', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ ROC curves saved to {save_path}")
        plt.close()
    
    def plot_metrics_comparison(self, metrics_list, save_path):
        """
        Plot metrics comparison across datasets
        
        Args:
            metrics_list: List of metrics dictionaries
            save_path: Path to save plot
        """
        datasets = [m['dataset'] for m in metrics_list]
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'fpr']
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 4))
        
        for idx, metric in enumerate(metrics_to_plot):
            values = [m[metric] for m in metrics_list]
            
            bars = axes[idx].bar(datasets, values, color=['#2ecc71', '#3498db', '#e74c3c'])
            axes[idx].set_ylabel('Score', fontsize=10)
            axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            axes[idx].set_ylim([0, 1.0])
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.3f}',
                             ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Metrics comparison saved to {save_path}")
        plt.close()
    
    def generate_report(self, metrics_list, save_path):
        """
        Generate comprehensive evaluation report
        
        Args:
            metrics_list: List of metrics from different datasets
            save_path: Path to save JSON report
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': 'LSTM IDS',
            'evaluation_type': 'Cross-Dataset Zero-Day Detection',
            'datasets': metrics_list
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"  ✓ Evaluation report saved to {save_path}")


def main():
    """Main evaluation workflow"""
    print("="*70)
    print("UAV IDS Cross-Dataset Evaluation")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configuration
    WINDOW_SIZE = 10
    BATCH_SIZE = 64
    THRESHOLD = 0.3  # Adjusted threshold for better attack detection
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Step 1: Load model
    print("Step 1: Loading trained model...")
    print("-"*70)
    
    model = create_model(
        input_size=9,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        use_attention=False,
        device=device
    )
    
    # Load best model from training
    model = load_model(model, path='../models/lstm_ids_best.pth', device=device)
    
    # Step 2: Prepare datasets
    print("\nStep 2: Preparing test datasets...")
    print("-"*70)
    
    preprocessor = UAVDataPreprocessor(window_size=WINDOW_SIZE)
    preprocessor.load_scaler()  # Load scaler fitted during training
    
    # Prepare Dataset-2 (Command Injection/Replay)
    print("\nPreparing Dataset-2 (Injection/Replay attacks)...")
    X_test2, y_test2 = preprocessor.prepare_unseen_data(
        '../data/dataset_2_injection_replay.csv',
        stride=1
    )
    
    # Prepare Dataset-3 (GPS Spoofing/Sensor Anomalies)
    print("\nPreparing Dataset-3 (GPS spoofing/sensor anomalies)...")
    X_test3, y_test3 = preprocessor.prepare_unseen_data(
        '../data/dataset_3_gps_spoofing.csv',
        stride=1
    )
    
    # Create data loaders
    from torch.utils.data import DataLoader
    from preprocess import UAVSequenceDataset
    
    test_dataset2 = UAVSequenceDataset(X_test2, y_test2)
    test_loader2 = DataLoader(test_dataset2, batch_size=BATCH_SIZE, shuffle=False)
    
    test_dataset3 = UAVSequenceDataset(X_test3, y_test3)
    test_loader3 = DataLoader(test_dataset3, batch_size=BATCH_SIZE, shuffle=False)
    
    # Step 3: Evaluate
    print("\n" + "="*70)
    print("Step 3: Evaluation")
    print("="*70)
    
    evaluator = IDSEvaluator(model, device=device)
    
    # Evaluate Dataset-2
    metrics2, y_true2, y_pred2, y_prob2 = evaluator.evaluate_dataset(
        test_loader2, 'Dataset-2 (Injection/Replay)', threshold=THRESHOLD
    )
    
    # Evaluate Dataset-3
    metrics3, y_true3, y_pred3, y_prob3 = evaluator.evaluate_dataset(
        test_loader3, 'Dataset-3 (GPS/Sensor)', threshold=THRESHOLD
    )
    
    # Step 4: Visualizations
    print("\n" + "="*70)
    print("Step 4: Generating visualizations")
    print("="*70)
    
    # Confusion matrices
    print("\nGenerating confusion matrices...")
    evaluator.plot_confusion_matrix(
        np.array(metrics2['confusion_matrix']),
        'Dataset-2 (Injection/Replay)',
        '../results/confusion_matrix_dataset2.png'
    )
    evaluator.plot_confusion_matrix(
        np.array(metrics3['confusion_matrix']),
        'Dataset-3 (GPS/Sensor)',
        '../results/confusion_matrix_dataset3.png'
    )
    
    # ROC curves
    print("\nGenerating ROC curves...")
    roc_data = [
        ('Dataset-2 (Injection/Replay)', y_true2, y_prob2),
        ('Dataset-3 (GPS/Sensor)', y_true3, y_prob3)
    ]
    evaluator.plot_roc_curve(
        roc_data,
        '../results/roc_curves.png'
    )
    
    # Metrics comparison
    print("\nGenerating metrics comparison...")
    evaluator.plot_metrics_comparison(
        [metrics2, metrics3],
        '../results/metrics_comparison.png'
    )
    
    # Step 5: Generate report
    print("\n" + "="*70)
    print("Step 5: Generating evaluation report")
    print("="*70)
    
    evaluator.generate_report(
        [metrics2, metrics3],
        '../results/evaluation_report.json'
    )
    
    # Final summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print("\nCross-Dataset Performance (Zero-Day Attack Detection):")
    print("\nDataset-2 (Command Injection & Replay Attacks):")
    print(f"  Accuracy:  {metrics2['accuracy']:.4f}")
    print(f"  Precision: {metrics2['precision']:.4f}")
    print(f"  Recall:    {metrics2['recall']:.4f}")
    print(f"  F1-Score:  {metrics2['f1_score']:.4f}")
    print(f"  FPR:       {metrics2['fpr']:.4f}")
    
    print("\nDataset-3 (GPS Spoofing & Sensor Anomalies):")
    print(f"  Accuracy:  {metrics3['accuracy']:.4f}")
    print(f"  Precision: {metrics3['precision']:.4f}")
    print(f"  Recall:    {metrics3['recall']:.4f}")
    print(f"  F1-Score:  {metrics3['f1_score']:.4f}")
    print(f"  FPR:       {metrics3['fpr']:.4f}")
    
    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70)
    print("\nKey findings:")
    print("- Model trained on normal data (Dataset-1)")
    print("- Tested on unseen attack types (Datasets 2 & 3)")
    print("- Demonstrates zero-day attack detection capability")
    print("- All results saved to results/ directory")
    print("="*70)


if __name__ == "__main__":
    main()
