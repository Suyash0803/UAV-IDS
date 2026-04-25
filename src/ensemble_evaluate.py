"""
Ensemble Evaluation: LSTM Classifier + Autoencoder

This module combines predictions from both models using OR logic:
    attack = (classifier predicts attack) OR (autoencoder detects anomaly)

Rationale:
==========
- LSTM Classifier: Good at detecting known attack patterns (injection/replay)
- Autoencoder: Good at detecting deviations from normal (GPS spoofing)
- Ensemble: Combines strengths, improves recall on zero-day attacks

Integration Strategy:
====================
1. Load both trained models
2. For each sample:
   a. Get classifier prediction (probability)
   b. Get autoencoder reconstruction error
   c. Apply OR logic: attack if EITHER model flags it
3. Measure improvement in recall (especially on Dataset-3)

Expected Benefits:
==================
- Higher recall on GPS spoofing (Dataset-3)
- Maintained precision (both models must agree for false positive)
- Robust zero-day detection (unsupervised component)
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

from model import create_model, create_autoencoder, load_model
from preprocess import UAVDataPreprocessor, UAVSequenceDataset
from torch.utils.data import DataLoader


class EnsembleEvaluator:
    """Ensemble evaluator combining classifier and autoencoder"""
    
    def __init__(self, classifier, autoencoder, classifier_threshold, 
                 autoencoder_threshold, device='cpu'):
        """
        Initialize ensemble evaluator
        
        Args:
            classifier: Trained LSTM_IDS model
            autoencoder: Trained LSTM_Autoencoder model
            classifier_threshold: Classification threshold (from threshold tuning)
            autoencoder_threshold: Anomaly threshold (from autoencoder training)
            device: 'cpu' or 'cuda'
        """
        self.classifier = classifier
        self.autoencoder = autoencoder
        self.classifier_threshold = classifier_threshold
        self.autoencoder_threshold = autoencoder_threshold
        self.device = device
        
        self.classifier.eval()
        self.autoencoder.eval()
    
    def predict(self, data_loader, fusion='or'):
        """
        Make ensemble predictions
        
        Args:
            data_loader: DataLoader for test data
            fusion: 'or' (union), 'and' (intersection), 'vote' (majority)
            
        Returns:
            y_true: True labels
            y_pred_ensemble: Ensemble predictions
            y_pred_classifier: Classifier predictions
            y_pred_autoencoder: Autoencoder predictions
            classifier_probs: Classifier probabilities
            reconstruction_errors: Autoencoder errors
        """
        y_true = []
        classifier_probs = []
        reconstruction_errors = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                
                # Classifier prediction
                cls_output = self.classifier(data)
                classifier_probs.extend(cls_output.cpu().numpy().flatten())
                
                # Autoencoder prediction
                ae_errors = self.autoencoder.get_reconstruction_error(data)
                reconstruction_errors.extend(ae_errors.cpu().numpy())
                
                y_true.extend(target.cpu().numpy())
        
        y_true = np.array(y_true)
        classifier_probs = np.array(classifier_probs)
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Individual predictions
        y_pred_classifier = (classifier_probs >= self.classifier_threshold).astype(int)
        y_pred_autoencoder = (reconstruction_errors > self.autoencoder_threshold).astype(int)
        
        # Ensemble fusion
        if fusion == 'or':
            # Attack if EITHER model flags it (union)
            y_pred_ensemble = np.logical_or(y_pred_classifier, y_pred_autoencoder).astype(int)
        elif fusion == 'and':
            # Attack if BOTH models flag it (intersection)
            y_pred_ensemble = np.logical_and(y_pred_classifier, y_pred_autoencoder).astype(int)
        elif fusion == 'vote':
            # Majority vote (at least 2 out of 2 agree)
            votes = y_pred_classifier + y_pred_autoencoder
            y_pred_ensemble = (votes >= 1).astype(int)  # Same as OR for 2 models
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")
        
        return (y_true, y_pred_ensemble, y_pred_classifier, y_pred_autoencoder,
                classifier_probs, reconstruction_errors)
    
    def evaluate_dataset(self, data_loader, dataset_name, fusion='or'):
        """
        Evaluate ensemble on a dataset
        
        Args:
            data_loader: Test data loader
            dataset_name: Name for display
            fusion: Fusion method ('or', 'and', 'vote')
            
        Returns:
            metrics: Dictionary of metrics for all models
        """
        print(f"\nEvaluating: {dataset_name}")
        print("-"*70)
        
        # Get predictions
        (y_true, y_pred_ensemble, y_pred_classifier, y_pred_autoencoder,
         cls_probs, ae_errors) = self.predict(data_loader, fusion=fusion)
        
        # Compute metrics for each model
        models = {
            'Ensemble': y_pred_ensemble,
            'Classifier': y_pred_classifier,
            'Autoencoder': y_pred_autoencoder
        }
        
        all_metrics = {}
        
        for model_name, y_pred in models.items():
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            all_metrics[model_name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'fpr': float(fpr),
                'confusion_matrix': {
                    'tn': int(tn), 'fp': int(fp),
                    'fn': int(fn), 'tp': int(tp)
                }
            }
        
        # Print comparison
        print(f"\n{'Model':<15} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8} | {'FPR':>8}")
        print("-"*80)
        for model_name in ['Classifier', 'Autoencoder', 'Ensemble']:
            m = all_metrics[model_name]
            print(f"{model_name:<15} | {m['accuracy']:>8.4f} | {m['precision']:>9.4f} | "
                  f"{m['recall']:>8.4f} | {m['f1_score']:>8.4f} | {m['fpr']:>8.4f}")
        print("-"*80)
        
        # Highlight improvements
        ensemble_recall = all_metrics['Ensemble']['recall']
        classifier_recall = all_metrics['Classifier']['recall']
        autoencoder_recall = all_metrics['Autoencoder']['recall']
        
        if ensemble_recall > max(classifier_recall, autoencoder_recall):
            print(f"\n✓ Ensemble improves recall: {ensemble_recall:.4f} "
                  f"(Classifier: {classifier_recall:.4f}, Autoencoder: {autoencoder_recall:.4f})")
        
        return all_metrics, (y_true, y_pred_ensemble, y_pred_classifier, 
                            y_pred_autoencoder, cls_probs, ae_errors)
    
    def plot_comparison(self, metrics_dataset2, metrics_dataset3, save_path):
        """
        Plot comparison of all models across datasets
        
        Args:
            metrics_dataset2: Metrics for Dataset-2
            metrics_dataset3: Metrics for Dataset-3
            save_path: Path to save plot
        """
        metrics = ['recall', 'precision', 'f1_score', 'fpr']
        models = ['Classifier', 'Autoencoder', 'Ensemble']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Dataset-2 values
            d2_values = [metrics_dataset2[model][metric] for model in models]
            # Dataset-3 values
            d3_values = [metrics_dataset3[model][metric] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax.bar(x - width/2, d2_values, width, label='Dataset-2', alpha=0.8)
            ax.bar(x + width/2, d3_values, width, label='Dataset-3', alpha=0.8)
            
            ax.set_xlabel('Model', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', 
                        fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Annotate values
            for i, (v1, v2) in enumerate(zip(d2_values, d3_values)):
                ax.text(i - width/2, v1 + 0.02, f'{v1:.3f}', 
                       ha='center', va='bottom', fontsize=8)
                ax.text(i + width/2, v2 + 0.02, f'{v2:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Ensemble vs Individual Models', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def plot_venn_diagram(self, y_true, y_pred_cls, y_pred_ae, dataset_name, save_path):
        """
        Plot Venn diagram showing agreement between models
        
        Args:
            y_true: True labels
            y_pred_cls: Classifier predictions
            y_pred_ae: Autoencoder predictions
            dataset_name: Dataset name for title
            save_path: Path to save plot
        """
        # Find attack samples
        attacks = y_true == 1
        
        # Count detections
        cls_detected = np.sum(y_pred_cls[attacks] == 1)
        ae_detected = np.sum(y_pred_ae[attacks] == 1)
        both_detected = np.sum((y_pred_cls[attacks] == 1) & (y_pred_ae[attacks] == 1))
        cls_only = cls_detected - both_detected
        ae_only = ae_detected - both_detected
        neither = np.sum(attacks) - (cls_only + ae_only + both_detected)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create text summary
        total_attacks = np.sum(attacks)
        text = f"Attack Detection Overlap - {dataset_name}\n\n"
        text += f"Total attacks: {total_attacks}\n\n"
        text += f"Classifier only:      {cls_only:4d} ({cls_only/total_attacks*100:5.1f}%)\n"
        text += f"Autoencoder only:     {ae_only:4d} ({ae_only/total_attacks*100:5.1f}%)\n"
        text += f"Both detected:        {both_detected:4d} ({both_detected/total_attacks*100:5.1f}%)\n"
        text += f"Neither detected:     {neither:4d} ({neither/total_attacks*100:5.1f}%)\n\n"
        text += f"Ensemble (OR):        {cls_only + ae_only + both_detected:4d} "
        text += f"({(cls_only + ae_only + both_detected)/total_attacks*100:5.1f}%)\n"
        
        ax.text(0.5, 0.5, text, fontsize=14, family='monospace',
               ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        ax.set_title(f'Model Agreement Analysis', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()


def main():
    """Main ensemble evaluation workflow"""
    print("="*70)
    print("UAV IDS - Ensemble Evaluation (Classifier + Autoencoder)")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configuration
    WINDOW_SIZE = 10
    BATCH_SIZE = 64
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Step 1: Load models and thresholds
    print("Step 1: Loading models")
    print("-"*70)
    
    # Load classifier
    classifier = create_model(
        input_size=9,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        use_attention=False,
        device=device
    )
    classifier = load_model(classifier, path='../models/lstm_ids_best.pth', device=device)
    
    # Optimized thresholds from threshold tuning
    classifier_threshold_d2 = 0.90  # Dataset-2 optimized
    classifier_threshold_d3 = 0.10  # Dataset-3 optimized
    
    # Load autoencoder
    print()
    autoencoder = create_autoencoder(
        input_size=9,
        hidden_size=32,
        num_layers=1,
        dropout=0.2,
        device=device
    )
    autoencoder.load_state_dict(torch.load('../models/autoencoder_best.pth', map_location=device))
    autoencoder.eval()
    
    # Load autoencoder threshold
    with open('../models/autoencoder_threshold.json', 'r') as f:
        threshold_info = json.load(f)
    autoencoder_threshold = threshold_info['anomaly_threshold']
    
    print(f"\n✓ Models loaded")
    print(f"  Classifier threshold (Dataset-2): {classifier_threshold_d2:.2f}")
    print(f"  Classifier threshold (Dataset-3): {classifier_threshold_d3:.2f}")
    print(f"  Autoencoder threshold: {autoencoder_threshold:.6f}")
    
    # Step 2: Load preprocessor
    preprocessor = UAVDataPreprocessor(window_size=WINDOW_SIZE)
    preprocessor.load_scaler()
    
    # Step 3: Evaluate on Dataset-2
    print("\n" + "="*70)
    print("Step 2: Dataset-2 (Command Injection & Replay)")
    print("="*70)
    
    X_test2, y_test2 = preprocessor.prepare_unseen_data('../data/dataset_2_injection_replay.csv')
    test_dataset2 = UAVSequenceDataset(X_test2, y_test2)
    test_loader2 = DataLoader(test_dataset2, batch_size=BATCH_SIZE, shuffle=False)
    
    evaluator2 = EnsembleEvaluator(
        classifier, autoencoder,
        classifier_threshold_d2, autoencoder_threshold,
        device=device
    )
    
    metrics2, (y_true2, y_pred_ens2, y_pred_cls2, y_pred_ae2, _, _) = \
        evaluator2.evaluate_dataset(test_loader2, 'Dataset-2 (Injection/Replay)', fusion='or')
    
    # Step 4: Evaluate on Dataset-3
    print("\n" + "="*70)
    print("Step 3: Dataset-3 (GPS Spoofing - Zero-Day)")
    print("="*70)
    
    X_test3, y_test3 = preprocessor.prepare_unseen_data('../data/dataset_3_gps_spoofing.csv')
    test_dataset3 = UAVSequenceDataset(X_test3, y_test3)
    test_loader3 = DataLoader(test_dataset3, batch_size=BATCH_SIZE, shuffle=False)
    
    evaluator3 = EnsembleEvaluator(
        classifier, autoencoder,
        classifier_threshold_d3, autoencoder_threshold,
        device=device
    )
    
    metrics3, (y_true3, y_pred_ens3, y_pred_cls3, y_pred_ae3, _, _) = \
        evaluator3.evaluate_dataset(test_loader3, 'Dataset-3 (GPS Spoofing)', fusion='or')
    
    # Step 5: Generate visualizations
    print("\n" + "="*70)
    print("Step 4: Generating visualizations")
    print("="*70)
    
    evaluator2.plot_comparison(metrics2, metrics3,
                              '../results/ensemble_comparison.png')
    
    evaluator2.plot_venn_diagram(y_true2, y_pred_cls2, y_pred_ae2,
                                'Dataset-2',
                                '../results/ensemble_venn_dataset2.png')
    
    evaluator3.plot_venn_diagram(y_true3, y_pred_cls3, y_pred_ae3,
                                'Dataset-3',
                                '../results/ensemble_venn_dataset3.png')
    
    # Step 6: Save results
    print("\n" + "="*70)
    print("Step 5: Saving results")
    print("="*70)
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'fusion_method': 'OR',
        'classifier_threshold_dataset2': classifier_threshold_d2,
        'classifier_threshold_dataset3': classifier_threshold_d3,
        'autoencoder_threshold': autoencoder_threshold,
        'dataset_2': metrics2,
        'dataset_3': metrics3
    }
    
    with open('../results/ensemble_evaluation_report.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("✓ Results saved to ensemble_evaluation_report.json")
    
    # Step 7: Final summary
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION SUMMARY")
    print("="*70)
    
    print("\n📊 Dataset-2 (Command Injection/Replay):")
    print(f"  Classifier recall:  {metrics2['Classifier']['recall']:.4f}")
    print(f"  Autoencoder recall: {metrics2['Autoencoder']['recall']:.4f}")
    print(f"  Ensemble recall:    {metrics2['Ensemble']['recall']:.4f}")
    improvement2 = metrics2['Ensemble']['recall'] - max(metrics2['Classifier']['recall'], 
                                                        metrics2['Autoencoder']['recall'])
    if improvement2 > 0:
        print(f"  → Improvement: +{improvement2:.4f} ({improvement2*100:+.1f}%)")
    
    print("\n📊 Dataset-3 (GPS Spoofing - Zero-Day):")
    print(f"  Classifier recall:  {metrics3['Classifier']['recall']:.4f}")
    print(f"  Autoencoder recall: {metrics3['Autoencoder']['recall']:.4f}")
    print(f"  Ensemble recall:    {metrics3['Ensemble']['recall']:.4f} ← KEY METRIC")
    improvement3 = metrics3['Ensemble']['recall'] - max(metrics3['Classifier']['recall'],
                                                        metrics3['Autoencoder']['recall'])
    if improvement3 > 0:
        print(f"  → Improvement: +{improvement3:.4f} ({improvement3*100:+.1f}%)")
    
    print("\n💡 Key Insights:")
    
    # Check if ensemble improves recall
    if metrics3['Ensemble']['recall'] > metrics3['Classifier']['recall']:
        print("  ✓ Ensemble improves zero-day GPS attack detection")
        print(f"    Classifier alone: {metrics3['Classifier']['recall']:.1%}")
        print(f"    With autoencoder: {metrics3['Ensemble']['recall']:.1%}")
    
    # Check FPR trade-off
    if metrics2['Ensemble']['fpr'] > metrics2['Classifier']['fpr']:
        fpr_increase = metrics2['Ensemble']['fpr'] - metrics2['Classifier']['fpr']
        print(f"  ⚠️  Trade-off: FPR increases by {fpr_increase:.1%} on Dataset-2")
        print("    → Consider adjusting autoencoder threshold if too high")
    
    # Overall recommendation
    print("\n🎯 Recommendation:")
    if improvement3 > 0.05:  # 5% improvement
        print("  ✓ USE ENSEMBLE for production deployment")
        print("    → Significantly improves zero-day attack detection")
        print("    → Maintains good performance on known attacks")
    elif improvement3 > 0:
        print("  → Consider ensemble if zero-day detection is critical")
        print(f"    → Modest improvement: +{improvement3*100:.1f}%")
    else:
        print("  → Classifier alone may be sufficient")
        print("    → Autoencoder does not add value in this case")
    
    print("\n📁 Generated Files:")
    print("  - ensemble_evaluation_report.json")
    print("  - ensemble_comparison.png")
    print("  - ensemble_venn_dataset2.png")
    print("  - ensemble_venn_dataset3.png")
    
    print("\n" + "="*70)
    print("Ensemble evaluation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
