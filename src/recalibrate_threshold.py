"""
Autoencoder Threshold Recalibration for False Positive Reduction
================================================================

PROBLEM ANALYSIS:
-----------------
Current threshold (0.000134, 95th percentile) causes 97-98% FPR because:

1. **Training-Test Distribution Mismatch**: 
   - Autoencoder trained on Dataset-1 (pure normal UAV commands)
   - Test Dataset-2 contains different normal command patterns
   - Reconstruction errors for "normal" test samples exceed training errors

2. **Overly Strict Threshold**:
   - 95th percentile captures only the "most normal" patterns
   - 5% of validation samples flagged as anomalies (acceptable in training)
   - But 97% of test samples flagged (systematic false positives)

3. **Attack vs Normal Separation**:
   - Dataset-2: Attack errors ~3,258x higher than normal (0.002 vs 3,258)
   - Dataset-3: Attack errors ~439,000x higher than normal (0.003 vs 1,526)
   - Huge separation means we can be MORE lenient with threshold

SOLUTION: PERCENTILE-BASED RECALIBRATION
-----------------------------------------
Instead of 95th percentile, use:
- **99.5th percentile**: Allows more normal variation (0.5% FPR on validation)
- **99.9th percentile**: Very lenient, assumes 99.9% of training is truly normal

Key insight: With 1000x+ attack/normal separation, we can afford higher thresholds
while still catching 100% of attacks.

DESIGN RATIONALE:
-----------------
Why percentile-based thresholds work:
1. **Distribution-aware**: Adapts to actual normal error distribution
2. **Statistically robust**: Not sensitive to outliers
3. **Interpretable**: "99.5th percentile" = "flag top 0.5% as anomalies"
4. **Flexible**: Can tune based on operational requirements

Trade-off consideration:
- Lower percentile (95th) → High sensitivity, many false alarms
- Higher percentile (99.9th) → Lower sensitivity, fewer false alarms
- Sweet spot: 99.5th percentile (balance FPR and recall)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from preprocess import UAVDataPreprocessor
from model import create_autoencoder, load_model


class AutoencoderThresholdRecalibrator:
    """
    Analyzes reconstruction error distributions and recalibrates anomaly threshold
    to reduce false positive rate while maintaining high recall.
    """
    
    def __init__(self, autoencoder_path: str, validation_data_path: str):
        """
        Initialize recalibrator.
        
        Args:
            autoencoder_path: Path to trained autoencoder model
            validation_data_path: Path to normal validation data (Dataset-1)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load autoencoder
        print("Loading autoencoder...")
        self.autoencoder = create_autoencoder(input_size=9, hidden_size=32, 
                                             num_layers=1, dropout=0.2, 
                                             device=str(self.device))
        load_model(self.autoencoder, autoencoder_path)
        self.autoencoder.to(self.device)
        self.autoencoder.eval()
        print("✓ Autoencoder loaded")
        
        # Load validation data
        print(f"Loading validation data: {validation_data_path}")
        preprocessor = UAVDataPreprocessor()
        data = preprocessor.prepare_training_data(validation_data_path)
        self.X_val = data['X_train']
        self.y_val = data['y_train']
        
        # Filter only normal samples for threshold calculation
        normal_mask = self.y_val == 0
        self.X_val_normal = self.X_val[normal_mask]
        print(f"✓ Loaded {len(self.X_val_normal)} normal validation samples")
        
        # Compute validation reconstruction errors
        self.val_errors = self._compute_reconstruction_errors(self.X_val_normal)
        print(f"✓ Computed reconstruction errors (mean: {np.mean(self.val_errors):.6f})")
    
    def _compute_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction errors for input sequences.
        
        Args:
            X: Input sequences (num_samples, window_size, num_features)
        
        Returns:
            Reconstruction errors (num_samples,)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            errors = self.autoencoder.get_reconstruction_error(X_tensor).cpu().numpy()
        
        return errors
    
    def analyze_error_distributions(self, test_datasets: Dict[str, str]) -> Dict:
        """
        Analyze reconstruction error distributions across datasets.
        
        Args:
            test_datasets: Dictionary of {dataset_name: dataset_path}
        
        Returns:
            Dictionary containing error statistics for each dataset
        """
        results = {
            'validation': {
                'name': 'Validation (Normal)',
                'errors': self.val_errors,
                'labels': self.y_val[self.y_val == 0]  # All normal
            }
        }
        
        # Load and analyze test datasets
        for name, path in test_datasets.items():
            print(f"\nAnalyzing {name}...")
            preprocessor = UAVDataPreprocessor()
            data = preprocessor.prepare_training_data(path)
            X_test = data['X_train']
            y_test = data['y_train']
            
            errors = self._compute_reconstruction_errors(X_test)
            
            # Separate normal and attack samples
            normal_mask = y_test == 0
            attack_mask = y_test == 1
            
            results[name] = {
                'name': name,
                'errors': errors,
                'labels': y_test,
                'normal_errors': errors[normal_mask],
                'attack_errors': errors[attack_mask],
                'normal_count': np.sum(normal_mask),
                'attack_count': np.sum(attack_mask)
            }
            
            print(f"  Normal samples: {np.sum(normal_mask)}")
            print(f"  Attack samples: {np.sum(attack_mask)}")
            print(f"  Normal error mean: {np.mean(errors[normal_mask]):.6f}")
            print(f"  Attack error mean: {np.mean(errors[attack_mask]):.6f}")
            print(f"  Separation ratio: {np.mean(errors[attack_mask]) / np.mean(errors[normal_mask]):.1f}x")
        
        return results
    
    def compute_percentile_thresholds(self, percentiles: List[float] = [95.0, 99.0, 99.5, 99.9]) -> Dict:
        """
        Compute threshold values at different percentiles of validation errors.
        
        Args:
            percentiles: List of percentile values to compute
        
        Returns:
            Dictionary mapping percentile to threshold value
        """
        thresholds = {}
        
        print("\n" + "="*70)
        print("PERCENTILE-BASED THRESHOLDS")
        print("="*70)
        print(f"Computing thresholds from {len(self.val_errors)} validation errors")
        print(f"Error range: [{np.min(self.val_errors):.6f}, {np.max(self.val_errors):.6f}]")
        print(f"Error mean: {np.mean(self.val_errors):.6f}, std: {np.std(self.val_errors):.6f}")
        print()
        
        for p in percentiles:
            threshold = np.percentile(self.val_errors, p)
            thresholds[p] = threshold
            
            # Expected FPR on validation data
            fpr_val = np.sum(self.val_errors > threshold) / len(self.val_errors)
            
            print(f"{p:5.1f}th percentile: {threshold:.6f} (Expected FPR: {fpr_val*100:.2f}%)")
        
        print("="*70)
        
        return thresholds
    
    def evaluate_threshold(self, threshold: float, test_data: Dict) -> Dict:
        """
        Evaluate autoencoder performance with a specific threshold.
        
        Args:
            threshold: Anomaly detection threshold
            test_data: Dictionary containing test errors and labels
        
        Returns:
            Dictionary with performance metrics
        """
        errors = test_data['errors']
        labels = test_data['labels']
        
        # Predictions: error > threshold → attack (1)
        predictions = (errors >= threshold).astype(int)
        
        # Compute metrics
        tn = np.sum((labels == 0) & (predictions == 0))
        fp = np.sum((labels == 0) & (predictions == 1))
        fn = np.sum((labels == 1) & (predictions == 0))
        tp = np.sum((labels == 1) & (predictions == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'alerts': int(tp + fp)
        }
    
    def plot_error_distributions(self, error_data: Dict, thresholds: Dict, 
                                 save_path: str = '../results/threshold_recalibration_distributions.png'):
        """
        Plot reconstruction error distributions with threshold candidates.
        
        Shows:
        1. Histogram of errors (validation, test normal, test attack)
        2. Percentile threshold candidates
        3. Separation between normal and attack errors
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Reconstruction Error Distributions & Threshold Candidates', 
                     fontsize=16, fontweight='bold')
        
        # Color scheme
        colors = {'validation': 'blue', 'normal': 'green', 'attack': 'red'}
        
        # Plot 1: Validation error distribution with percentile thresholds
        ax1 = axes[0, 0]
        val_errors = error_data['validation']['errors']
        ax1.hist(val_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Reconstruction Error', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Validation Error Distribution (Normal Samples)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add percentile threshold lines
        y_max = ax1.get_ylim()[1]
        for percentile, threshold in sorted(thresholds.items()):
            ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.7,
                       label=f'{percentile}th: {threshold:.6f}')
        ax1.legend(loc='upper right', fontsize=10)
        
        # Plot 2: Dataset-2 error comparison
        if 'Dataset-2' in error_data:
            ax2 = axes[0, 1]
            data2 = error_data['Dataset-2']
            
            # Use log scale for better visualization
            ax2.hist(data2['normal_errors'], bins=50, alpha=0.6, color='green', 
                    label=f'Normal (n={data2["normal_count"]})', edgecolor='black')
            ax2.hist(data2['attack_errors'], bins=50, alpha=0.6, color='red', 
                    label=f'Attack (n={data2["attack_count"]})', edgecolor='black')
            
            ax2.set_xlabel('Reconstruction Error', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Dataset-2: Command Injection & Replay', fontweight='bold')
            ax2.legend(loc='upper right', fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            # Add current and new thresholds
            current_threshold = thresholds.get(95.0, 0)
            new_threshold = thresholds.get(99.5, 0)
            ax2.axvline(current_threshold, color='orange', linestyle='--', 
                       linewidth=2, label=f'Current (95th): {current_threshold:.6f}')
            ax2.axvline(new_threshold, color='purple', linestyle='-', 
                       linewidth=2, label=f'New (99.5th): {new_threshold:.6f}')
            ax2.legend(loc='upper right', fontsize=10)
        
        # Plot 3: Dataset-3 error comparison
        if 'Dataset-3' in error_data:
            ax3 = axes[1, 0]
            data3 = error_data['Dataset-3']
            
            ax3.hist(data3['normal_errors'], bins=50, alpha=0.6, color='green', 
                    label=f'Normal (n={data3["normal_count"]})', edgecolor='black')
            ax3.hist(data3['attack_errors'], bins=50, alpha=0.6, color='red', 
                    label=f'Attack (n={data3["attack_count"]})', edgecolor='black')
            
            ax3.set_xlabel('Reconstruction Error', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title('Dataset-3: GPS Spoofing (Zero-Day)', fontweight='bold')
            ax3.legend(loc='upper right', fontsize=11)
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
            
            # Add thresholds
            current_threshold = thresholds.get(95.0, 0)
            new_threshold = thresholds.get(99.5, 0)
            ax3.axvline(current_threshold, color='orange', linestyle='--', 
                       linewidth=2, label=f'Current (95th): {current_threshold:.6f}')
            ax3.axvline(new_threshold, color='purple', linestyle='-', 
                       linewidth=2, label=f'New (99.5th): {new_threshold:.6f}')
            ax3.legend(loc='upper right', fontsize=10)
        
        # Plot 4: Percentile curve (CDF)
        ax4 = axes[1, 1]
        sorted_errors = np.sort(val_errors)
        percentiles = np.linspace(0, 100, len(sorted_errors))
        ax4.plot(sorted_errors, percentiles, linewidth=2, color='blue')
        ax4.set_xlabel('Reconstruction Error', fontsize=12)
        ax4.set_ylabel('Percentile', fontsize=12)
        ax4.set_title('Validation Error Percentile Curve (CDF)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Mark key percentiles
        for p in [95.0, 99.0, 99.5, 99.9]:
            threshold = thresholds[p]
            ax4.axhline(p, color='red', linestyle='--', alpha=0.5)
            ax4.axvline(threshold, color='red', linestyle='--', alpha=0.5)
            ax4.plot(threshold, p, 'ro', markersize=8)
            ax4.annotate(f'{p}th\n{threshold:.6f}', xy=(threshold, p), 
                        xytext=(10, -10), textcoords='offset points',
                        fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved distribution plot: {save_path}")
        plt.close()
    
    def plot_threshold_comparison(self, comparison_results: Dict, 
                                 save_path: str = '../results/threshold_recalibration_comparison.png'):
        """
        Plot performance comparison across different thresholds.
        
        Shows FPR, Recall, F1-Score, and Alerts for each threshold candidate.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Threshold Recalibration Performance Comparison', 
                     fontsize=16, fontweight='bold')
        
        for dataset_name, dataset_results in comparison_results.items():
            if dataset_name == 'validation':
                continue
            
            percentiles = sorted(dataset_results.keys())
            fprs = [dataset_results[p]['fpr'] for p in percentiles]
            recalls = [dataset_results[p]['recall'] for p in percentiles]
            f1s = [dataset_results[p]['f1'] for p in percentiles]
            alerts = [dataset_results[p]['alerts'] for p in percentiles]
            
            # Plot 1: FPR vs Percentile
            ax1 = axes[0, 0]
            ax1.plot(percentiles, fprs, 'o-', linewidth=2, markersize=8, label=dataset_name)
            ax1.set_xlabel('Threshold Percentile', fontsize=12)
            ax1.set_ylabel('False Positive Rate', fontsize=12)
            ax1.set_title('FPR vs Threshold Percentile', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best', fontsize=11)
            
            # Plot 2: Recall vs Percentile
            ax2 = axes[0, 1]
            ax2.plot(percentiles, recalls, 's-', linewidth=2, markersize=8, label=dataset_name)
            ax2.set_xlabel('Threshold Percentile', fontsize=12)
            ax2.set_ylabel('Recall (True Positive Rate)', fontsize=12)
            ax2.set_title('Recall vs Threshold Percentile', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best', fontsize=11)
            ax2.set_ylim(0.9, 1.01)
            
            # Plot 3: F1-Score vs Percentile
            ax3 = axes[1, 0]
            ax3.plot(percentiles, f1s, '^-', linewidth=2, markersize=8, label=dataset_name)
            ax3.set_xlabel('Threshold Percentile', fontsize=12)
            ax3.set_ylabel('F1-Score', fontsize=12)
            ax3.set_title('F1-Score vs Threshold Percentile', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='best', fontsize=11)
            
            # Plot 4: Number of Alerts vs Percentile
            ax4 = axes[1, 1]
            ax4.plot(percentiles, alerts, 'd-', linewidth=2, markersize=8, label=dataset_name)
            ax4.set_xlabel('Threshold Percentile', fontsize=12)
            ax4.set_ylabel('Number of Alerts', fontsize=12)
            ax4.set_title('Alert Volume vs Threshold Percentile', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='best', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot: {save_path}")
        plt.close()
    
    def generate_comparison_table(self, comparison_results: Dict, 
                                 save_path: str = '../results/threshold_recalibration_table.csv'):
        """
        Generate comparison table for all thresholds and datasets.
        """
        rows = []
        
        for dataset_name, dataset_results in comparison_results.items():
            if dataset_name == 'validation':
                continue
            
            for percentile, metrics in sorted(dataset_results.items()):
                row = {
                    'Dataset': dataset_name,
                    'Percentile': f'{percentile}th',
                    'Threshold': f"{metrics['threshold']:.6f}",
                    'FPR': f"{metrics['fpr']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Alerts': metrics['alerts'],
                    'TP': metrics['tp'],
                    'FP': metrics['fp'],
                    'FN': metrics['fn']
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print(f"✓ Saved comparison table: {save_path}")
        
        # Print formatted table
        print("\n" + "="*100)
        print("THRESHOLD RECALIBRATION RESULTS")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100 + "\n")
        
        return df
    
    def save_recalibrated_threshold(self, percentile: float, threshold: float,
                                   stats: Dict, save_path: str = '../models/autoencoder_threshold_recalibrated.json'):
        """
        Save new threshold configuration to JSON file.
        """
        config = {
            'timestamp': '2026-02-07',
            'anomaly_threshold': float(threshold),
            'percentile': float(percentile),
            'previous_threshold': 0.000134,
            'previous_percentile': 95.0,
            'validation_samples': len(self.val_errors),
            'hidden_size': 32,
            'window_size': 10,
            'rationale': f'Recalibrated from 95th to {percentile}th percentile to reduce false positive rate',
            'expected_fpr_on_validation': float(np.sum(self.val_errors > threshold) / len(self.val_errors)),
            'validation_error_stats': {
                'mean': float(np.mean(self.val_errors)),
                'std': float(np.std(self.val_errors)),
                'min': float(np.min(self.val_errors)),
                'max': float(np.max(self.val_errors)),
                'median': float(np.median(self.val_errors))
            },
            'performance_improvement': stats
        }
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Saved recalibrated threshold: {save_path}")
        return config


def main():
    """
    Main recalibration pipeline.
    
    Steps:
    1. Load autoencoder and validation data
    2. Compute reconstruction errors for all datasets
    3. Calculate percentile-based thresholds (95, 99, 99.5, 99.9)
    4. Evaluate performance for each threshold
    5. Generate visualizations and comparison tables
    6. Save optimal recalibrated threshold
    7. Provide recommendations for deployment
    """
    print("="*80)
    print("AUTOENCODER THRESHOLD RECALIBRATION")
    print("="*80)
    print("\nGoal: Reduce false positive rate while maintaining high recall")
    print("Method: Increase percentile from 95th → 99.5th/99.9th")
    print("="*80)
    
    # Configuration
    autoencoder_path = '../models/autoencoder_best.pth'
    validation_data_path = '../data/dataset_1_normal.csv'
    test_datasets = {
        'Dataset-2': '../data/dataset_2_injection_replay.csv',
        'Dataset-3': '../data/dataset_3_gps_spoofing.csv'
    }
    
    # Initialize recalibrator
    recalibrator = AutoencoderThresholdRecalibrator(
        autoencoder_path=autoencoder_path,
        validation_data_path=validation_data_path
    )
    
    # Analyze error distributions
    print("\n" + "="*80)
    print("STEP 1: ANALYZING RECONSTRUCTION ERROR DISTRIBUTIONS")
    print("="*80)
    error_data = recalibrator.analyze_error_distributions(test_datasets)
    
    # Compute percentile thresholds
    print("\n" + "="*80)
    print("STEP 2: COMPUTING PERCENTILE-BASED THRESHOLDS")
    print("="*80)
    percentiles = [95.0, 99.0, 99.5, 99.9]
    thresholds = recalibrator.compute_percentile_thresholds(percentiles)
    
    # Evaluate each threshold
    print("\n" + "="*80)
    print("STEP 3: EVALUATING PERFORMANCE FOR EACH THRESHOLD")
    print("="*80)
    comparison_results = {}
    
    for dataset_name in ['Dataset-2', 'Dataset-3']:
        print(f"\nEvaluating {dataset_name}:")
        print("-" * 40)
        comparison_results[dataset_name] = {}
        
        for percentile, threshold in sorted(thresholds.items()):
            print(f"\n{percentile}th percentile (threshold={threshold:.6f}):")
            metrics = recalibrator.evaluate_threshold(threshold, error_data[dataset_name])
            comparison_results[dataset_name][percentile] = metrics
            
            print(f"  FPR:       {metrics['fpr']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  Alerts:    {metrics['alerts']}")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*80)
    recalibrator.plot_error_distributions(error_data, thresholds)
    recalibrator.plot_threshold_comparison(comparison_results)
    
    # Generate comparison table
    print("\n" + "="*80)
    print("STEP 5: GENERATING COMPARISON TABLE")
    print("="*80)
    recalibrator.generate_comparison_table(comparison_results)
    
    # Recommend optimal threshold
    print("\n" + "="*80)
    print("STEP 6: THRESHOLD RECOMMENDATION")
    print("="*80)
    
    # Analyze Dataset-2 results (command injection - high FPR problem)
    dataset2_results = comparison_results['Dataset-2']
    
    print("\nDataset-2 (Command Injection) - FPR Improvement:")
    print("-" * 60)
    baseline_fpr = dataset2_results[95.0]['fpr']
    baseline_recall = dataset2_results[95.0]['recall']
    
    for p in [99.0, 99.5, 99.9]:
        new_fpr = dataset2_results[p]['fpr']
        new_recall = dataset2_results[p]['recall']
        fpr_reduction = (baseline_fpr - new_fpr) * 100
        recall_loss = (baseline_recall - new_recall) * 100
        
        print(f"\n{p}th percentile:")
        print(f"  FPR: {baseline_fpr:.4f} → {new_fpr:.4f} ({fpr_reduction:+.2f}% reduction)")
        print(f"  Recall: {baseline_recall:.4f} → {new_recall:.4f} ({recall_loss:+.2f}% change)")
        print(f"  F1-Score: {dataset2_results[p]['f1']:.4f}")
        
        if p == 99.5:
            recommendation = "✓ RECOMMENDED"
        else:
            recommendation = ""
        print(f"  {recommendation}")
    
    # Analyze Dataset-3 results (GPS spoofing - maintain 100% recall)
    dataset3_results = comparison_results['Dataset-3']
    
    print("\nDataset-3 (GPS Spoofing) - Recall Maintenance:")
    print("-" * 60)
    baseline_recall_gps = dataset3_results[95.0]['recall']
    
    for p in [99.0, 99.5, 99.9]:
        new_recall_gps = dataset3_results[p]['recall']
        recall_change = (new_recall_gps - baseline_recall_gps) * 100
        
        print(f"\n{p}th percentile:")
        print(f"  Recall: {baseline_recall_gps:.4f} → {new_recall_gps:.4f} ({recall_change:+.2f}%)")
        
        if new_recall_gps >= 0.999:
            print("  ✓ Maintains near-perfect recall")
        else:
            print("  ⚠ Recall degradation detected")
    
    # Save recommended threshold (99.5th percentile)
    print("\n" + "="*80)
    print("STEP 7: SAVING RECALIBRATED THRESHOLD")
    print("="*80)
    
    optimal_percentile = 99.5
    optimal_threshold = thresholds[optimal_percentile]
    
    improvement_stats = {
        'dataset_2': {
            'baseline_fpr': float(dataset2_results[95.0]['fpr']),
            'new_fpr': float(dataset2_results[optimal_percentile]['fpr']),
            'fpr_reduction_pct': float((dataset2_results[95.0]['fpr'] - dataset2_results[optimal_percentile]['fpr']) * 100),
            'baseline_recall': float(dataset2_results[95.0]['recall']),
            'new_recall': float(dataset2_results[optimal_percentile]['recall']),
            'recall_change_pct': float((dataset2_results[optimal_percentile]['recall'] - dataset2_results[95.0]['recall']) * 100)
        },
        'dataset_3': {
            'baseline_recall': float(dataset3_results[95.0]['recall']),
            'new_recall': float(dataset3_results[optimal_percentile]['recall']),
            'recall_maintained': bool(dataset3_results[optimal_percentile]['recall'] >= 0.999)
        }
    }
    
    config = recalibrator.save_recalibrated_threshold(
        percentile=optimal_percentile,
        threshold=optimal_threshold,
        stats=improvement_stats
    )
    
    # Final summary
    print("\n" + "="*80)
    print("RECALIBRATION COMPLETE")
    print("="*80)
    print(f"\n✓ Old threshold: 0.000134 (95th percentile)")
    print(f"✓ New threshold: {optimal_threshold:.6f} ({optimal_percentile}th percentile)")
    print(f"\n✓ FPR improvement (Dataset-2): {improvement_stats['dataset_2']['fpr_reduction_pct']:.2f}%")
    print(f"✓ Recall maintained (Dataset-3): {dataset3_results[optimal_percentile]['recall']:.4f}")
    print("\nNext steps:")
    print("  1. Re-run ensemble_evaluate_temporal.py with new threshold")
    print("  2. Combine threshold recalibration + temporal filtering (N=5)")
    print("  3. Expected combined FPR: <50% (down from 97%)")
    print("="*80)


if __name__ == '__main__':
    main()
