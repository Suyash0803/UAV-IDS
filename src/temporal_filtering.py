"""
Temporal Consistency Filtering for UAV IDS
==========================================

WHY TEMPORAL FILTERING REDUCES FALSE POSITIVES:
-----------------------------------------------
1. **Real attacks persist**: Actual attacks (injection, GPS spoofing) affect multiple 
   consecutive time windows because they alter the system state continuously.
   Example: GPS spoofing causes sustained coordinate changes, not just a single blip.

2. **False positives are sporadic**: Random noise, sensor glitches, or benign anomalies
   typically affect only 1-2 isolated windows before returning to normal patterns.
   Example: Brief sensor reading fluctuation that doesn't represent an attack.

3. **Sliding window overlap**: Our sequences use window_size=10 with stride=1, meaning
   consecutive predictions analyze overlapping data. A true anomaly appears in multiple
   overlapping windows, while noise affects fewer windows.

4. **Temporal correlation**: Attack signatures have temporal dependencies. Requiring
   N consecutive detections acts as a low-pass filter, smoothing out high-frequency
   (isolated) false alarms while preserving low-frequency (persistent) true attacks.

TRADE-OFF: Detection Delay
---------------------------
- Without filtering: Instant detection (N=1 window)
- With N=3: Delay of 2 additional windows (~20ms if 10ms per window)
- With N=5: Delay of 4 additional windows (~40ms)

For UAV security, a 20-40ms delay is acceptable if it significantly reduces false alarms,
which otherwise would cause:
- Unnecessary emergency responses (landing, shutdown)
- Operator alert fatigue
- System instability from frequent mode switches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from pathlib import Path


class TemporalConsistencyFilter:
    """
    Applies temporal consistency checking to IDS predictions.
    
    Raises alert ONLY if anomaly detected in N consecutive windows.
    This reduces sporadic false positives while maintaining sensitivity to persistent attacks.
    """
    
    def __init__(self, min_consecutive: int = 3):
        """
        Initialize temporal filter.
        
        Args:
            min_consecutive: Minimum consecutive windows with anomaly to raise alert (N)
        """
        self.min_consecutive = min_consecutive
        self.reset()
    
    def reset(self):
        """Reset internal state for new sequence."""
        self.history = []
    
    def filter_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply temporal consistency filtering to predictions.
        
        Args:
            predictions: Binary array of per-window predictions (0=normal, 1=attack)
                        Shape: (num_windows,)
        
        Returns:
            filtered_predictions: Binary array after temporal filtering
                                 Shape: (num_windows,)
        
        Logic:
            - For each window i, check if predictions[i-N+1:i+1] are ALL 1 (attack)
            - If yes: filtered_predictions[i] = 1 (confirmed attack)
            - If no: filtered_predictions[i] = 0 (not enough consecutive evidence)
        
        Example with min_consecutive=3:
            predictions:         [0, 1, 0, 1, 1, 1, 1, 0, 1, 1]
            filtered_predictions: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
                                            ^^^^ Only here do we have 3+ consecutive 1s
        """
        n = len(predictions)
        filtered = np.zeros(n, dtype=int)
        
        for i in range(n):
            # Look back min_consecutive windows (including current)
            start_idx = max(0, i - self.min_consecutive + 1)
            window_preds = predictions[start_idx:i+1]
            
            # Require ALL predictions in this window to be 1 (attack)
            # AND we must have at least min_consecutive predictions to check
            if len(window_preds) >= self.min_consecutive:
                if np.all(window_preds == 1):
                    filtered[i] = 1
        
        return filtered
    
    def compute_detection_delay(self, original_preds: np.ndarray, 
                                filtered_preds: np.ndarray) -> Dict[str, float]:
        """
        Compute detection delay introduced by temporal filtering.
        
        Detection delay = number of windows between first attack detection (original)
                         and first confirmed detection (filtered)
        
        Args:
            original_preds: Unfiltered predictions
            filtered_preds: Temporally filtered predictions
        
        Returns:
            Dictionary with delay statistics
        """
        # Find first attack detection in original predictions
        attack_indices = np.where(original_preds == 1)[0]
        if len(attack_indices) == 0:
            return {'mean_delay': 0, 'max_delay': 0, 'num_attacks': 0}
        
        delays = []
        
        # For each attack in original, find when it's confirmed in filtered
        i = 0
        while i < len(attack_indices):
            first_detect = attack_indices[i]
            
            # Find when this attack is confirmed in filtered predictions
            # Search forward from first detection
            confirmed_idx = None
            for j in range(first_detect, len(filtered_preds)):
                if filtered_preds[j] == 1:
                    confirmed_idx = j
                    break
            
            if confirmed_idx is not None:
                delay = confirmed_idx - first_detect
                delays.append(delay)
            
            # Skip to next attack segment (avoid counting same attack multiple times)
            i += 1
            while i < len(attack_indices) and attack_indices[i] <= first_detect + 10:
                i += 1
        
        if len(delays) == 0:
            return {'mean_delay': float('inf'), 'max_delay': float('inf'), 'num_attacks': 0}
        
        return {
            'mean_delay': np.mean(delays),
            'max_delay': np.max(delays),
            'min_delay': np.min(delays),
            'median_delay': np.median(delays),
            'num_attacks': len(delays)
        }


class TemporalFilteringEvaluator:
    """
    Evaluates impact of temporal filtering on IDS performance.
    
    Compares metrics with and without temporal consistency, and analyzes
    the trade-off between FPR reduction and detection delay.
    """
    
    def __init__(self, results_dir: str = '../results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def evaluate_with_temporal_filtering(self, 
                                         predictions: np.ndarray,
                                         true_labels: np.ndarray,
                                         n_values: List[int] = [1, 2, 3, 4, 5, 7, 10],
                                         dataset_name: str = 'Dataset') -> Dict:
        """
        Evaluate IDS performance across different temporal filtering thresholds.
        
        Args:
            predictions: Original model predictions (0=normal, 1=attack)
            true_labels: Ground truth labels
            n_values: List of N values (min consecutive windows) to test
            dataset_name: Name of dataset for labeling
        
        Returns:
            Dictionary with results for each N value
        """
        results = {}
        
        for N in n_values:
            print(f"\n{'='*60}")
            print(f"Evaluating with N = {N} consecutive windows")
            print(f"{'='*60}")
            
            # Apply temporal filtering
            filter_obj = TemporalConsistencyFilter(min_consecutive=N)
            filtered_preds = filter_obj.filter_predictions(predictions)
            
            # Compute metrics
            metrics = self._compute_metrics(true_labels, filtered_preds)
            
            # Compute detection delay
            delay_stats = filter_obj.compute_detection_delay(predictions, filtered_preds)
            
            # Combine results
            results[N] = {
                'metrics': metrics,
                'delay': delay_stats,
                'num_alerts': int(np.sum(filtered_preds)),
                'original_alerts': int(np.sum(predictions))
            }
            
            # Print summary
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1']:.4f}")
            print(f"FPR:       {metrics['fpr']:.4f}")
            print(f"Alerts: {results[N]['num_alerts']} (original: {results[N]['original_alerts']})")
            if delay_stats['num_attacks'] > 0:
                print(f"Detection Delay: {delay_stats['mean_delay']:.2f} windows (mean), "
                      f"{delay_stats['max_delay']:.0f} (max)")
            else:
                print(f"Detection Delay: No attacks confirmed with N={N}")
        
        return results
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute classification metrics."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    def plot_temporal_filtering_impact(self, results: Dict, dataset_name: str):
        """
        Plot impact of temporal filtering parameter N on metrics and delay.
        
        Creates 4-panel figure:
        1. FPR vs N (primary goal: reduce false positives)
        2. Recall vs N (monitor: ensure we don't lose true attack detection)
        3. Detection Delay vs N (trade-off: acceptable latency increase)
        4. F1-Score vs N (overall: balance precision and recall)
        """
        n_values = sorted(results.keys())
        
        # Extract metrics
        fprs = [results[n]['metrics']['fpr'] for n in n_values]
        recalls = [results[n]['metrics']['recall'] for n in n_values]
        f1s = [results[n]['metrics']['f1'] for n in n_values]
        precisions = [results[n]['metrics']['precision'] for n in n_values]
        mean_delays = [results[n]['delay']['mean_delay'] if results[n]['delay']['num_attacks'] > 0 
                       else float('nan') for n in n_values]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Temporal Filtering Impact - {dataset_name}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: FPR vs N (most important - goal is to reduce FPR)
        ax1 = axes[0, 0]
        ax1.plot(n_values, fprs, 'o-', color='red', linewidth=2, markersize=8)
        ax1.set_xlabel('N (Consecutive Windows Required)', fontsize=12)
        ax1.set_ylabel('False Positive Rate', fontsize=12)
        ax1.set_title('FPR Reduction with Temporal Filtering', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(fprs) * 1.1)
        
        # Annotate FPR values
        for i, (n, fpr) in enumerate(zip(n_values, fprs)):
            ax1.annotate(f'{fpr:.3f}', xy=(n, fpr), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9)
        
        # Plot 2: Recall vs N (ensure we don't lose attack detection)
        ax2 = axes[0, 1]
        ax2.plot(n_values, recalls, 'o-', color='green', linewidth=2, markersize=8)
        ax2.set_xlabel('N (Consecutive Windows Required)', fontsize=12)
        ax2.set_ylabel('Recall (True Positive Rate)', fontsize=12)
        ax2.set_title('Attack Detection Sensitivity', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        # Annotate recall values
        for i, (n, rec) in enumerate(zip(n_values, recalls)):
            ax2.annotate(f'{rec:.3f}', xy=(n, rec), xytext=(0, -15),
                        textcoords='offset points', ha='center', fontsize=9)
        
        # Plot 3: Detection Delay vs N (trade-off analysis)
        ax3 = axes[1, 0]
        valid_delays = [(n, d) for n, d in zip(n_values, mean_delays) if not np.isnan(d)]
        if valid_delays:
            ns, delays = zip(*valid_delays)
            ax3.plot(ns, delays, 'o-', color='orange', linewidth=2, markersize=8)
            ax3.set_xlabel('N (Consecutive Windows Required)', fontsize=12)
            ax3.set_ylabel('Mean Detection Delay (windows)', fontsize=12)
            ax3.set_title('Detection Latency Trade-off', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Annotate delay values
            for n, delay in zip(ns, delays):
                ax3.annotate(f'{delay:.1f}', xy=(n, delay), xytext=(0, 10),
                            textcoords='offset points', ha='center', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No attack confirmations', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_xlabel('N (Consecutive Windows Required)', fontsize=12)
            ax3.set_ylabel('Mean Detection Delay (windows)', fontsize=12)
        
        # Plot 4: F1-Score and Precision vs N (overall performance)
        ax4 = axes[1, 1]
        ax4.plot(n_values, f1s, 'o-', color='blue', linewidth=2, markersize=8, label='F1-Score')
        ax4.plot(n_values, precisions, 's--', color='purple', linewidth=2, markersize=8, 
                label='Precision')
        ax4.set_xlabel('N (Consecutive Windows Required)', fontsize=12)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Overall Performance Metrics', fontweight='bold')
        ax4.legend(loc='best', fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'temporal_filtering_{dataset_name.lower().replace(" ", "_")}.png'
        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot: {filepath}")
        
        plt.close()
    
    def plot_comparison_before_after(self, y_true: np.ndarray, 
                                    original_preds: np.ndarray,
                                    filtered_preds: np.ndarray,
                                    N: int,
                                    dataset_name: str,
                                    max_samples: int = 500):
        """
        Visualize temporal filtering effect on predictions over time.
        
        Shows:
        - Ground truth labels
        - Original predictions (before filtering)
        - Filtered predictions (after temporal consistency check)
        
        Highlights where filtering removes false positives.
        """
        # Limit samples for visibility
        n_samples = min(max_samples, len(y_true))
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        x = np.arange(n_samples)
        
        # Plot ground truth as background
        attack_mask = y_true[:n_samples] == 1
        ax.fill_between(x, 0, 1, where=attack_mask, alpha=0.2, color='red', 
                        label='True Attack Regions')
        
        # Plot original predictions
        ax.scatter(x, original_preds[:n_samples] * 0.6, alpha=0.6, c='orange', 
                  marker='x', s=50, label='Original Predictions')
        
        # Plot filtered predictions
        ax.scatter(x, filtered_preds[:n_samples] * 0.4, alpha=0.8, c='blue', 
                  marker='o', s=30, label=f'Filtered Predictions (N={N})')
        
        ax.set_xlabel('Time Window Index', fontsize=12)
        ax.set_ylabel('Prediction', fontsize=12)
        ax.set_title(f'Temporal Filtering Effect - {dataset_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_yticks([0, 0.4, 0.6, 1])
        ax.set_yticklabels(['Normal', f'Filtered\nAttack', f'Original\nAttack', ''])
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'temporal_filtering_timeline_{dataset_name.lower().replace(" ", "_")}_N{N}.png'
        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved timeline plot: {filepath}")
        
        plt.close()
    
    def generate_comparison_table(self, results: Dict, dataset_name: str) -> pd.DataFrame:
        """
        Generate comparison table of metrics across different N values.
        """
        rows = []
        for N in sorted(results.keys()):
            metrics = results[N]['metrics']
            delay = results[N]['delay']
            
            row = {
                'N': N,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'FPR': f"{metrics['fpr']:.4f}",
                'Alerts': results[N]['num_alerts'],
                'Mean Delay': f"{delay['mean_delay']:.2f}" if delay['num_attacks'] > 0 else 'N/A'
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        filename = f'temporal_filtering_comparison_{dataset_name.lower().replace(" ", "_")}.csv'
        filepath = self.results_dir / filename
        df.to_csv(filepath, index=False)
        print(f"✓ Saved comparison table: {filepath}")
        
        # Print table
        print(f"\n{'='*80}")
        print(f"Temporal Filtering Results - {dataset_name}")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        print(f"{'='*80}\n")
        
        return df


def main():
    """
    Example usage: Apply temporal filtering to ensemble IDS predictions.
    
    This demonstrates how to integrate temporal consistency into the evaluation pipeline.
    """
    print("="*80)
    print("UAV IDS - Temporal Consistency Filtering")
    print("="*80)
    print("\nThis script demonstrates temporal filtering to reduce false positives.")
    print("It requires existing ensemble predictions from ensemble_evaluate.py")
    print("\nTo integrate with your ensemble IDS:")
    print("1. Load ensemble predictions from evaluate_autoencoder.py or ensemble_evaluate.py")
    print("2. Apply TemporalConsistencyFilter with desired N value")
    print("3. Compare metrics before and after filtering")
    print("\nExample code:")
    print("-" * 80)
    print("""
    # Load predictions (example)
    # ensemble_preds = np.load('results/ensemble_predictions.npy')
    # true_labels = np.load('results/true_labels.npy')
    
    # Apply temporal filtering
    filter_obj = TemporalConsistencyFilter(min_consecutive=3)
    filtered_preds = filter_obj.filter_predictions(ensemble_preds)
    
    # Evaluate impact
    evaluator = TemporalFilteringEvaluator()
    results = evaluator.evaluate_with_temporal_filtering(
        predictions=ensemble_preds,
        true_labels=true_labels,
        n_values=[1, 2, 3, 4, 5, 7, 10],
        dataset_name='Dataset-2'
    )
    
    # Plot results
    evaluator.plot_temporal_filtering_impact(results, 'Dataset-2')
    evaluator.plot_comparison_before_after(true_labels, ensemble_preds, 
                                          filtered_preds, N=3, dataset_name='Dataset-2')
    """)
    print("-" * 80)
    print("\n✓ Temporal filtering module ready!")
    print("  Run ensemble_evaluate_temporal.py to apply to your IDS datasets.")


if __name__ == '__main__':
    main()
