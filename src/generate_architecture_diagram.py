"""
Generate UAV IDS Architecture Diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    """Create comprehensive UAV IDS architecture diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Color scheme
    color_input = '#E8F5E9'
    color_preprocess = '#FFF3E0'
    color_lstm = '#E3F2FD'
    color_autoencoder = '#F3E5F5'
    color_ensemble = '#FFEBEE'
    color_output = '#E8F5E9'
    
    # Title
    ax.text(5, 13.5, 'UAV Intrusion Detection System Architecture', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    ax.text(5, 13, 'Hybrid LSTM-Autoencoder Ensemble for IoV Security', 
            ha='center', va='top', fontsize=12, style='italic', color='gray')
    
    # ============ INPUT LAYER ============
    input_box = FancyBboxPatch((3.5, 11.5), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=color_input, linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 12, 'UAV Telemetry Input', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(5, 11.7, '9 Features × 10 Timesteps', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # Feature details (smaller box on the right)
    ax.text(8.5, 11.9, 'Features:', ha='left', va='top', fontsize=8, fontweight='bold')
    features = ['• GPS (lat, lon, alt)', '• Motion (vel, pitch, roll, yaw)', 
                '• Battery', '• Command ID']
    for i, feat in enumerate(features):
        ax.text(8.5, 11.6 - i*0.25, feat, ha='left', va='top', fontsize=7)
    
    # Arrow down
    ax.arrow(5, 11.5, 0, -0.5, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=1.5)
    
    # ============ PREPROCESSING ============
    preprocess_box = FancyBboxPatch((3, 10), 4, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   edgecolor='black', facecolor=color_preprocess, linewidth=2)
    ax.add_patch(preprocess_box)
    ax.text(5, 10.5, 'Data Preprocessing', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(5, 10.2, 'Normalization → Sliding Window (size=10)', 
            ha='center', va='center', fontsize=8, style='italic')
    
    # Arrow down splits into two
    ax.arrow(5, 10, 0, -0.3, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=1.5)
    
    # Split arrows
    ax.plot([5, 2.5], [9.7, 9.2], 'k-', linewidth=2)
    ax.plot([5, 7.5], [9.7, 9.2], 'k-', linewidth=2)
    
    ax.arrow(2.5, 9.2, 0, -0.2, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=1.5)
    ax.arrow(7.5, 9.2, 0, -0.2, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=1.5)
    
    # ============ LEFT PATH: LSTM CLASSIFIER ============
    ax.text(2.5, 8.8, 'Path A: Temporal Classification', ha='center', va='top', 
            fontsize=10, fontweight='bold', color='#1565C0')
    
    # LSTM Layer 1
    lstm1_box = FancyBboxPatch((1, 7.5), 3, 0.7, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#1976D2', facecolor=color_lstm, linewidth=2)
    ax.add_patch(lstm1_box)
    ax.text(2.5, 8, 'Bidirectional LSTM-1', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(2.5, 7.7, '64 hidden units', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # Arrow
    ax.arrow(2.5, 7.5, 0, -0.3, head_width=0.12, head_length=0.08, 
             fc='black', ec='black', linewidth=1.2)
    
    # LSTM Layer 2
    lstm2_box = FancyBboxPatch((1, 6.5), 3, 0.7, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#1976D2', facecolor=color_lstm, linewidth=2)
    ax.add_patch(lstm2_box)
    ax.text(2.5, 7, 'Bidirectional LSTM-2', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(2.5, 6.7, '64 hidden units', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # Arrow
    ax.arrow(2.5, 6.5, 0, -0.3, head_width=0.12, head_length=0.08, 
             fc='black', ec='black', linewidth=1.2)
    
    # Dropout
    dropout1_box = FancyBboxPatch((1.3, 5.6), 2.4, 0.5, 
                                 boxstyle="round,pad=0.05", 
                                 edgecolor='gray', facecolor='#E1F5FE', 
                                 linewidth=1, linestyle='dashed')
    ax.add_patch(dropout1_box)
    ax.text(2.5, 5.9, 'Dropout (0.3)', ha='center', va='center', fontsize=8)
    
    # Arrow
    ax.arrow(2.5, 5.6, 0, -0.2, head_width=0.12, head_length=0.08, 
             fc='black', ec='black', linewidth=1.2)
    
    # FC Layer
    fc_box = FancyBboxPatch((1, 4.5), 3, 0.7, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='#1976D2', facecolor=color_lstm, linewidth=2)
    ax.add_patch(fc_box)
    ax.text(2.5, 5, 'Fully Connected', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(2.5, 4.7, '64 → 1 neuron + Sigmoid', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # Arrow
    ax.arrow(2.5, 4.5, 0, -0.3, head_width=0.12, head_length=0.08, 
             fc='black', ec='black', linewidth=1.2)
    
    # LSTM Output
    lstm_out_box = FancyBboxPatch((1.3, 3.5), 2.4, 0.6, 
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='#1976D2', facecolor='#BBDEFB', linewidth=2)
    ax.add_patch(lstm_out_box)
    ax.text(2.5, 3.9, 'LSTM Output', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    ax.text(2.5, 3.65, 'P(attack) ∈ [0,1]', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # ============ RIGHT PATH: AUTOENCODER ============
    ax.text(7.5, 8.8, 'Path B: Anomaly Detection', ha='center', va='top', 
            fontsize=10, fontweight='bold', color='#6A1B9A')
    
    # Encoder
    encoder_box = FancyBboxPatch((6, 7.5), 3, 0.7, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='#7B1FA2', facecolor=color_autoencoder, linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(7.5, 8, 'Encoder', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(7.5, 7.7, 'Compress to latent space', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # Arrow
    ax.arrow(7.5, 7.5, 0, -0.3, head_width=0.12, head_length=0.08, 
             fc='black', ec='black', linewidth=1.2)
    
    # Latent Space
    latent_box = FancyBboxPatch((6.3, 6.5), 2.4, 0.7, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#7B1FA2', facecolor='#E1BEE7', 
                               linewidth=2, linestyle='dashed')
    ax.add_patch(latent_box)
    ax.text(7.5, 7, 'Latent Space', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(7.5, 6.7, 'Bottleneck representation', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # Arrow
    ax.arrow(7.5, 6.5, 0, -0.3, head_width=0.12, head_length=0.08, 
             fc='black', ec='black', linewidth=1.2)
    
    # Decoder
    decoder_box = FancyBboxPatch((6, 5.5), 3, 0.7, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='#7B1FA2', facecolor=color_autoencoder, linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(7.5, 6, 'Decoder', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(7.5, 5.7, 'Reconstruct input', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # Arrow
    ax.arrow(7.5, 5.5, 0, -0.2, head_width=0.12, head_length=0.08, 
             fc='black', ec='black', linewidth=1.2)
    
    # Reconstruction Error
    recon_box = FancyBboxPatch((6, 4.5), 3, 0.7, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#7B1FA2', facecolor=color_autoencoder, linewidth=2)
    ax.add_patch(recon_box)
    ax.text(7.5, 5, 'Reconstruction Error', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(7.5, 4.7, 'MSE(input, output)', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # Arrow
    ax.arrow(7.5, 4.5, 0, -0.3, head_width=0.12, head_length=0.08, 
             fc='black', ec='black', linewidth=1.2)
    
    # Autoencoder Output
    ae_out_box = FancyBboxPatch((6.3, 3.5), 2.4, 0.6, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#7B1FA2', facecolor='#E1BEE7', linewidth=2)
    ax.add_patch(ae_out_box)
    ax.text(7.5, 3.9, 'Autoencoder Output', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    ax.text(7.5, 3.65, 'Error > threshold?', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # ============ ENSEMBLE FUSION ============
    # Arrows converging
    ax.plot([2.5, 5], [3.5, 2.8], 'k-', linewidth=2)
    ax.plot([7.5, 5], [3.5, 2.8], 'k-', linewidth=2)
    ax.arrow(5, 2.8, 0, -0.2, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=1.5)
    
    # Ensemble box
    ensemble_box = FancyBboxPatch((3.5, 1.5), 3, 0.9, 
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='#C62828', facecolor=color_ensemble, linewidth=3)
    ax.add_patch(ensemble_box)
    ax.text(5, 2.1, 'Ensemble Fusion', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(5, 1.8, 'OR Logic: Detect if either path flags attack', 
            ha='center', va='center', fontsize=8, style='italic')
    
    # Arrow down
    ax.arrow(5, 1.5, 0, -0.3, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=1.5)
    
    # ============ OUTPUT ============
    output_box = FancyBboxPatch((3.5, 0.3), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#2E7D32', facecolor=color_output, linewidth=3)
    ax.add_patch(output_box)
    ax.text(5, 0.8, 'Final Prediction', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(5, 0.5, 'Normal (0) or Attack (1)', ha='center', va='center', 
            fontsize=9, style='italic')
    
    # ============ STATISTICS BOX ============
    stats_box = FancyBboxPatch((0.2, 0.3), 2.5, 2.5, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#424242', facecolor='#FAFAFA', 
                              linewidth=2, linestyle='dashed')
    ax.add_patch(stats_box)
    ax.text(1.45, 2.6, 'Model Statistics', ha='center', va='top', 
            fontsize=9, fontweight='bold', color='#424242')
    
    stats_text = [
        'LSTM Parameters: 146K',
        'Training: Mixed data',
        '   70% Normal',
        '   30% Attacks',
        'Loss: Weighted BCE',
        'Attack weight: 10×',
        '',
        'Performance:',
        '✓ 96.7% Acc (Dataset-2)',
        '✓ 98.7% Acc (Dataset-3)'
    ]
    
    y_pos = 2.3
    for line in stats_text:
        if line.startswith('✓'):
            ax.text(0.35, y_pos, line, ha='left', va='top', fontsize=7, 
                   fontweight='bold', color='#2E7D32')
        elif line == '':
            y_pos -= 0.1
            continue
        else:
            ax.text(0.35, y_pos, line, ha='left', va='top', fontsize=7)
        y_pos -= 0.2
    
    # ============ LEGEND ============
    legend_elements = [
        mpatches.Patch(facecolor=color_lstm, edgecolor='#1976D2', 
                      label='LSTM Classifier Path', linewidth=1.5),
        mpatches.Patch(facecolor=color_autoencoder, edgecolor='#7B1FA2', 
                      label='Autoencoder Path', linewidth=1.5),
        mpatches.Patch(facecolor=color_ensemble, edgecolor='#C62828', 
                      label='Ensemble Fusion', linewidth=1.5),
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             fontsize=8, framealpha=0.95)
    
    plt.tight_layout()
    
    # Save with high DPI
    output_path = '../results/uav_ids_architecture.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Architecture diagram saved: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    print("🎨 Generating UAV IDS Architecture Diagram...")
    create_architecture_diagram()
    print("✅ Done!")
