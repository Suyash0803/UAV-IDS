"""
Comprehensive Results Viewer
Displays all result graphs in organized grid layouts
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import os

def display_all_graphs():
    """Display all result graphs in organized subplots"""
    
    results_dir = Path('../results')
    
    # Organize graphs by category
    graphs = {
        'Training': [
            'training_history.png'
        ],
        'Threshold Analysis': [
            'threshold_recalibration_comparison.png',
            'threshold_recalibration_distributions.png'
        ],
        'Temporal Filtering - Dataset 2': [
            'temporal_filtering_dataset-2_classifier.png',
            'temporal_filtering_dataset-2_autoencoder.png',
            'temporal_filtering_dataset-2_ensemble.png'
        ],
        'Temporal Filtering - Dataset 3': [
            'temporal_filtering_dataset-3_classifier.png',
            'temporal_filtering_dataset-3_autoencoder.png',
            'temporal_filtering_dataset-3_ensemble.png'
        ],
        'Temporal Timeline - Dataset 2': [
            'temporal_filtering_timeline_dataset-2_ensemble_N1.png',
            'temporal_filtering_timeline_dataset-2_ensemble_N3.png',
            'temporal_filtering_timeline_dataset-2_ensemble_N5.png'
        ],
        'Temporal Timeline - Dataset 3': [
            'temporal_filtering_timeline_dataset-3_ensemble_N1.png',
            'temporal_filtering_timeline_dataset-3_ensemble_N3.png',
            'temporal_filtering_timeline_dataset-3_ensemble_N5.png'
        ]
    }
    
    # Display each category in a separate figure
    for category, files in graphs.items():
        existing_files = [f for f in files if (results_dir / f).exists()]
        
        if not existing_files:
            print(f"⚠️  No graphs found for: {category}")
            continue
        
        n_graphs = len(existing_files)
        
        # Determine grid layout
        if n_graphs == 1:
            rows, cols = 1, 1
            figsize = (12, 8)
        elif n_graphs == 2:
            rows, cols = 1, 2
            figsize = (20, 8)
        elif n_graphs == 3:
            rows, cols = 1, 3
            figsize = (24, 6)
        elif n_graphs <= 4:
            rows, cols = 2, 2
            figsize = (16, 12)
        else:
            rows, cols = 2, 3
            figsize = (24, 12)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f'📊 {category}', fontsize=16, fontweight='bold', y=0.98)
        
        # Plot each graph
        for idx, filename in enumerate(existing_files, 1):
            filepath = results_dir / filename
            
            try:
                img = mpimg.imread(filepath)
                ax = plt.subplot(rows, cols, idx)
                ax.imshow(img)
                ax.axis('off')
                
                # Add title (filename without extension)
                title = filename.replace('.png', '').replace('_', ' ').title()
                ax.set_title(title, fontsize=10, pad=5)
                
                print(f"✅ Loaded: {filename}")
                
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
        
        plt.tight_layout()
        plt.show()
        print(f"\n{'='*60}\n")
    
    print("✅ All available graphs displayed!")


def create_single_pdf():
    """Create a single PDF with all graphs"""
    from matplotlib.backends.backend_pdf import PdfPages
    
    results_dir = Path('../results')
    pdf_path = results_dir / 'all_results_summary.pdf'
    
    # Get all PNG files
    png_files = sorted(results_dir.glob('*.png'))
    
    if not png_files:
        print("❌ No PNG files found in results directory")
        return
    
    print(f"📄 Creating PDF with {len(png_files)} graphs...")
    
    with PdfPages(pdf_path) as pdf:
        for filepath in png_files:
            try:
                img = mpimg.imread(filepath)
                
                # Create figure
                fig = plt.figure(figsize=(11, 8.5))
                ax = plt.subplot(111)
                ax.imshow(img)
                ax.axis('off')
                
                # Add title
                title = filepath.stem.replace('_', ' ').title()
                plt.title(title, fontsize=14, fontweight='bold', pad=10)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                print(f"  ✅ Added: {filepath.name}")
                
            except Exception as e:
                print(f"  ❌ Error adding {filepath.name}: {e}")
    
    print(f"\n✅ PDF created: {pdf_path}")
    print(f"   Total graphs: {len(png_files)}")


def print_results_summary():
    """Print a summary of all available results"""
    results_dir = Path('../results')
    
    print("=" * 70)
    print("📊 AVAILABLE RESULTS SUMMARY")
    print("=" * 70)
    
    # Count files by type
    png_files = list(results_dir.glob('*.png'))
    json_files = list(results_dir.glob('*.json'))
    csv_files = list(results_dir.glob('*.csv'))
    npy_files = list(results_dir.glob('*.npy'))
    
    print(f"\n📈 Graphs (PNG):     {len(png_files)} files")
    print(f"📋 Reports (JSON):   {len(json_files)} files")
    print(f"📊 Data (CSV):       {len(csv_files)} files")
    print(f"💾 Arrays (NPY):     {len(npy_files)} files")
    
    print("\n" + "=" * 70)
    print("🖼️  AVAILABLE GRAPHS:")
    print("=" * 70)
    
    for idx, file in enumerate(sorted(png_files), 1):
        print(f"{idx:2}. {file.name}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("🎨 UAV IDS - COMPREHENSIVE RESULTS VIEWER")
    print("="*70 + "\n")
    
    # Print summary first
    print_results_summary()
    
    print("\n" + "="*70)
    print("OPTIONS:")
    print("="*70)
    print("1. Display all graphs in separate windows (organized by category)")
    print("2. Create a single PDF with all graphs")
    print("3. Both (display + create PDF)")
    print("="*70 + "\n")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Enter choice (1/2/3) [default: 1]: ").strip() or "1"
    
    print()
    
    if choice == "1":
        display_all_graphs()
    elif choice == "2":
        create_single_pdf()
    elif choice == "3":
        display_all_graphs()
        print("\n" + "="*70 + "\n")
        create_single_pdf()
    else:
        print("❌ Invalid choice. Please run again and select 1, 2, or 3.")
