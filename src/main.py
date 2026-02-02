"""
Main Execution Script for UAV IDS

This script runs the complete pipeline:
1. Generate datasets
2. Train the model
3. Evaluate on cross-datasets

Usage:
    python main.py --all              # Run complete pipeline
    python main.py --generate         # Only generate datasets
    python main.py --train            # Only train model
    python main.py --evaluate         # Only evaluate model
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from generate_datasets import main as generate_main
from train import main as train_main
from evaluate import main as evaluate_main


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_all():
    """Run complete pipeline"""
    print_header("UAV IDS - COMPLETE PIPELINE")
    
    print("Phase 1/3: Dataset Generation")
    print("-"*70)
    generate_main()
    
    print("\n\n")
    print_header("Phase 2/3: Model Training")
    train_main()
    
    print("\n\n")
    print_header("Phase 3/3: Cross-Dataset Evaluation")
    evaluate_main()
    
    print_header("PIPELINE COMPLETE!")
    print("All results saved to:")
    print("  - Models: models/")
    print("  - Results: results/")
    print("  - Datasets: data/")


def main():
    parser = argparse.ArgumentParser(
        description='UAV IDS Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all              Run complete pipeline
  python main.py --generate         Generate datasets only
  python main.py --train            Train model only
  python main.py --evaluate         Evaluate model only
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline (generate, train, evaluate)')
    parser.add_argument('--generate', action='store_true',
                       help='Generate datasets only')
    parser.add_argument('--train', action='store_true',
                       help='Train model only')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model only')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Execute based on arguments
    if args.all:
        run_all()
    else:
        if args.generate:
            print_header("Dataset Generation")
            generate_main()
        
        if args.train:
            print_header("Model Training")
            train_main()
        
        if args.evaluate:
            print_header("Model Evaluation")
            evaluate_main()


if __name__ == "__main__":
    main()
