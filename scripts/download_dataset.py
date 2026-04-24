"""
download_dataset.py - Standalone script to download the IMDB dataset

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output ./data
"""

import os
import argparse
import pandas as pd
import kagglehub
import shutil


def download_dataset(output_dir: str = "imdb_dataset"):
    """
    Download the IMDB dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
    
    Returns:
        Path to the downloaded CSV file
    """
    print("=" * 60)
    print("Downloading IMDB Dataset")
    print("=" * 60)
    
    # Download from Kaggle
    print("Downloading from Kaggle...")
    dataset_path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    print(f"Dataset downloaded to: {dataset_path}")
    
    # Find CSV file
    csv_file = None
    for f in os.listdir(dataset_path):
        if f.endswith(".csv"):
            csv_file = os.path.join(dataset_path, f)
            break
    
    if csv_file is None:
        raise FileNotFoundError("No CSV file found in downloaded dataset")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy to output directory
    output_csv = os.path.join(output_dir, "IMDB_Dataset.csv")
    shutil.copy(csv_file, output_csv)
    print(f"Dataset saved to: {output_csv}")
    
    # Load and show info
    df = pd.read_csv(output_csv)
    print(f"\nDataset info:")
    print(f"  Rows: {df.shape[0]}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Class distribution:")
    print(f"    Positive: {df['sentiment'].value_counts()['positive']}")
    print(f"    Negative: {df['sentiment'].value_counts()['negative']}")
    
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IMDB dataset")
    parser.add_argument("--output", "-o", default="imdb_dataset", help="Output directory")
    args = parser.parse_args()
    
    download_dataset(args.output)