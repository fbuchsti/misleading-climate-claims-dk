"""
Utility functions for saving results and handling file operations.
"""
import os
import pandas as pd


def ensure_directory(path: str):
    """
    Creates directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def save_incremental(results: list, output_path: str):
    """
    Saves current annotation results to disk.
    Overwrites file each time (safe because we accumulate results).
    """
    df = pd.DataFrame(results)
    df.to_parquet(output_path, index=False)