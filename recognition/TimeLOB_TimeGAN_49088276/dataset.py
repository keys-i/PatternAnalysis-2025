"""
A module to tell to how we handle the dataset.

Created By:
ID: s49088276

References:
- 
"""

import os

def test_dataset_exists():
    data_dir = "data"
    files = os.listdir(data_dir)
    print(f"Files in '{data_dir}': {files}")

if __name__ == "__main__":
    test_dataset_exists()