import os

from datasets import Dataset, load_dataset


def download_dataset(
    dataset_name: str, data_file_path: str, overwrite: bool = False
) -> Dataset:
    """Download a dataset from the Hugging Face Hub."""
    if not os.path.exists(data_file_path):
        os.makedirs(data_file_path)
    return load_dataset(dataset_name, data_files=data_file_path)
