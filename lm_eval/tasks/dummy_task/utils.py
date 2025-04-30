import datasets
import pandas as pd
from datasets import Dataset


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process the palindrome CSV dataset."""
    # If you're working with a raw CSV instead of a dataset
    # You could use this approach instead of the dataset.map method
    if isinstance(dataset, dict):
        # Convert raw data to a proper dataset
        df = pd.DataFrame(dataset)
        processed_dataset = Dataset.from_pandas(df)
        return processed_dataset

    # If already working with a dataset, just ensure it has the right format
    return dataset


def doc_to_text(doc):
    """Convert document to input text format."""
    return f"Question: Is the following string a palindrome: '{doc['question']}'?\nA palindrome is a string that reads the same forward and backward.\nAnswer:"


def doc_to_target(doc):
    """Convert document to target output format."""
    # Convert 0/1 to readable answer
    return "The answer is " + ("Yes" if doc["answer"] == 1 else "No")
