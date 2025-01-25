import torch
from torch.utils.data import DataLoader
import numpy as np

def print_label_distribution(loader:DataLoader, class_names: list, name: str):
    """
    Print the label distribution in the given dataloader.

    Args:
        loader(DataLoader): The DataLoader whose distribution is supposed to be calculated.
        slotclass (list): List of class names corresponding to the labels.
        name (str): The name of the dataset (train, test, or validation).
    """
    label_counts = {}
    # Iterate through the loader and count occurrences of each label
    for _, labels in loader:
        for label in labels:
            label = label.item()  # Convert tensor to integer
            label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\n{name} loader label distribution:")
    for label in sorted(label_counts.keys()):
        print(f"Class {class_names[label]}: {label_counts[label]} samples")
