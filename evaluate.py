import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.factory import ModelFactory
from src.utils import calculate_sensitivity_specivity
from src.dataset import create_dataloaders  


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a classification model.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory path.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    return parser.parse_args()

# get the device
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# Define evalator class
class Evaluator:
    def __init__(self, model_name, num_classes, checkpoint_path):
        self.device = get_device()
        self.model = ModelFactory(model_name, num_classes)().to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint {checkpoint_path}, trained for {checkpoint['epoch']} epochs.")

    def evaluate(self, test_loader):
        self.model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        test_loss /= test_total
        test_acc = test_correct / test_total
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        sensitivity, specificity = calculate_sensitivity_specivity(all_labels, all_preds, len(set(all_labels)))

        print(f"Test Results | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f}")


if __name__ == "__main__":
    args = parse_args()
    _, _, test_loader, _, _, _ = create_dataloaders(dataset_dir=args.dataset_dir, batch_size=args.batch_size)
    evaluator = Evaluator(args.model_name, num_classes=8, checkpoint_path=args.checkpoint)
    evaluator.evaluate(test_loader)
