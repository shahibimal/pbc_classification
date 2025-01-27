import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from src.models.factory import ModelFactory
from early_stopping import EarlyStopping
from utils import calculate_sensitivity_specivity, plot_metrics

class LossFactory:
    """Helper class to create loss function dynamically."""
    def create_loss(self, name):
        if name == "CrossEntropyLoss":
            return CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{name} is not implemented.")

# Classification Model
class ClassificationModel:
    def __init__(self, model_name, num_classes, loss_fn, lr=1e-4, use_scheduler=True, device='cuda'):
        self.device = device
        self.model = get_model_factory(model_name, num_classes).create_model().to(self.device)
        self.loss_fn = LossFactory().create_loss(loss_fn)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        ) if use_scheduler else None

    def train(self, train_loader, val_loader, epochs, early_stopping_patience=5, early_stopping_delta=0.001):
        early_stopping = EarlyStopping(patience=early_stopping_patience, delta=early_stopping_delta, verbose=True)

        train_loss_history = []
        train_acc_history = []
        val_acc_history = []
        val_metrics_history = []

        print(f"Training started with learning rate = {self.optimizer.param_groups[0]['lr']}")

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs} started.")
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Training loop with tqdm progress bar
            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs} - Training") as pbar:
                for i, (inputs, labels) in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)

                    if torch.isnan(loss).any():
                        print("NaN loss detected!")
                        return

                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # Update tqdm progress bar
                    pbar.set_postfix(loss=running_loss / total, accuracy=correct / total)

            # Calculate training metrics
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)

            # Validation phase
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            val_acc_history.append(val_acc)
            val_metrics_history.append(val_metrics)

            print(f'Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}')
            print(f'Validation Metrics: Precision: {val_metrics[0]:.4f}, Recall: {val_metrics[1]:.4f}, F1: {val_metrics[2]:.4f}, Sensitivity: {val_metrics[3]:.4f}, Specificity: {val_metrics[4]:.4f}')

            # Early stopping check
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
                break

            # Step the scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)

        # Plot training and validation metrics
        plot_metrics(train_loss_history, train_acc_history, val_acc_history)  
        return val_acc

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []

        with tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation") as pbar_val:
            with torch.no_grad():
                for i, (inputs, labels) in pbar_val:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    # Store predictions and labels for metrics
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())

                    # Update tqdm progress bar
                    pbar_val.set_postfix(val_loss=val_loss / val_total, val_accuracy=val_correct / val_total)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Compute additional metrics for validation
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        sensitivity, specificity = calculate_sensitivity_specificity(all_labels, all_preds, len(set(all_labels)))

        return val_loss, val_acc, (precision, recall, f1, sensitivity, specificity)

    def test(self, test_loader):
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        test_acc = accuracy_score(y_true, y_pred)
        test_precision = precision_score(y_true, y_pred, average="weighted")
        test_recall = recall_score(y_true, y_pred, average="weighted")

        return test_acc, test_precision, test_recall
