from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split

from utils import print_label_distribution

# Define the class names 
class_names = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']


# Transformation pipeline
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Define FilteredDataset class
class FilteredDataset(Dataset):
    def __init__(self, dataset, class_names):
        self.dataset = dataset
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        self.filtered_indices = []
        self.targets = []
        
        print("Dataset classes before filtering:", dataset.classes)  # Add this line for debugging

        # Filter and remap labels
        for idx, (_, label) in enumerate(dataset.samples):
            class_name = dataset.classes[label]
            if class_name in class_names:
                self.filtered_indices.append(idx)
                self.targets.append(self.class_to_idx[class_name])

        print(f"Number of samples after filtering: {len(self.filtered_indices)}")  # Add this line for debugging

    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        image, _ = self.dataset[original_idx]
        label = self.targets[idx]
        return image, label

# Create the dataset
dataset_dir = '../data/PBC_dataset_normal_DIB_224/PBC_dataset_normal_DIB_224'
full_dataset = datasets.ImageFolder(dataset_dir, transform=transform['train'])

# Create filtered dataset
filtered_dataset = FilteredDataset(full_dataset, class_names)

# Perform stratified split
indices = np.arange(len(filtered_dataset))
targets = np.array(filtered_dataset.targets)

# First split: train and temp (val + test)
train_idx, temp_idx = train_test_split(
    indices, 
    test_size=0.2,  # 40% for val+test
    stratify=targets,
    random_state=42
)

# Second split: val and test from temp
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,  # Split temp into equal val and test
    stratify=targets[temp_idx],
    random_state=42
)

# Create datasets for train, validation, and test splits
train_dataset = Subset(filtered_dataset, train_idx)
val_dataset = Subset(filtered_dataset, val_idx)
test_dataset = Subset(filtered_dataset, test_idx)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# Verification
print("Verifying datasets and loaders:")
print(f"\nTotal dataset size: {len(filtered_dataset)}")
print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# Check distribution for each loader using class_names
print("\nLabel distribution across splits:")
print_label_distribution(train_loader, class_names, "Train")
print_label_distribution(val_loader, class_names, "Validation")
print_label_distribution(test_loader, class_names, "Test")
