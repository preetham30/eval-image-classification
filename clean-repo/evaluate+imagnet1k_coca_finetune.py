import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import open_clip
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import matplotlib.pyplot as plt

# Configuration
MODEL_NAME = "coca_ViT-L-14" #"coca_ViT-B-32"
PRETRAINED = "laion2B-s13B-b90k" 
BATCH_SIZE = 256
NUM_CLASSES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = "models/imagenet1k_simple"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")

# Load model
print("Loading model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name=MODEL_NAME,
    pretrained=PRETRAINED
)
model = model.to(DEVICE)

# Modify for classification
vision_encoder = model.visual
vision_encoder.head = nn.Linear(vision_encoder.output_dim, NUM_CLASSES).to(DEVICE)

# Load trained weights
if os.path.exists(MODEL_PATH):
    print(f"Loading model weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    print(f"No saved model found at {MODEL_PATH}, using pretrained weights only")
model.eval()

def prepare_datasets(preprocess_fn):
    print("Loading datasets...")
    train_dataset = ImageFolder(root='imagenet/train', transform=preprocess_fn)
    test_dataset = ImageFolder(root='imagenet/val', transform=preprocess_fn)
    
    # Get all available class names from training set
    train_classes = set(train_dataset.classes)
    
    # Filter test set to only include classes present in training set
    filtered_samples = []
    for path, label in test_dataset.samples:
        class_name = test_dataset.classes[label]
        if class_name in train_classes:
            # Map to train dataset's label index
            new_label = train_dataset.class_to_idx[class_name]
            filtered_samples.append((path, new_label))
    
    # Create new test dataset with filtered samples
    test_dataset.samples = filtered_samples
    test_dataset.targets = [s[1] for s in filtered_samples]
    
    # Create validation set from 5% of training data
    np.random.seed(42)
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    split = int(0.05 * len(train_dataset))
    
    return {
        'val': Subset(train_dataset, indices[:split]),
        'test': test_dataset
    }

# Prepare datasets
print("Preparing datasets...")
datasets = prepare_datasets(preprocess)

# DataLoaders
val_loader = DataLoader(datasets['val'], batch_size=BATCH_SIZE,
                       num_workers=4, pin_memory=True)
test_loader = DataLoader(datasets['test'], batch_size=BATCH_SIZE,
                        num_workers=4, pin_memory=True)

def evaluate_model(model, loader, name="test"):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating {name}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            features = model.visual(images)
            if isinstance(features, tuple):
                features = features[0]
            logits = vision_encoder.head(features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n{name.upper()} Evaluation:")
    print(f"Loss: {total_loss/len(loader):.4f} | Accuracy: {accuracy:.4f}")
    print(f"First 10 predictions vs labels:", list(zip(all_preds[:10], all_labels[:10])))
    
    return accuracy

print("\nRunning evaluation...")
#val_acc = evaluate_model(model, val_loader, "val")
test_acc = evaluate_model(model, test_loader, "test")

print("\nEvaluation complete!")
#print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")