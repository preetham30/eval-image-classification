import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import open_clip
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import wandb
from PIL import Image
from pathlib import Path
import os

# Configuration
MODEL_NAME =  "coca_ViT-L-14"#"coca_ViT-B-32"
PRETRAINED = "laion2B-s13B-b90k" 
BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 5
LEARNING_RATE =  5e-5 
GRAD_ACCUM_STEPS = 2
SEED = 42
TEST_SIZE = 0.8 # 20% test
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = "models/trial_4"

# Create model directory
Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Initialize WandB
wandb.init(project="evals-reasoning-advex-experiement", 
           name=f"imagenet_100_vitl_14_20train_80val",
           config={
               "batch_size": BATCH_SIZE,
               "learning_rate": LEARNING_RATE,
               "architecture": MODEL_NAME
           })

# Dataset Class
class ImageNet100Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        if self.transform:
            image = self.transform(image.convert("RGB"))
            
        return image, torch.tensor(label, dtype=torch.long)


def prepare_datasets(preprocess_fn):
    ds = load_dataset("clane9/imagenet-100")
    
    # Split train into new_train (90%) and val (10%) - stratified
    train_labels = np.array(ds['train']['label'])
    train_idx, val_idx = train_test_split(
        np.arange(len(ds['train'])),
        test_size=TEST_SIZE,  # 10% validation
        random_state=SEED,
        stratify=train_labels
    )
    
    return {
        'train': ImageNet100Dataset(ds['train'].select(train_idx), preprocess_fn),
        'val': ImageNet100Dataset(ds['train'].select(val_idx), preprocess_fn),
        'test': ImageNet100Dataset(ds['validation'], preprocess_fn)  # Original validation becomes test
    }

# Model Setup
print("Loading model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name=MODEL_NAME,
    pretrained=PRETRAINED
)
model = model.to(DEVICE)

# Modify for classification
vision_encoder = model.visual
vision_encoder.head = nn.Linear(vision_encoder.output_dim, 100).to(DEVICE)

# Freeze all except head
for name, param in vision_encoder.named_parameters():
    if "head" not in name:
        param.requires_grad = False

# Prepare datasets
print("Preparing datasets...")
datasets = prepare_datasets(preprocess)

print(f"Train samples: {len(datasets['train'])}")
print(f"Val samples: {len(datasets['val'])}")
print(f"Test samples: {len(datasets['test'])}")

# DataLoaders
train_loader = DataLoader(datasets['train'], 
                         batch_size=BATCH_SIZE, 
                         shuffle=True, 
                         num_workers=4,
                         prefetch_factor=2)

val_loader = DataLoader(datasets['val'], 
                        batch_size=BATCH_SIZE, 
                        num_workers=2)

test_loader = DataLoader(datasets['test'], 
                         batch_size=BATCH_SIZE, 
                         num_workers=2)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(vision_encoder.head.parameters(), lr=LEARNING_RATE, 
                weight_decay=0.1,  # Added strong regularization
                betas=(0.9, 0.98)  # More stable momentum
                )
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

# Training function
def run_epoch(model, loader, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.set_grad_enabled(is_train):
        for i, (images, labels) in enumerate(tqdm(loader, desc="Training" if is_train else "Validating")):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with autocast():
                features = model.visual(images)
                if isinstance(features, tuple):
                    features = features[0]
                logits = vision_encoder.head(features)
                loss = criterion(logits, labels) / (GRAD_ACCUM_STEPS if is_train else 1)
            
            if is_train:
                scaler.scale(loss).backward()
                if (i + 1) % GRAD_ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss/len(loader), accuracy_score(all_labels, all_preds)

# Main training loop
best_val_acc = 0
epochs_no_improve = 0

for epoch in range(MAX_EPOCHS):
    train_loss, train_acc = run_epoch(model, train_loader)
    val_loss, val_acc = run_epoch(model, val_loader, False)
    scheduler.step(val_acc)
    
    # Logging
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "lr": optimizer.param_groups[0]['lr']
    })
    
    print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
    
    # Early stopping and model saving
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved best model to {model_path}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve == PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, checkpoint_path)

# Final evaluation
best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
model.load_state_dict(torch.load(best_model_path))
test_loss, test_acc = run_epoch(model, test_loader, False)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
wandb.log({"final_test_acc": test_acc})

# Save final model
final_model_path = os.path.join(MODEL_SAVE_DIR, "final_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Saved final model to {final_model_path}")

wandb.finish()