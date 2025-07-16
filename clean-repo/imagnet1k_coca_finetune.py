import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import numpy as np
import open_clip
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import wandb
from pathlib import Path
import os
from torch.optim.lr_scheduler import CosineAnnealingLR 
from torchvision import transforms                      
from transformers.optimization import Adafactor            # NEW
from torch_ema import ExponentialMovingAverage 

# Configuration
MODEL_NAME = "coca_ViT-L-14" #"coca_ViT-B-32"
PRETRAINED = "laion2B-s13B-b90k" 
BATCH_SIZE = 512
NUM_CLASSES = 1000  # ImageNet-1k has 1000 classes
MAX_EPOCHS = 90
PATIENCE = 5
LEARNING_RATE = 5e-4 
GRAD_ACCUM_STEPS = 1
SEED = 42
VAL_SPLIT = 0.1  # 10% for validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = "models/imagenet1k_simple"

# Create model directory
Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Initialize WandB
wandb.init(project="imagenet1k-classification", 
           name=f"imagenet1k_{MODEL_NAME}",
           config={
               "batch_size": BATCH_SIZE,
               "learning_rate": LEARNING_RATE,
               "architecture": MODEL_NAME,
               "pretrained": PRETRAINED,
               "num_classes": NUM_CLASSES
           })

def prepare_datasets(train_tf, val_tf):
    """Prepare train/val/test datasets"""
    print("Loading datasets...")
    train_dataset = ImageFolder(root='imagenet/train', transform=train_tf)

    val_dataset_full = ImageFolder(root='imagenet/train', transform=val_tf)

    test_dataset = ImageFolder(root='imagenet/val', transform=val_tf)
    
    # Split train into train/val
    train_idx, val_idx = train_test_split(
        list(range(len(train_dataset))),
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=train_dataset.targets
    )
    
    return {
        'train': torch.utils.data.Subset(train_dataset, train_idx),
        'val': torch.utils.data.Subset(val_dataset_full, val_idx),
        'test': test_dataset
    }

class AttnPoolHead(nn.Module):
    """
    Single-query multi-head attention pooler + linear classifier.
    The query token is learned from scratch; the vision encoder stays frozen.
    """
    def __init__(self, dim, n_classes, n_heads: int = 8):
        super().__init__()
        self.q     = nn.Parameter(torch.randn(1, 1, dim))          # (1,1,D)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.proj  = nn.Linear(dim, n_classes)

    def forward(self, x):                                           # x: (B,N,D)
        q = self.q.expand(x.size(0), -1, -1)                        # (B,1,D)
        pooled, _ = self.attn(q, x, x)                              # (B,1,D)
        return self.proj(pooled.squeeze(1))   


# Load model
print("Loading model...")
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
     model_name=MODEL_NAME,
     pretrained=PRETRAINED
 )

train_transform = transforms.Compose(
    [transforms.RandAugment(2, 20), *preprocess_train.transforms])

val_transform = preprocess_val

model = model.to(DEVICE)

# Modify for classification
vision_encoder = model.visual
vision_encoder.head = AttnPoolHead(
    dim=vision_encoder.output_dim,
    n_classes=NUM_CLASSES
).to(DEVICE)

# Freeze all except head
for name, param in vision_encoder.named_parameters():
    if "head" not in name:
        param.requires_grad = False

# Prepare datasets
print("Preparing datasets...")
datasets = prepare_datasets(train_transform, val_transform)

print(f"Train samples: {len(datasets['train'])}")
print(f"Val samples: {len(datasets['val'])}")
print(f"Test samples: {len(datasets['test'])}")

# DataLoaders
train_loader = DataLoader(
    datasets['train'], 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=20,  # Good for 31-core CPU
    pin_memory_device="cuda",  # Fixed missing comma
    prefetch_factor=6,  # Optimal for 24 workers
    persistent_workers=True,  # Reduces overhead
    drop_last=True,  # Avoids partial batches
    pin_memory=True,  # Required for pin_memory_device
)

val_loader = DataLoader(
    datasets['val'], 
    batch_size=BATCH_SIZE, 
    num_workers=12,  # Good for validation
    prefetch_factor=6,
    persistent_workers=True,
    pin_memory_device="cuda",
    pin_memory=True,  # Don't forget this
)

test_loader = DataLoader(
    datasets['test'], 
    batch_size=BATCH_SIZE, 
    num_workers=12,  
    prefetch_factor=6,
    persistent_workers=True,
    pin_memory_device="cuda",
    pin_memory=True,  # Critical for speed
)

# Training setup
criterion = nn.CrossEntropyLoss(label_smoothing=0.2) 
optimizer = Adafactor(                                # use Adafactor
    vision_encoder.head.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01,
    relative_step=False
)
ema = ExponentialMovingAverage(                       # EMA 0.9999
    vision_encoder.head.parameters(), decay=0.9999
)
scaler = GradScaler()
scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)


def run_epoch(model, loader, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    batch_count = 0  # Track batch count for gradient accumulation
    
    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, desc="Training" if is_train else "Validating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with autocast():
                with torch.no_grad():     
                    features = model.visual(images)
                    if isinstance(features, tuple):
                        features = features[0]
                    if features.dim() == 2:                         # (B,D) → (B,1,D)
                        features = features.unsqueeze(1)
                logits = vision_encoder.head(features)
                loss = criterion(logits, labels) / (GRAD_ACCUM_STEPS if is_train else 1)
            
            if is_train:
                scaler.scale(loss).backward()
                batch_count += 1
                if batch_count % GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(                   # clip 1.0
                        vision_encoder.head.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    ema.update()  
            
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss/len(loader), accuracy_score(all_labels, all_preds)

# Training loop
best_val_acc = 0
epochs_no_improve = 0

for epoch in range(MAX_EPOCHS):
    train_loss, train_acc = run_epoch(model, train_loader)
    with ema.average_parameters():
        val_loss, val_acc = run_epoch(model, val_loader, False)
    scheduler.step()
    
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
#best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
#model.load_state_dict(torch.load(best_model_path))
with ema.average_parameters():                         
    test_loss, test_acc = run_epoch(model, test_loader, False)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
wandb.log({"final_test_acc": test_acc})

# Save final model
final_model_path = os.path.join(MODEL_SAVE_DIR, "final_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Saved final model to {final_model_path}")

wandb.finish()