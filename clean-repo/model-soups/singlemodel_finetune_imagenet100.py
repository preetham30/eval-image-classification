import argparse
import os
import torch
import open_clip
import numpy as np
from tqdm import tqdm
import time
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from timm.data.transforms_factory import transforms_imagenet_train

from utils import ModelWrapper, maybe_dictionarize_batch, cosine_lr
from zeroshot import zeroshot_classifier
from openai_imagenet_template import openai_imagenet_template
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# Constants
TEST_SIZE = 0.1
SEED = 42

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


def prepare_datasets(preprocess_fn, train_preprocess_fn=None):
    """Prepare ImageNet100 datasets with train/val/test splits"""
    ds = load_dataset("clane9/imagenet-100")
    
    # Split train into new_train (90%) and val (10%) - stratified
    train_labels = np.array(ds['train']['label'])
    train_idx, val_idx = train_test_split(
        np.arange(len(ds['train'])),
        test_size=TEST_SIZE,  # 10% validation
        random_state=SEED,
        stratify=train_labels
    )
    
    # Use different preprocessing for train if provided
    if train_preprocess_fn is None:
        train_preprocess_fn = preprocess_fn
    
    datasets = {
        'train': ImageNet100Dataset(ds['train'].select(train_idx), train_preprocess_fn),
        'val': ImageNet100Dataset(ds['train'].select(val_idx), preprocess_fn),
        'test': ImageNet100Dataset(ds['validation'], preprocess_fn)  # Original validation becomes test
    }
    
    return datasets


def get_imagenet100_classnames():
    """Get the class names for ImageNet100"""
    # Load a small sample to get class names
    ds = load_dataset("clane9/imagenet-100", split="train[:1]")
    
    # Get the class names from the dataset features
    class_names = ds.features['label'].names
    return class_names


def create_data_loaders(datasets, batch_size, num_workers):
    """Create data loaders for train, val, and test sets"""
    loaders = {}
    
    # Training loader with shuffle
    loaders['train'] = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation and test loaders without shuffle
    for split in ['val', 'test']:
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders


def evaluate_model(model, data_loader, loss_fn, device):
    """Evaluate model on given data loader"""
    model.eval()
    correct, count = 0.0, 0.0
    total_loss = 0.0

    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating")
        for batch in pbar:
            if isinstance(batch, dict):
                inputs, labels = batch['images'].to(device), batch['labels'].to(device)
            else:
                inputs, labels = batch[0].to(device), batch[1].to(device)

            logits = model(inputs)
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            count += len(logits)

            all_predictions.extend(pred.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_description(
                f"Val loss: {loss.item():.4f}   Acc: {100*correct/count:.2f}%")
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / count

    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )

    return avg_loss, accuracy, precision, recall, f1


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('~/ssd/checkpoints/soups_2'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--custom-template", action="store_true", default=False,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--warmup-length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--model",
        default='ViT-B-32',
        help='Model to use -- you can try another like ViT-L/14'
    )
    parser.add_argument(
        "--name",
        default='finetune_imagenet100',
        help='Filename for the checkpoints.'
    )
    parser.add_argument(
        "--timm-aug", action="store_true", default=False,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    DEVICE = 'cuda'

    # Set up templates
    if args.custom_template:
        #template = [lambda x : f"a photo of a {x}."]
        template = [
                 lambda c: f"a photo of a {c}.",
                 lambda c: f"a bad photo of a {c}.",
                 lambda c: f"a cropped photo of a {c}.",
                 lambda c: f"a dark photo of a {c}.",
                 lambda c: f"a bright photo of a {c}.",
                 lambda c: f"a low-res photo of a {c}.",
                 lambda c: f"a high-res photo of a {c}.",
                 lambda c: f"a weird photo of a {c}.",
                 lambda c: f"a close-up photo of a {c}.",
                 lambda c: f"a blurry photo of a {c}."
            ]
    else:
        template = openai_imagenet_template

    # Load base model
    base_model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, 
        pretrained='laion2b_s34b_b79k',
        device='cuda'
    )

    tokenizer = open_clip.get_tokenizer(args.model) 
    
    # Set up data augmentation
    if args.timm_aug:
        from timm.data import create_transform
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        
        # Get input size
        img_size = 224  # Default for ViT-B-32
        if hasattr(base_model.visual, 'image_size'):
            img_size = base_model.visual.image_size
        elif isinstance(getattr(base_model.visual, 'patch_size', None), int):
            img_size = base_model.visual.patch_size * base_model.visual.num_patches_per_side
        
        # Strong augmentation policy
        train_preprocess = create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
    
        # Adjust for CLIP's normalization
        train_preprocess.transforms[-1].mean = (0.48145466, 0.4578275, 0.40821073)
        train_preprocess.transforms[-1].std = (0.26862954, 0.26130258, 0.27577711)

    else:
        train_preprocess = preprocess
    
    # Prepare datasets
    print("Loading ImageNet100 datasets...")
    datasets = prepare_datasets(preprocess, train_preprocess)
    data_loaders = create_data_loaders(datasets, args.batch_size, args.workers)
    
    # Get class names
    classnames = get_imagenet100_classnames()
    print(f"Number of classes: {len(classnames)}")
    
    # Create zero-shot classifier
    clf = zeroshot_classifier(base_model, tokenizer, classnames, template, DEVICE)
    NUM_CLASSES = len(classnames)
    feature_dim = base_model.visual.output_dim

    # Create model wrapper
    model = ModelWrapper(base_model, feature_dim, NUM_CLASSES, normalize=True, initial_weights=clf)
    for p in model.parameters():
        p.data = p.data.float()

    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model, device_ids=devices)

    # Set up optimizer and scheduler
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.wd)

    num_batches = len(data_loaders['train'])
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Save initial model
    model_path = os.path.join(args.model_location, f'{args.name}_0.pt')
    print('Saving initial model to', model_path)
    os.makedirs(args.model_location, exist_ok=True)
    torch.save(model.module.state_dict(), model_path)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print('='*60)
        
        # Train
        model.train()
        end = time.time()
        epoch_loss = 0.0
        
        for i, batch in enumerate(data_loaders['train']):
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()
            
            # Handle both dict and tuple batch formats
            if isinstance(batch, dict):
                inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)
            else:
                inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            
            data_time = time.time() - end

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_time = time.time() - end
            end = time.time()

            if i % 20 == 0:
                percent_complete = 100.0 * i / len(data_loaders['train'])
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(data_loaders['train'])}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        avg_train_loss = epoch_loss / len(data_loaders['train'])
        print(f"Average train loss: {avg_train_loss:.6f}")

        # Evaluate on validation set
        print('*'*80)
        print('Starting validation evaluation')
        val_loss, val_acc, _, _, _ = evaluate_model(model, data_loaders['val'], loss_fn, DEVICE)
        print(f'Val loss at epoch {epoch}: {val_loss:.6f}')
        print(f'Val acc at epoch {epoch}: {100*val_acc:.2f}%')

        # Save model checkpoint
        model_path = os.path.join(args.model_location, f'{args.name}_{epoch + 1}.pt')
        print('Saving model to', model_path)
        torch.save(model.module.state_dict(), model_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.model_location, f'{args.name}_best.pt')
            print(f'New best validation accuracy: {100*val_acc:.2f}%. Saving best model to {best_model_path}')
            torch.save(model.module.state_dict(), best_model_path)

    # Final evaluation on test set
    print('\n' + '='*80)
    print('Final evaluation on test set')
    print('='*80)
    
    # Load best model for test evaluation
    best_model_path = os.path.join(args.model_location, f'{args.name}_best.pt')
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.module.load_state_dict(torch.load(best_model_path))
    
    test_loss, test_acc, precision, recall, f1 = evaluate_model(model, data_loaders['test'], loss_fn, DEVICE)
    print(f'Final test loss: {test_loss:.6f}')
    print(f'Final test acc: {test_acc:.4f}%')
    print(f'Final test precision: {precision:.4f}%')
    print(f'Final test recall: {recall:.4f}%')
    print(f'Final test f1: {test_acc:.4f}%')
    print(f'Best validation acc: {100*best_val_acc:.2f}%')