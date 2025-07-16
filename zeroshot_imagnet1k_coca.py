import os
import open_clip
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# Config
MODEL_NAME = "coca_ViT-B-32" #"coca_ViT-B-32"
PRETRAINED = "laion2B-s13B-b90k"
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name=MODEL_NAME,
    pretrained=PRETRAINED
)
model = model.to(DEVICE)
model.eval()

# Dataset - this will automatically assign class indices based on folder names
dataset = ImageFolder(root='imagenet/val_zeroshot', transform=preprocess)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

# Get the actual class names in the correct order (alphabetical)
actual_class_names = dataset.classes
print(f"Found {len(actual_class_names)} classes in dataset")
print(f"First few classes: {actual_class_names[:5]}")

# Simplified templates (better for CoCa)
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

# Generate text features in the same order as the dataset classes
text_features = []
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    for class_name in tqdm(actual_class_names, desc="Processing classes"):
        class_name = class_name.replace("_", " ")
        texts = [template.format(class_name) for template in imagenet_templates]
        text_inputs = tokenizer(texts).to(DEVICE)
        class_features = model.encode_text(text_inputs)
        class_features /= class_features.norm(dim=-1, keepdim=True)
        text_features.append(class_features.mean(dim=0))
    
    text_features = torch.stack(text_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Verify alignment
print(f"Text features shape: {text_features.shape}")  # Should be (num_classes, feature_dim)
print(f"Number of classes: {len(dataset.classes)}")

# Evaluation
correct_1, correct_5, total = 0, 0, 0

for images, targets in tqdm(dataloader, desc="Evaluating"):
    images = images.to(DEVICE)
    targets = targets.to(DEVICE)
    
    # Debug: print some targets and their corresponding class names
    if total == 0:  # Only print for first batch
        unique_targets = torch.unique(targets)
        print("\nSample target indices and their class names:")
        for idx in unique_targets[:5]:  # Print first 5 unique targets
            print(f"Index {idx.item()}: {actual_class_names[idx.item()]}")
    
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity (cosine between -1 and 1)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get predictions
        _, preds = similarity.topk(5, dim=-1)
        correct = preds.eq(targets.view(-1, 1))
        
        correct_1 += correct[:, 0].sum().item()
        correct_5 += correct.any(dim=1).sum().item()
        total += targets.size(0)

print(f"\nFinal Results:")
print(f"Total samples evaluated: {total}")
print(f"Top-1 Accuracy: {100 * correct_1 / total:.2f}%")
print(f"Top-5 Accuracy: {100 * correct_5 / total:.2f}%")