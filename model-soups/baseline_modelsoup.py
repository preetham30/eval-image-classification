import argparse
import os
import wget
import torch
import time
import open_clip
import json
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from utils import get_model_from_sd, ModelWrapper
from datasets import load_dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset

# Constants for ImageNet100
SEED = 42
TEST_SIZE = 0.1  # 10% validation split

class DictImageFolder(ImageFolder):
    """ImageFolder that returns {'images': img, 'labels': label} instead of (img, label)."""
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return {
            'images': img,
            'labels': torch.tensor(target, dtype=torch.long),
        }

def test_model_on_dataset(model, dataset):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if isinstance(model, torch.nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model
    
    actual_model.to(device)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8
    )
    
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in loader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            
            logits = actual_model(images)
            
            # Safety check for output dimensions
            if logits.shape[1] != 1000:
                print(f"Warning: Unexpected output dimension {logits.shape[1]} (expected 100)")
                continue
                
            pred = logits.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            
        return 100.0 * correct / total  # Return percentage

def prepare_datasets(val_transform):
    test_dataset = DictImageFolder(
        root=os.path.expanduser('~/evals_reasoning/imagenet/val'),
        transform=val_transform,
    )
    return {'val': test_dataset}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('~/ssd/checkpoints/soups'),
        help="Where to download or find the models.",
    )
    parser.add_argument(
        "--download-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--eval-individual-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--greedy-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    return parser.parse_args()

def print_model_performance(results):
    """Print individual model performance in a readable format"""
    print("\nModel Performance Summary:")
    print("-" * 60)
    print(f"{'Model':<15}{'Train Acc':>12}{'Val Acc':>12}{'Test Acc':>12}")
    print("-" * 60)
    print(f"{results['model_name']:<15}"
          #f"{results['train']:>12.2f}"
          f"{results['val']:>12.2f}")
          #f"{results['test']:>12.2f}")
    print("-" * 60)

if __name__ == '__main__':
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Add this line
    NUM_MODELS = 2
    INDIVIDUAL_MODEL_RESULTS_FILE = 'individual_model_results.jsonl'
    GREEDY_SOUP_RESULTS_FILE = 'greedy_soup_results.jsonl'

    # Step 1: Download models (optional)
    if args.download_models:
        if not os.path.exists(args.model_location):
            os.mkdir(args.model_location)
        for i in range(NUM_MODELS):
            print(f'\nDownloading model {i} of {NUM_MODELS - 1}')
            wget.download(
                f'https://github.com/mlfoundations/model-soups/releases/download/v0.0.2/model_{i}.pt',
                out=args.model_location
            )

    model_paths = [os.path.join(args.model_location, f'model_{i}.pt') for i in range(NUM_MODELS)]
    model_name = 'ViT-L-14'
    pretrained = 'laion2b_s32b_b82k'  # or other OpenCLIP pretrained weights

    # Step 2: Evaluate individual models
    if args.eval_individual_models or args.greedy_soup:
        
        base_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device='cpu')
        tokenizer = open_clip.get_tokenizer(model_name)
        datasets = prepare_datasets(preprocess)

    if args.eval_individual_models:
        if os.path.exists(INDIVIDUAL_MODEL_RESULTS_FILE):
            os.remove(INDIVIDUAL_MODEL_RESULTS_FILE)
        for j, model_path in enumerate(model_paths):
            assert os.path.exists(model_path)
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))

            #model = ModelWrapper(base_model, feature_dim=512, num_classes=100)

            model = get_model_from_sd(state_dict, base_model)

            model.load_state_dict(state_dict, strict=False)  # strict=False allows for missing keys
        
            # Verify the classification head wasn't overwritten
            if model.classification_head.out_features == 1000:
                #model.classification_head = torch.nn.Linear(512, 100).to(device)
               print("Reset classification head is at 1000 classes")


            results = {'model_name': f'model_{j}'}
            
            # Evaluate on train, val, and test splits
            for split_name, dataset in datasets.items():
                print(f'\nEvaluating model {j} of {NUM_MODELS - 1} on {split_name} set...')
                accuracy = test_model_on_dataset(model, dataset)
                results[split_name] = accuracy
                print(f"{split_name.capitalize()} Accuracy: {accuracy:.2f}%")

            # Print performance summary for this model
            print_model_performance(results)

            with open(INDIVIDUAL_MODEL_RESULTS_FILE, 'a+') as f:
                f.write(json.dumps(results) + '\n')

    # Step 3: Greedy Soup (only if individual models were evaluated)
    if args.greedy_soup and os.path.exists(INDIVIDUAL_MODEL_RESULTS_FILE):
        if os.path.exists(GREEDY_SOUP_RESULTS_FILE):
            os.remove(GREEDY_SOUP_RESULTS_FILE)

        # Sort models by decreasing accuracy on validation set
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        individual_model_val_accs = {}
        for _, row in individual_model_db.iterrows():
            individual_model_val_accs[row['model_name']] = row['val']
        individual_model_val_accs = sorted(individual_model_val_accs.items(), key=operator.itemgetter(1))
        individual_model_val_accs.reverse()
        sorted_models = [x[0] for x in individual_model_val_accs]
        
        # Start with best performing model
        greedy_soup_ingredients = [sorted_models[0]]
        greedy_soup_params = torch.load(os.path.join(args.model_location, f'{sorted_models[0]}.pt'))
        best_val_acc_so_far = individual_model_val_accs[0][1]
        val_dataset = datasets['val']

        print("\nBuilding Greedy Soup:")
        print(f"Initial ingredient: {sorted_models[0]} with val acc {best_val_acc_so_far:.2f}%")

        # Iterate through models and consider adding to soup
        for i in range(1, NUM_MODELS):
            print(f'\nTesting model {i} ({sorted_models[i]}) for potential addition...')

            # Create potential soup
            new_ingredient_params = torch.load(os.path.join(args.model_location, f'{sorted_models[i]}.pt'))
            num_ingredients = len(greedy_soup_ingredients)
            potential_greedy_soup_params = {
                k: greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
                    new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                for k in new_ingredient_params
            }

            # Test potential soup
            model = get_model_from_sd(potential_greedy_soup_params, base_model)
            val_accuracy = test_model_on_dataset(model, val_dataset)

            # Add to soup if performance improves
            print(f'Potential soup val acc: {val_accuracy:.2f}% (current best: {best_val_acc_so_far:.2f}%)')
            if val_accuracy > best_val_acc_so_far:
                greedy_soup_ingredients.append(sorted_models[i])
                best_val_acc_so_far = val_accuracy
                greedy_soup_params = potential_greedy_soup_params
                print(f'-> Added to soup. New ingredients: {greedy_soup_ingredients}')

        # Evaluate final greedy soup
        model = get_model_from_sd(greedy_soup_params, base_model)
        results = {
            'model_name': 'greedy_soup',
            'ingredients': greedy_soup_ingredients,
            'num_ingredients': len(greedy_soup_ingredients)
        }
        
        print("\nFinal Greedy Soup Evaluation:")
        for split_name, dataset in datasets.items():
            print(f'\nEvaluating on {split_name} set...')
            accuracy = test_model_on_dataset(model, dataset)
            results[split_name] = accuracy
            print(f"{split_name.capitalize()} Accuracy: {accuracy:.2f}%")

        with open(GREEDY_SOUP_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')

    # Step 4: Plot results
    if args.plot and os.path.exists(INDIVIDUAL_MODEL_RESULTS_FILE):
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        
        fig = plt.figure(constrained_layout=True, figsize=(10, 6))
        ax = fig.subplots()

        # Plot individual models
        ax.scatter(
            individual_model_db['val'],
            individual_model_db['test'],
            marker='o',
            color='blue',
            s=100,
            label='Individual Models'
        )
        
        # Annotate each point with model name
        for i, row in individual_model_db.iterrows():
            ax.annotate(row['model_name'], 
                        (row['val'], row['test']),
                        textcoords="offset points",
                        xytext=(0,5),
                        ha='center')

        # Plot greedy soup if available
        if os.path.exists(GREEDY_SOUP_RESULTS_FILE):
            greedy_soup_db = pd.read_json(GREEDY_SOUP_RESULTS_FILE, lines=True)
            ax.scatter(
                greedy_soup_db['val'],
                greedy_soup_db['test'],
                marker='*',
                color='red',
                s=300,
                label='Greedy Soup'
            )
            ax.annotate('Greedy Soup', 
                       (greedy_soup_db['val'].values[0], greedy_soup_db['test'].values[0]),
                       textcoords="offset points",
                       xytext=(0,10),
                       ha='center',
                       weight='bold')

        ax.set_ylabel('Test Accuracy (%)', fontsize=14)
        ax.set_xlabel('Validation Accuracy (%)', fontsize=14)
        ax.set_title('Model Performance Comparison', fontsize=16)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300)
        print("\nSaved performance plot to 'model_performance_comparison.png'")