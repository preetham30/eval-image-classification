import os
import tarfile
import subprocess
from tqdm import tqdm
import shutil
import requests
import json
import urllib

from pathlib import Path
from mapping import wnid_to_class

class ImageNetDownloader:
    def __init__(self, root='./imagenet'):
        self.root = root
        self.urls = {
            'train': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar',
            'val': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar',
            'devkit': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz'
        }
        
        self.class_index = self._download_class_index()
        print(f"Total classes in index: {len(self.class_index)}")  # Should be 1000


    def download_file(self, url, filename):
        """More robust download with checks"""
        os.makedirs('downloads', exist_ok=True)
        path = os.path.join('downloads', filename)
        
        if os.path.exists(path):
            try:
                with tarfile.open(path) as test_tar:
                    test_tar.getmembers() 
                print(f"{filename} exists and is valid")
                return path
            except:
                print(f"{filename} exists but is corrupted, re-downloading...")
                os.remove(path)
        
        print(f"Downloading {filename}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)
        
        # Verify download completed
        if os.path.getsize(path) != total_size:
            raise ValueError("Download incomplete!")
        
        return path

    def _download_class_index(self):
        """Download imagenet_class_index.json if missing"""
        output_path = os.path.join(self.root, 'imagenet_class_index.json')
        if not os.path.exists(output_path):
            print("Downloading imagenet_class_index.json...")
            subprocess.run([
                "wget", 
                "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
                "-O", output_path
            ], check=True)
        with open(output_path) as f:
            return json.load(f)  # Format: {"0": ["n01440764", "tench"], ...}

    def _extract_devkit(self):
        """Extract the devkit to get validation ground truth"""
        devkit_path = self.download_file(self.urls['devkit'], 'devkit.tar.gz')
        devkit_dir = os.path.join(self.root, 'devkit')
        os.makedirs(devkit_dir, exist_ok=True)
        
        with tarfile.open(devkit_path) as tar:
            tar.extractall(devkit_dir)
        
        # Find  ground truth file
        gt_file = None
        for root, _, files in os.walk(devkit_dir):
            if 'ILSVRC2012_validation_ground_truth.txt' in files:
                gt_file = os.path.join(root, 'ILSVRC2012_validation_ground_truth.txt')
                break
        
        if not gt_file:
            raise FileNotFoundError("Could not find validation ground truth file in devkit")
        
        return gt_file

    def _organize_validation_set(self, val_dir, gt_file, class_name=False):
        """
        Organizes the ImageNet 2012 validation set into classification format using valprep.sh approach.
        
        Args:
            val_dir (str): Directory containing unpacked validation images.
            gt_file (str): Path to ILSVRC2012_validation_ground_truth.txt.
        """
        script_url = "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"
        script_path = os.path.join(val_dir, "valprep.sh")

        # Download valprep.sh if not present
        if not os.path.exists(script_path):
            print("Downloading valprep.sh...")
            urllib.request.urlretrieve(script_url, script_path)
            os.chmod(script_path, 0o755)

        # Ensure ground truth file is in the same directory (as expected by script)
        gt_filename = os.path.basename(gt_file)
        if os.path.abspath(os.path.dirname(gt_file)) != os.path.abspath(val_dir):
            dest_gt = os.path.join(val_dir, gt_filename)
            shutil.copy(gt_file, dest_gt)
            gt_file = dest_gt

        # Run the shell script
        print("Organizing validation set using valprep.sh...")
        subprocess.run(["bash", "valprep.sh"], cwd=val_dir, check=True)
        print("Done organizing validation images.")

        gt_path = os.path.join(val_dir, 'ILSVRC2012_validation_ground_truth.txt')
        script_path = os.path.join(val_dir, 'valprep.sh')

        if os.path.exists(gt_path):
            shutil.os.remove(gt_path)

        if os.path.exists(script_path):
            shutil.os.remove(script_path)

        if class_name:
            print("Organizing validation set based on class names...")
            for dir_name in os.listdir(val_dir):
                dir_path = os.path.join(val_dir, dir_name)
                if os.path.isdir(dir_path) and dir_name in wnid_to_class:
                    new_name = wnid_to_class[dir_name]
                    class_id_name = f"{new_name}"
                    new_path = os.path.join(val_dir, class_id_name)
                    os.rename(dir_path, new_path)
                    print(f"Renamed: {dir_name} → {class_id_name }")
        

    def prepare_imagenet(self):
        try:
            print("Starting ImageNet preparation...")
            
            # 1. Download files with verification
            train_tar = self.download_file(self.urls['train'], 'train.tar')
            val_tar = self.download_file(self.urls['val'], 'val.tar')

            # 2. Extract with validation
            self._extract_train(train_tar)
            self._extract_val(val_tar)
            
            print("\nImageNet successfully prepared at:", self.root)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Cleaning up temporary files...")
            shutil.rmtree('temp_train', ignore_errors=True)
            shutil.rmtree(os.path.join(self.root, 'train'), ignore_errors=True)
            shutil.rmtree(os.path.join(self.root, 'val'), ignore_errors=True)
            raise

    def prepare_imagenet_zeroshot(self):
        try:
            print("Starting ImageNet zero-shot validation preparation...")
            
            # 1. Download files with verification
            val_tar = '/home/preetham/evals_reasoning/downloads/val.tar'

            # 2. Extract with validation
            self._extract_val(val_tar, 'val_zeroshot')
            
            print("\nImageNet successfully prepared at:", self.root)

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Cleaning up temporary files...")
            shutil.rmtree('temp_dir', ignore_errors=True)
            raise

    def _extract_train(self, tar_path):
        """Two-stage extraction of training data"""
        print("\nExtracting training set...")
        train_dir = os.path.join(self.root, 'train')
        os.makedirs(train_dir, exist_ok=True)

        temp_dir = 'temp_dir'
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # First extract: the main tar file contains multiple class tar files
            with tarfile.open(tar_path) as tar:
                tar.extractall(temp_dir)

            # Second extract: each class tar file into its own folder
            for class_tar in tqdm(os.listdir(temp_dir), desc="Extracting class archives"):
                if not class_tar.endswith('.tar'):
                    continue

                class_id = os.path.splitext(class_tar)[0]
                class_dir = os.path.join(train_dir, class_id)
                os.makedirs(class_dir, exist_ok=True)

                class_tar_path = os.path.join(temp_dir, class_tar)
                with tarfile.open(class_tar_path) as tar:
                    tar.extractall(class_dir)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _extract_val(self, tar_path, val_dir_name=None):
        """Extract and organize validation set"""
        print("\nExtracting validation set...")
        if val_dir_name is None:
            val_dir = os.path.join(self.root, 'val')
            os.makedirs(val_dir, exist_ok=True)
        else:
            val_dir = os.path.join(self.root, val_dir_name)
            os.makedirs(val_dir, exist_ok=True)
        
        # Verify tar integrity first
        try:
            with tarfile.open(tar_path) as test_tar:
                test_tar.getmembers()
        except tarfile.ReadError:
            raise ValueError("Validation tar file is corrupted! Please delete and re-download")
        
        # Actual extraction
        with tarfile.open(tar_path) as tar:
            tar.extractall(val_dir)
        
        # Download devkit for ground truth
        print("Downloading devkit for validation labels...")
        gt_file = self._extract_devkit()
        
        # Organize validation set
        if val_dir_name is None:
            self._organize_validation_set(val_dir, gt_file)
        else:
            self._organize_validation_set(val_dir, gt_file, class_name=True)

if __name__ == '__main__':
    downloader = ImageNetDownloader()

    #downloader.prepare_imagenet()

    # Validation datatset prep for zero-shot classfication
    downloader.prepare_imagenet_zeroshot()