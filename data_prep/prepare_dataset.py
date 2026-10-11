"""
data_prep/prepare_dataset.py

Downloads the "140k Real and Fake Faces" dataset via kagglehub, subsamples
it to a manageable train/val/test split, and writes a manifest.csv that
training/train_image_model.py (and explainability/gradcam.py) consume.

Requires kagglehub auth beforehand: either call kagglehub.login() once
interactively, or set the KAGGLE_USERNAME / KAGGLE_KEY environment variables.

Usage:
  python prepare_dataset.py --output-dir /path/to/data/140k_subset \
      --n-train 8000 --n-val 1000 --n-test 1000
"""
import argparse
import random
import shutil
from pathlib import Path

import kagglehub
import pandas as pd

KAGGLE_SLUG = "xhlulu/140k-real-and-fake-faces"


def find_split_class_dirs(root, min_images=100):
    """Recursively find 'real'/'fake' folders with enough images, grouped by
    their parent (split) folder name -- avoids hardcoding exact nesting depth,
    which varies across Kaggle mirrors."""
    root = Path(root)
    found = {}
    for cls in ['real', 'fake']:
        for p in root.rglob(cls):
            if p.is_dir():
                img_files = list(p.glob('*.jpg')) + list(p.glob('*.png'))
                if len(img_files) > min_images:
                    split_name = p.parent.name.lower()
                    found.setdefault(split_name, {})[cls] = p
    return found


def main():
    parser = argparse.ArgumentParser(description="Download + subsample the 140k Real/Fake Faces dataset.")
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--n-train', type=int, default=8000, help="Images per class for train.")
    parser.add_argument('--n-val', type=int, default=1000, help="Images per class for val.")
    parser.add_argument('--n-test', type=int, default=1000, help="Images per class for test.")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    for split in ['train', 'val', 'test']:
        for cls in ['real', 'fake']:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    print(f"Downloading {KAGGLE_SLUG} via kagglehub...")
    dataset_path = kagglehub.dataset_download(KAGGLE_SLUG)
    print("Downloaded to:", dataset_path)

    split_dirs = find_split_class_dirs(dataset_path)
    print("Detected splits:", {k: list(v.keys()) for k, v in split_dirs.items()})

    split_aliases = {'train': 'train', 'valid': 'val', 'val': 'val', 'test': 'test'}
    n_per_class = {'train': args.n_train, 'val': args.n_val, 'test': args.n_test}

    for detected_split, classes in split_dirs.items():
        our_split = split_aliases.get(detected_split)
        if our_split is None:
            print(f"Skipping unrecognized split folder: {detected_split}")
            continue
        for cls, src_dir in classes.items():
            dst_dir = output_dir / our_split / cls
            all_files = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png'))
            n = min(n_per_class[our_split], len(all_files))
            for f in random.sample(all_files, n):
                shutil.copy(f, dst_dir / f.name)
            print(f'{our_split}/{cls}: copied {n} of {len(all_files)} available')

    records = []
    for split in ['train', 'val', 'test']:
        for cls in ['real', 'fake']:
            for f in (output_dir / split / cls).iterdir():
                if f.is_file():
                    records.append({'filepath': str(f), 'label': cls, 'split': split})

    manifest_df = pd.DataFrame(records)
    manifest_path = output_dir / 'manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    print(manifest_df.groupby(['split', 'label']).size())
    print(f"Manifest written to {manifest_path} ({len(manifest_df)} images total)")


if __name__ == '__main__':
    main()
