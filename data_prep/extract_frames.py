"""
data_prep/extract_frames.py

Downloads a stratified subset of FaceForensics++ (c23, Kaggle mirror) videos,
face-crops evenly-spaced frames from each with MTCNN, and writes a
manifest.csv in the same schema as prepare_dataset.py's output -- so the
same DeepfakeDataset / train_image_model.py work on this data unmodified.

Splits are assigned PER VIDEO before extraction, to avoid frames from the
same video leaking across train/val/test (frames from one video are highly
correlated, so splitting at the frame level would quietly inflate test
accuracy).

Requires kagglehub auth beforehand, same as prepare_dataset.py.

Usage:
  python extract_frames.py --output-dir /path/to/data/ffpp_frames \
      --manipulation-types Deepfakes Face2Face \
      --n-real 180 --n-fake-per-type 90 --frames-per-video 15
"""
import argparse
import random
from pathlib import Path

import cv2
import kagglehub
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm.auto import tqdm

KAGGLE_SLUG = "xdxd003/ff-c23"


def find_folder(root, name):
    matches = list(Path(root).rglob(name))
    return matches[0] if matches else None


def assign_splits(items, train_frac=0.7, val_frac=0.15, rng=None):
    items = list(items)
    (rng or random).shuffle(items)
    n = len(items)
    n_train, n_val = int(n * train_frac), int(n * val_frac)
    return ([(i, 'train') for i in items[:n_train]] +
            [(i, 'val') for i in items[n_train:n_train + n_val]] +
            [(i, 'test') for i in items[n_train + n_val:]])


def extract_and_crop_frames(video_path, out_dir, n_frames, source_tag, mtcnn):
    """Reusable at inference time too -- see MODEL_HANDOFF.md for how
    Raghav's app should call the same face-crop step on user uploads."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    saved = 0
    stem = Path(video_path).stem
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out_path = Path(out_dir) / f"{stem}_{source_tag}_f{int(idx)}.jpg"
        face = mtcnn(pil_img, save_path=str(out_path))  # MTCNN saves the crop itself
        if face is not None:
            saved += 1
    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="FF++ subset download + face-cropped frame extraction.")
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--manipulation-types', nargs='+', default=['Deepfakes', 'Face2Face'])
    parser.add_argument('--n-real', type=int, default=180)
    parser.add_argument('--n-fake-per-type', type=int, default=90)
    parser.add_argument('--frames-per-video', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    print(f"Downloading {KAGGLE_SLUG} via kagglehub...")
    dataset_path = kagglehub.dataset_download(KAGGLE_SLUG)
    ff_root = Path(dataset_path)

    real_dir = find_folder(ff_root, 'original')
    if real_dir is None:
        raise FileNotFoundError("Could not find 'original' folder in the downloaded dataset -- check its structure.")
    real_videos = list(real_dir.glob('*.mp4'))
    real_with_splits = assign_splits(rng.sample(real_videos, min(args.n_real, len(real_videos))), rng=rng)

    fake_with_splits = []
    for m_type in args.manipulation_types:
        m_dir = find_folder(ff_root, m_type)
        if m_dir is None:
            print(f"WARNING: manipulation type '{m_type}' not found, skipping.")
            continue
        m_videos = list(m_dir.glob('*.mp4'))
        sampled = rng.sample(m_videos, min(args.n_fake_per_type, len(m_videos)))
        for video, split in assign_splits(sampled, rng=rng):
            fake_with_splits.append((video, split, m_type))

    print(f"Real: {len(real_with_splits)} videos | Fake: {len(fake_with_splits)} videos")

    mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device=device)

    manifest_records = []
    print("Extracting real videos...")
    for video_path, split in tqdm(real_with_splits):
        n = extract_and_crop_frames(video_path, output_dir / split / 'real', args.frames_per_video, 'real', mtcnn)
        manifest_records.append({'video': video_path.name, 'label': 'real', 'manipulation': 'original',
                                  'split': split, 'frames_saved': n})

    print("Extracting fake videos...")
    for video_path, split, m_type in tqdm(fake_with_splits):
        n = extract_and_crop_frames(video_path, output_dir / split / 'fake', args.frames_per_video, m_type, mtcnn)
        manifest_records.append({'video': video_path.name, 'label': 'fake', 'manipulation': m_type,
                                  'split': split, 'frames_saved': n})

    video_manifest_df = pd.DataFrame(manifest_records)
    video_manifest_df.to_csv(output_dir / 'video_manifest.csv', index=False)
    print(video_manifest_df.groupby(['split', 'label'])['frames_saved'].sum())
    print(f"Videos with zero faces detected: {(video_manifest_df['frames_saved'] == 0).sum()}")

    records = []
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            for f in (output_dir / split / label).glob('*.jpg'):
                records.append({'filepath': str(f), 'label': label, 'split': split})

    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(output_dir / 'manifest.csv', index=False)
    print(manifest_df.groupby(['split', 'label']).size())
    print(f"Total frames: {len(manifest_df)}")


if __name__ == '__main__':
    main()
