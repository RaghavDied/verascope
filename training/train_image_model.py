"""
training/train_image_model.py

Fine-tunes an EfficientNet-B4 (via timm) binary deepfake classifier.
Supports:
  - A fresh training run from ImageNet-pretrained weights (checkpoint 3 style)
  - Fine-tuning an existing VeraScope checkpoint on additional/combined data,
    at a lower learning rate (checkpoint 5 style)
  - Resuming an interrupted run from a saved training-state file (automatic,
    based on whether models-dir already has a matching training-state file)

Usage:
  # Fresh training run on the 140k dataset
  python train_image_model.py \
      --manifests /path/to/140k_subset/manifest.csv \
      --models-dir /path/to/models \
      --epochs 10 --batch-size 32 --lr 1e-4 \
      --output-name effnet_b4_deepfake.pt

  # Fine-tune an existing checkpoint on combined 140k + FF++ data
  python train_image_model.py \
      --manifests /path/to/140k_subset/manifest.csv /path/to/ffpp_frames/manifest.csv \
      --models-dir /path/to/models \
      --epochs 4 --batch-size 32 --lr 3e-5 \
      --resume-from /path/to/models/effnet_b4_deepfake.pt \
      --output-name effnet_b4_deepfake_v2.pt
"""
import argparse
import csv
import os
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

IMG_SIZE = 380  # EfficientNet-B4's native input resolution
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, val_transform


class DeepfakeDataset(Dataset):
    """Expects a dataframe with 'filepath' and 'label' columns, label in {'real','fake'}."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {'real': 0, 'fake': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['filepath']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.label_map[row['label']]
        return image, torch.tensor(label, dtype=torch.float32)


class EfficientNetB4Deepfake(nn.Module):
    """Single-logit binary classifier: sigmoid(output) = P(fake).

    IMPORTANT for anyone loading this checkpoint elsewhere (e.g. app/inference.py):
    the saved state_dict is for THIS exact wrapper class, not a raw timm
    efficientnet_b4 -- construct this class first, then load_state_dict into it.
    Input must be a 380x380 RGB image, ImageNet-normalized, and should already
    be a cropped face (see MODEL_HANDOFF.md) -- the model never saw uncropped
    full-scene images during training.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def load_manifests(manifest_paths):
    frames = []
    for path in manifest_paths:
        df = pd.read_csv(path)
        df['split'] = df['split'].replace({'valid': 'val'})
        df['source'] = Path(path).parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run_epoch(model, dataloader, criterion, device, optimizer=None, scaler=None, is_train=True, use_amp=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_targets, all_preds = [], []
    with torch.set_grad_enabled(is_train):
        for images, targets in tqdm(dataloader, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad()
            with amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(targets)
            probs = torch.sigmoid(logits).detach().float().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_targets, binary_preds)
    f1 = f1_score(all_targets, binary_preds)
    auc = roc_auc_score(all_targets, all_preds)
    return avg_loss, acc, f1, auc


def main():
    parser = argparse.ArgumentParser(description="Train/fine-tune the VeraScope EfficientNet-B4 deepfake classifier.")
    parser.add_argument('--manifests', nargs='+', required=True,
                         help="One or more manifest.csv paths (filepath,label,split columns). "
                              "Pass multiple to train on combined data (e.g. 140k + FF++).")
    parser.add_argument('--models-dir', required=True, help="Directory to save checkpoints and logs into.")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--resume-from', default=None,
                         help="Path to an existing .pt weights file to start from (fine-tuning). "
                              "Use a lower --lr (e.g. 3e-5) when fine-tuning.")
    parser.add_argument('--output-name', default='effnet_b4_deepfake.pt',
                         help="Filename for the best-validation-AUC model weights.")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.models_dir, exist_ok=True)
    best_model_path = os.path.join(args.models_dir, args.output_name)
    stem = Path(args.output_name).stem
    latest_state_path = os.path.join(args.models_dir, f"latest_training_state_{stem}.pt")
    log_path = os.path.join(args.models_dir, f"training_log_{stem}.csv")

    combined_df = load_manifests(args.manifests)
    train_df = combined_df[combined_df['split'] == 'train'].reset_index(drop=True)
    val_df = combined_df[combined_df['split'] == 'val'].reset_index(drop=True)
    print(f"Train: {len(train_df)} images | Val: {len(val_df)} images")
    print(train_df.groupby(['source', 'label']).size())

    train_transform, val_transform = build_transforms()
    train_loader = DataLoader(DeepfakeDataset(train_df, train_transform), batch_size=args.batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(DeepfakeDataset(val_df, val_transform), batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    model = EfficientNetB4Deepfake(pretrained=(args.resume_from is None)).to(device)
    if args.resume_from:
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
        print(f"Loaded starting weights from {args.resume_from}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = amp.GradScaler()

    start_epoch = 1
    best_val_auc = -1.0
    if os.path.exists(latest_state_path):
        print("Found existing training state -- resuming.")
        ckpt = torch.load(latest_state_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_auc = ckpt['best_val_auc']
        print(f"Resuming from epoch {start_epoch}. Best val AUC so far: {best_val_auc:.4f}")

    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'train_loss', 'train_acc', 'train_auc', 'val_loss', 'val_acc', 'val_auc', 'lr'])

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc, _, train_auc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, is_train=True)
        val_loss, val_acc, _, val_auc = run_epoch(
            model, val_loader, criterion, device, is_train=False)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}/{args.epochs} -> "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f} --- "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | LR: {current_lr:.2e}")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, train_auc, val_loss, val_acc, val_auc, current_lr])

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_auc': max(best_val_auc, val_auc),
        }, latest_state_path)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best val AUC ({val_auc:.4f}) -- saved to {best_model_path}")

    print(f"\nDone. Best val AUC: {best_val_auc:.4f}. Deliverable: {best_model_path}")


if __name__ == '__main__':
    main()
