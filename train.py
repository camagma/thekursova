import argparse
import copy
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from src.data import SynPainDataset, collect_image_paths
from src.face_detector import load_face_detector


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pain/no-pain detector on SynPain dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root directory for SynPain data.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--model-out", type=Path, default=Path("models/pain_resnet18.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-face", action="store_true", help="Crop detected face during training/inference.")
    parser.add_argument("--face-conf", type=float, default=0.5, help="Face detector confidence.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(
    items: List[Tuple[Path, int]],
    batch_size: int,
    num_workers: int,
    face_net,
    face_conf: float,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    labels = [label for _, label in items]
    train_items, test_items = train_test_split(items, test_size=0.15, stratify=labels, random_state=42)
    train_labels = [label for _, label in train_items]
    train_items, val_items = train_test_split(
        train_items, test_size=0.15, stratify=train_labels, random_state=43
    )

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_ds = SynPainDataset(train_items, transform=train_transform, face_net=face_net, face_conf=face_conf)
    val_ds = SynPainDataset(val_items, transform=eval_transform, face_net=None, face_conf=face_conf)
    test_ds = SynPainDataset(test_items, transform=eval_transform, face_net=None, face_conf=face_conf)

    class_counts = np.bincount([label for _, label in train_items])
    class_weights = torch.tensor((class_counts.sum() / class_counts), dtype=torch.float32)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, class_weights


def build_model(device: torch.device) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return {"acc": acc, "f1": f1, "labels": all_labels, "preds": all_preds}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * images.size(0)
    return epoch_loss / len(loader.dataset)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    items = collect_image_paths(args.data_dir)
    face_net = load_face_detector() if args.use_face else None
    train_loader, val_loader, test_loader, class_weights = build_loaders(
        items, batch_size=args.batch_size, num_workers=args.num_workers, face_net=face_net, face_conf=args.face_conf
    )

    model = build_model(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"loss: {train_loss:.4f} val_acc: {val_metrics['acc']:.3f} val_f1: {val_metrics['f1']:.3f}"
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    print("Test metrics")
    print(f"  accuracy: {test_metrics['acc']:.3f}")
    print(f"  f1-score: {test_metrics['f1']:.3f}")
    print("  confusion matrix:")
    print(confusion_matrix(test_metrics["labels"], test_metrics["preds"]))
    print("  classification report:")
    print(classification_report(test_metrics["labels"], test_metrics["preds"], target_names=["NoPain", "Pain"]))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": {"NoPain": 0, "Pain": 1},
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        args.model_out,
    )
    print(f"Saved best model to {args.model_out}")


if __name__ == "__main__":
    main()
