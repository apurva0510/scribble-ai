import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from model import QuickDrawCNN, save_model


class QuickDrawImageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.class_names = sorted(
            path.name for path in self.data_dir.iterdir() if path.is_dir()
        )

        if not self.class_names:
            raise ValueError(f"No class folders found in {self.data_dir}")

        self.samples = []
        for label, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            for image_path in sorted(class_dir.glob("*.png")):
                self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(f"No PNG images found under {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L").resize((28, 28))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.tensor(image_array).view(1, 28, 28)
        return image_tensor, label


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total if total else 0.0


def train_model(args):
    dataset = QuickDrawImageDataset(args.data_dir)
    train_size = max(1, int(len(dataset) * args.train_split))
    val_size = len(dataset) - train_size

    if val_size == 0:
        train_dataset = dataset
        val_dataset = None
    else:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size)
        if val_dataset is not None
        else None
    )

    model = QuickDrawCNN(num_classes=len(dataset.class_names))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        message = (
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={total_loss / total:.4f} accuracy={correct / total:.3f}"
        )

        if val_loader is not None:
            message += f" val_accuracy={evaluate(model, val_loader):.3f}"

        print(message)

    save_model(model, dataset.class_names, args.output)
    print(f"Saved model to {args.output}")
    print(f"Classes: {', '.join(dataset.class_names)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a QuickDraw doodle classifier.")
    parser.add_argument("--data-dir", default="data/quickdraw")
    parser.add_argument("--output", default="models/quickdraw_model.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
