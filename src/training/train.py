import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
import mlflow

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.dataset import TuSimpleDataset
from src.models.model import build_model
from src.training.losses import DiceBCELoss
from src.training.metrics import compute_iou


def train(data_root, checkpoint_dir='models', num_epochs=20, lr=1e-3,
          batch_size=16, resume_checkpoint=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Datasets
    full_dataset_aug = TuSimpleDataset(root_dir=data_root, augment=True)
    full_dataset_noaug = TuSimpleDataset(root_dir=data_root, augment=False)

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.85 * len(full_dataset_aug))
    val_size = len(full_dataset_aug) - train_size
    train_indices, val_indices = random_split(
        range(len(full_dataset_aug)), [train_size, val_size], generator=generator
    )

    train_dataset = Subset(full_dataset_aug, train_indices.indices)
    val_dataset = Subset(full_dataset_noaug, val_indices.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = build_model().to(device)
    if resume_checkpoint:
        model.load_state_dict(torch.load(resume_checkpoint, map_location=device))
        print(f"Resumed from {resume_checkpoint}")

    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    mlflow.set_experiment("lane-detection")
    best_val_iou = 0.0

    with mlflow.start_run():
        mlflow.log_params({
            "lr": lr, "batch_size": batch_size, "epochs": num_epochs,
            "encoder": "mobilenet_v2", "loss": "Dice+BCE"
        })

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                images, masks = images.to(device).float(), masks.to(device).float()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss, val_iou = 0.0, 0.0
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    images, masks = images.to(device).float(), masks.to(device).float()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    val_iou += compute_iou(outputs, masks).item()
            avg_val_loss = val_loss / len(val_loader)
            avg_val_iou = val_iou / len(val_loader)

            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Val IoU={avg_val_iou:.4f}")

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_iou", avg_val_iou, step=epoch)

            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                save_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), save_path)
                print(f"  -> New best model saved (IoU: {best_val_iou:.4f})")

    print("Training complete!")
    return best_val_iou


if __name__ == "__main__":
    train(data_root='/content/data/TUSimple/train_set')