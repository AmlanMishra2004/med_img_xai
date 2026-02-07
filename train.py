"""
CheXpert Training Script - Replicating the training from the paper
Based on DenseNet-121 with uncertainty handling (U-on policy)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm
import time

# Configuration based on the paper
class Config:
    # Dataset
    img_size = 224
    num_classes = 14  # CheXpert has 14 observations
    
    # Training
    batch_size = 16
    epochs = 3  # As specified in paper: Batch Size / Epochs: 16/3
    
    # Optimizer (Adam with β1=0.9, β2=0.999)
    learning_rate = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    
    # Scheduler (ReduceLROnPlateau)
    scheduler_factor = 0.1
    scheduler_patience = 1
    
    # Early stopping
    early_stop_patience = 3
    
    # Data paths - update these to your actual paths
    data_root = os.path.expanduser("~/.cache/kagglehub/datasets/ashery/chexpert/versions/1")
    train_csv = "train.csv"
    valid_csv = "valid.csv"
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpointing
    checkpoint_dir = "./checkpoints"
    best_model_path = "./best_model.pth"


class CheXpertDataset(Dataset):
    """CheXpert Dataset with U-on policy for uncertain labels"""
    
    # The 14 observations in CheXpert
    PATHOLOGIES = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    def __init__(self, csv_path, data_root, transform=None, u_policy='ones'):
        """
        Args:
            csv_path: Path to the CSV file
            data_root: Root directory of the dataset
            transform: torchvision transforms
            u_policy: How to handle uncertain labels (-1)
                     'ones' (U-on): Map -1 to 1 (positive)
                     'zeros' (U-zero): Map -1 to 0 (negative)
        """
        self.data_root = data_root
        self.transform = transform
        self.u_policy = u_policy
        
        # Read CSV
        self.df = pd.read_csv(os.path.join(data_root, csv_path))
        
        # Handle uncertain labels according to policy
        for pathology in self.PATHOLOGIES:
            if pathology in self.df.columns:
                if u_policy == 'ones':
                    # U-on: Map -1 (uncertain) to 1 (positive)
                    self.df[pathology] = self.df[pathology].replace(-1, 1)
                elif u_policy == 'zeros':
                    # U-zero: Map -1 (uncertain) to 0 (negative)
                    self.df[pathology] = self.df[pathology].replace(-1, 0)
                
                # Fill NaN with 0
                self.df[pathology] = self.df[pathology].fillna(0)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image path from CSV
        img_path_from_csv = self.df.iloc[idx]['Path']
        
        # Handle path: CSV might have 'CheXpert-v1.0-small/' prefix that doesn't exist in actual structure
        # Remove 'CheXpert-v1.0-small/' prefix if present
        if img_path_from_csv.startswith('CheXpert-v1.0-small/'):
            img_path_from_csv = img_path_from_csv.replace('CheXpert-v1.0-small/', '', 1)
        
        # Build full path
        img_path = os.path.join(self.data_root, img_path_from_csv)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels for all 14 pathologies
        labels = []
        for pathology in self.PATHOLOGIES:
            label = self.df.iloc[idx][pathology] if pathology in self.df.columns else 0
            labels.append(label)
        
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, labels


def get_transforms(train=True):
    """
    Get transforms as specified in the paper:
    - Resize to 224x224
    - Normalize with ImageNet mean/std
    """
    if train:
        return transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 pretrained on ImageNet
    Modified for multi-label classification with sigmoid activation
    """
    def __init__(self, num_classes=14):
        super(DenseNet121Classifier, self).__init__()
        
        # Load pretrained DenseNet-121 (using weights parameter for newer PyTorch)
        try:
            # For PyTorch >= 0.13
            from torchvision.models import DenseNet121_Weights
            self.densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        except ImportError:
            # For older PyTorch versions
            self.densenet = models.densenet121(pretrained=True)
        
        # Replace the classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        # Forward pass
        logits = self.densenet(x)
        # Apply sigmoid for independent probabilities per class
        probs = torch.sigmoid(logits)
        return logits, probs


class BCEWithLogLossPerLabel(nn.Module):
    """
    Binary Cross-Entropy loss computed per label independently
    As specified in the paper: BCE with log
    """
    def __init__(self):
        super(BCEWithLogLossPerLabel, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, targets):
        # Compute BCE per sample per label
        loss_per_label = self.bce(logits, targets)
        
        # Average across samples and labels
        # Paper specifies: -1/N * sum over N samples * sum over C classes
        loss = loss_per_label.mean()
        
        return loss


def compute_auroc(targets, predictions):
    """Compute AUROC for each class"""
    aurocs = []
    for i in range(targets.shape[1]):
        try:
            auroc = roc_auc_score(targets[:, i], predictions[:, i])
            aurocs.append(auroc)
        except:
            aurocs.append(np.nan)
    return aurocs


def train_epoch(model, dataloader, criterion, optimizer, device, epoch_num):
    """Train for one epoch with progress bar"""
    model.train()
    running_loss = 0.0
    
    # Progress bar for training
    pbar = tqdm(dataloader, desc=f'Epoch {epoch_num} [Train]', 
                leave=True, dynamic_ncols=True)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, probs = model(images)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update progress bar with current loss
        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device, epoch_num):
    """Validate the model with progress bar"""
    model.eval()
    running_loss = 0.0
    
    all_labels = []
    all_probs = []
    
    # Progress bar for validation
    pbar = tqdm(dataloader, desc=f'Epoch {epoch_num} [Valid]', 
                leave=True, dynamic_ncols=True)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, probs = model(images)
            
            # Compute loss
            loss = criterion(logits, labels)
            running_loss += loss.item()
            
            # Store for AUROC computation
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_loss = running_loss / len(dataloader)
    
    # Compute AUROC
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    aurocs = compute_auroc(all_labels, all_probs)
    mean_auroc = np.nanmean(aurocs)
    
    return avg_loss, mean_auroc, aurocs


def train_model():
    """Main training function"""
    
    # Create checkpoint directory
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    
    # Initialize model
    print(f"Initializing DenseNet-121 on {Config.device}")
    model = DenseNet121Classifier(num_classes=Config.num_classes)
    model = model.to(Config.device)
    
    # Loss function
    criterion = BCEWithLogLossPerLabel()
    
    # Optimizer (Adam with β1=0.9, β2=0.999)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=Config.learning_rate,
        betas=(Config.beta1, Config.beta2)
    )
    
    # Scheduler (ReduceLROnPlateau based on validation loss)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=Config.scheduler_factor,
        patience=Config.scheduler_patience
    )
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = CheXpertDataset(
        csv_path=Config.train_csv,
        data_root=Config.data_root,
        transform=get_transforms(train=True),
        u_policy='ones'  # U-on policy as per paper
    )
    
    valid_dataset = CheXpertDataset(
        csv_path=Config.valid_csv,
        data_root=Config.data_root,
        transform=get_transforms(train=False),
        u_policy='ones'
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    best_auroc = 0.0
    epochs_without_improvement = 0
    
    print("\nStarting training...")
    print(f"Total epochs: {Config.epochs}")  # 3 epochs as per paper
    print(f"Batch size: {Config.batch_size}")
    print(f"Initial learning rate: {Config.learning_rate}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Total training steps: {len(train_loader) * Config.epochs}")
    print("=" * 80)
    
    # Track total training time
    total_start_time = time.time()
    epoch_times = []
    
    for epoch in range(Config.epochs):
        epoch_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{Config.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, Config.device, epoch + 1)
        
        # Validate
        val_loss, val_auroc, class_aurocs = validate(model, valid_loader, criterion, Config.device, epoch + 1)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Estimate remaining time
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = Config.epochs - (epoch + 1)
        estimated_time_remaining = avg_epoch_time * remaining_epochs
        
        print(f"\n{'─'*80}")
        print(f"Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val AUROC:  {val_auroc:.4f}")
        print(f"\nTiming:")
        print(f"  Epoch time: {epoch_time/60:.1f} min ({epoch_time:.0f}s)")
        if remaining_epochs > 0:
            print(f"  Estimated time remaining: {estimated_time_remaining/60:.1f} min ({estimated_time_remaining/3600:.2f}h)")
        print(f"  Total elapsed: {(time.time() - total_start_time)/60:.1f} min")
        
        # Print top 5 pathologies by AUROC
        print(f"\nTop 5 Pathologies by AUROC:")
        auroc_pairs = [(CheXpertDataset.PATHOLOGIES[i], auroc) 
                       for i, auroc in enumerate(class_aurocs) if not np.isnan(auroc)]
        auroc_pairs.sort(key=lambda x: x[1], reverse=True)
        for pathology, auroc in auroc_pairs[:5]:
            print(f"  {pathology:30s}: {auroc:.4f}")
        
        # Get current learning rate before scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        
        # Step scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Get current learning rate after scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check if learning rate was reduced
        if current_lr < old_lr:
            print(f"\n⚠ Learning rate reduced: {old_lr:.2e} → {current_lr:.2e}")
        
        print(f"\nCurrent learning rate: {current_lr:.2e}")
        
        # Save best model based on AUROC
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
                'val_loss': val_loss,
                'class_aurocs': class_aurocs,
                'train_loss': train_loss,
                'epoch_time': epoch_time
            }, Config.best_model_path)
            print(f"\n✓ New best model saved! AUROC: {best_auroc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"\n⚠ No improvement for {epochs_without_improvement} epoch(s)")
        
        # Early stopping
        if epochs_without_improvement >= Config.early_stop_patience:
            print(f"\n{'='*80}")
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"{'='*80}")
            break
    
    # Final summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"{'='*80}")
    print(f"Best validation AUROC: {best_auroc:.4f}")
    print(f"Total training time: {total_time/60:.1f} min ({total_time/3600:.2f}h)")
    print(f"Average time per epoch: {np.mean(epoch_times)/60:.1f} min")
    print(f"{'='*80}")
    
    return model


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train the model
    model = train_model()
    
    print("\nDone! Best model saved to:", Config.best_model_path)