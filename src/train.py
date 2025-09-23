import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import CarDamageDataset, get_transforms, collate_fn, validate_dataset
from model import get_model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    with tqdm(data_loader, desc=f'Epoch {epoch}') as pbar:
        for images, targets in pbar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    return epoch_loss / num_batches


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Evaluating'):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            total_loss += losses.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_model(data_dir, num_epochs=10, batch_size=2, lr=0.005):
    """
    Main training function
    
    Args:
        data_dir (str): Path to dataset directory
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        lr (float): Learning rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Validate datasets before training
    print("\n" + "="*50)
    print("DATASET VALIDATION")
    print("="*50)
    
    train_valid = validate_dataset(os.path.join(data_dir, 'train'))
    print()
    val_valid = validate_dataset(os.path.join(data_dir, 'val'))
    
    if not train_valid or not val_valid:
        print("\n❌ Dataset validation failed. Please fix the issues before training.")
        return
    
    print("\n✅ All datasets validated successfully!")
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    # Create datasets
    train_dataset = CarDamageDataset(
        os.path.join(data_dir, 'train'),
        transforms=get_transforms(train=True)
    )
    
    val_dataset = CarDamageDataset(
        os.path.join(data_dir, 'val'),
        transforms=get_transforms(train=False)
    )
    
    if len(train_dataset) == 0:
        print("❌ No valid training images found!")
        return
    
    if len(val_dataset) == 0:
        print("❌ No valid validation images found!")
        return
    
    # Create data loaders (num_workers=0 for Windows compatibility)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Create model
    model = get_model(num_classes=2)  # background + damage
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=0.9, weight_decay=0.0005
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        
        lr_scheduler.step()
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'models/best_model.pth')
            print(f'Best model saved with val_loss: {val_loss:.4f}')
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_curves.png')
    plt.show()
    
    print('Training completed!')


if __name__ == '__main__':
    # Train the model
    train_model('dataset', num_epochs=10, batch_size=2, lr=0.005)