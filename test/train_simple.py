"""
Simplified training script that focuses on training without complex validation
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('src')

from dataset import CarDamageDataset, get_transforms, collate_fn
from model import get_model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    with tqdm(data_loader, desc=f'Training Epoch {epoch+1}') as pbar:
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


def simple_train_model(data_dir, num_epochs=5, batch_size=2, lr=0.005):
    """
    Simplified training function without validation complexities
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🚀 Using device: {device}')
    
    # Create training dataset only
    print("📂 Loading training dataset...")
    train_dataset = CarDamageDataset(
        os.path.join(data_dir, 'train'),
        transforms=get_transforms(train=True)
    )
    
    print(f"✅ Found {len(train_dataset)} training images")
    
    if len(train_dataset) == 0:
        print("❌ No training images found!")
        return
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    print("🤖 Creating model...")
    model = get_model(num_classes=2)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=0.9, weight_decay=0.0005
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    print(f"🏋️ Starting training for {num_epochs} epochs...")
    train_losses = []
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        train_losses.append(train_loss)
        
        lr_scheduler.step()
        
        print(f"✅ Epoch {epoch+1} completed - Train Loss: {train_loss:.4f}")
        
        # Save model every epoch
        os.makedirs('models', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
        }, 'models/latest_model.pth')
        
        # Save best model (based on training loss)
        if epoch == 0 or train_loss < min(train_losses[:-1]):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, 'models/best_model.pth')
            print(f"🏆 New best model saved (loss: {train_loss:.4f})")
    
    # Plot training curve
    print(f"\n📈 Creating training curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', marker='o', linewidth=2, markersize=8)
    plt.title('Training Loss Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curve_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📊 Final Results:")
    print(f"   Final Training Loss: {train_losses[-1]:.4f}")
    print(f"   Best Training Loss: {min(train_losses):.4f}")
    print(f"   Total Epochs: {num_epochs}")
    print(f"   Model saved to: models/best_model.pth")
    
    # Test model loading
    try:
        print(f"\n🔍 Testing model loading...")
        from model import load_model
        test_model = load_model('models/best_model.pth', num_classes=2)
        print(f"✅ Model loading successful!")
        
        # Quick inference test
        test_model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(3, 512, 512).to(device)
            predictions = test_model([dummy_input])
        print(f"✅ Inference test successful!")
        
    except Exception as e:
        print(f"⚠️ Model testing failed: {e}")
    
    print(f"\n🚀 Ready for inference! Run:")
    print(f"   streamlit run app.py")


if __name__ == '__main__':
    # Simple training with fewer epochs
    simple_train_model('dataset', num_epochs=5, batch_size=2, lr=0.005)