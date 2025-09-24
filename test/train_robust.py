"""
Robust training script with fixes for NaN losses
"""

import os
import sys
import torch
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('src')

from dataset import CarDamageDataset, get_transforms, collate_fn
from model import get_model


def filter_valid_targets(images, targets):
    """Filter out images with invalid targets"""
    valid_images = []
    valid_targets = []
    
    for img, target in zip(images, targets):
        # Check if target has valid boxes
        if len(target['boxes']) > 0:
            boxes = target['boxes']
            # Check for valid box coordinates
            valid_boxes_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            
            if valid_boxes_mask.any():
                # Filter to only valid boxes
                for key in ['boxes', 'labels', 'masks', 'area', 'iscrowd']:
                    if key in target and len(target[key]) > 0:
                        target[key] = target[key][valid_boxes_mask]
                
                valid_images.append(img)
                valid_targets.append(target)
    
    return valid_images, valid_targets


def robust_train_one_epoch(model, optimizer, data_loader, device, epoch, max_grad_norm=1.0):
    model.train()
    epoch_loss = 0
    num_valid_batches = 0
    
    with tqdm(data_loader, desc=f'Epoch {epoch+1}') as pbar:
        for batch_idx, (images, targets) in enumerate(pbar):
            
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Filter valid targets
            valid_images, valid_targets = filter_valid_targets(images, targets)
            
            if len(valid_images) == 0:
                print(f"⚠️ Batch {batch_idx}: No valid targets, skipping...")
                continue
            
            try:
                # Forward pass
                loss_dict = model(valid_images, valid_targets)
                
                # Check for NaN losses
                losses_list = []
                for k, v in loss_dict.items():
                    if torch.isnan(v) or torch.isinf(v):
                        print(f"⚠️ Invalid loss {k}: {v.item()}, skipping batch...")
                        break
                    losses_list.append(v)
                else:
                    # All losses are valid
                    total_loss = sum(losses_list)
                    
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"⚠️ Total loss is invalid: {total_loss.item()}, skipping batch...")
                        continue
                    
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    
                    # Gradient clipping
                    nn_utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # Check gradients before optimizer step
                    has_nan_grad = False
                    for param in model.parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            has_nan_grad = True
                            break
                    
                    if has_nan_grad:
                        print(f"⚠️ NaN/Inf gradients detected, skipping optimization step...")
                        continue
                    
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    num_valid_batches += 1
                    
                    pbar.set_postfix({
                        'loss': f'{total_loss.item():.4f}',
                        'valid_batches': num_valid_batches
                    })
                
            except Exception as e:
                print(f"⚠️ Error in batch {batch_idx}: {e}")
                continue
    
    if num_valid_batches == 0:
        print("❌ No valid batches processed!")
        return float('nan')
    
    return epoch_loss / num_valid_batches


def robust_train_model(data_dir, num_epochs=5, batch_size=2, initial_lr=0.0001):
    """
    Robust training with multiple safeguards against NaN losses
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🚀 Using device: {device}')
    
    # Create dataset
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
    model = get_model(num_classes=2, pretrained=True)
    model.to(device)
    
    # Optimizer with lower learning rate
    optimizer = torch.optim.Adam(  # Using Adam instead of SGD
        [p for p in model.parameters() if p.requires_grad],
        lr=initial_lr,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=initial_lr * 10,  # Warm up to higher LR
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1  # 10% of training for warmup
    )
    
    print(f"🏋️ Starting training for {num_epochs} epochs...")
    print(f"📊 Training parameters:")
    print(f"   Initial LR: {initial_lr}")
    print(f"   Optimizer: Adam")
    print(f"   Batch size: {batch_size}")
    print(f"   Max gradient norm: 1.0")
    
    train_losses = []
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = robust_train_one_epoch(
            model, optimizer, train_loader, device, epoch, max_grad_norm=1.0
        )
        
        if np.isnan(train_loss):
            print(f"❌ Training failed at epoch {epoch+1}")
            break
        
        train_losses.append(train_loss)
        
        print(f"✅ Epoch {epoch+1} completed - Train Loss: {train_loss:.4f}")
        print(f"📈 Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
        }, 'models/latest_model.pth')
        
        # Save best model
        if epoch == 0 or train_loss < min(train_losses[:-1]):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, 'models/best_model.pth')
            print(f"🏆 New best model saved (loss: {train_loss:.4f})")
    
    if len(train_losses) > 0 and not np.isnan(train_losses[-1]):
        # Plot training curve
        print(f"\n📈 Creating training curve...")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, 'b-o', linewidth=2, markersize=8)
        plt.title('Training Loss (Robust Training)', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        plt.tight_layout()
        plt.savefig('training_curve_robust.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Final Results:")
        print(f"   Final Training Loss: {train_losses[-1]:.4f}")
        print(f"   Best Training Loss: {min(train_losses):.4f}")
        print(f"   Successful Epochs: {len(train_losses)}")
        
        # Test model
        try:
            print(f"\n🔍 Testing model loading...")
            from model import load_model
            test_model = load_model('models/best_model.pth', num_classes=2)
            print(f"✅ Model loading successful!")
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
        
        print(f"\n🚀 Ready for inference! Run:")
        print(f"   streamlit run app.py")
    else:
        print(f"\n❌ Training failed!")
        print(f"Try running debug_train.py first to identify issues")


if __name__ == '__main__':
    robust_train_model('dataset', num_epochs=5, batch_size=1, initial_lr=0.0001)