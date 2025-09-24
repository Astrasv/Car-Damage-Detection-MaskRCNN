"""
Quick test script to verify training setup works
"""

import sys
import torch
sys.path.append('src')

from dataset import CarDamageDataset, get_transforms, validate_dataset
from model import get_model


def test_dataset_loading():
    """Test if dataset can be loaded without errors"""
    print("🧪 Testing dataset loading...")
    
    try:
        # Validate dataset
        train_valid = validate_dataset('dataset/train')
        val_valid = validate_dataset('dataset/val')
        
        if not train_valid or not val_valid:
            print("❌ Dataset validation failed")
            return False
        
        # Create datasets
        train_dataset = CarDamageDataset(
            'dataset/train',
            transforms=get_transforms(train=True)
        )
        
        val_dataset = CarDamageDataset(
            'dataset/val', 
            transforms=get_transforms(train=False)
        )
        
        print(f"✅ Train dataset: {len(train_dataset)} images")
        print(f"✅ Val dataset: {len(val_dataset)} images")
        
        # Test loading one sample
        if len(train_dataset) > 0:
            image, target = train_dataset[0]
            print(f"✅ Sample loaded - Image shape: {image.shape}")
            print(f"✅ Target keys: {list(target.keys())}")
            print(f"✅ Number of damage regions: {len(target['boxes'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False


def test_model_creation():
    """Test if model can be created"""
    print("\n🧪 Testing model creation...")
    
    try:
        model = get_model(num_classes=2, pretrained=True)
        print(f"✅ Model created successfully")
        
        # Test forward pass with dummy data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Create dummy input
        dummy_image = torch.randn(1, 3, 512, 512).to(device)
        dummy_target = {
            'boxes': torch.tensor([[10, 10, 100, 100]], dtype=torch.float32).to(device),
            'labels': torch.tensor([1], dtype=torch.int64).to(device),
            'masks': torch.randint(0, 2, (1, 512, 512), dtype=torch.uint8).to(device)
        }
        
        # Test training mode (should return losses)
        model.train()
        loss_dict = model([dummy_image], [dummy_target])
        print(f"✅ Training forward pass successful")
        print(f"✅ Loss keys: {list(loss_dict.keys())}")
        
        # Test eval mode (should return predictions)
        model.eval()
        with torch.no_grad():
            predictions = model([dummy_image])
        print(f"✅ Evaluation forward pass successful")
        print(f"✅ Prediction keys: {list(predictions[0].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation/testing failed: {e}")
        return False


def test_training_components():
    """Test training components"""
    print("\n🧪 Testing training components...")
    
    try:
        from torch.utils.data import DataLoader
        from dataset import collate_fn
        
        # Test data loader
        dataset = CarDamageDataset('dataset/train', transforms=get_transforms(train=True))
        
        if len(dataset) == 0:
            print("⚠️ No training data available for testing")
            return True
        
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Test one batch
        for images, targets in loader:
            print(f"✅ DataLoader working - Batch size: {len(images)}")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Training components test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚗 Car Damage Detection - Training Test")
    print("=" * 50)
    
    # Test components
    dataset_ok = test_dataset_loading()
    model_ok = test_model_creation() 
    training_ok = test_training_components()
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Dataset Loading: {'✅ PASS' if dataset_ok else '❌ FAIL'}")
    print(f"   Model Creation: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"   Training Components: {'✅ PASS' if training_ok else '❌ FAIL'}")
    
    if dataset_ok and model_ok and training_ok:
        print("\n🎉 All tests passed! You can start training:")
        print("   python src/train.py")
        print("\n🚀 Or run a quick 2-epoch test:")
        print("   python -c \"from src.train import train_model; train_model('dataset', num_epochs=2, batch_size=1)\"")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before training.")
        
        if not dataset_ok:
            print("\n🔧 Dataset Issues - Try:")
            print("1. Run: python clean_dataset.py")
            print("2. Check that images exist in dataset/train/images/ and dataset/val/images/")
            print("3. Verify via_region_data.json files are valid")
        
        if not model_ok:
            print("\n🔧 Model Issues - Try:")
            print("1. Check PyTorch installation: pip install torch torchvision")
            print("2. Verify CUDA compatibility if using GPU")
        
        if not training_ok:
            print("\n🔧 Training Component Issues - Try:")
            print("1. Check dataset structure with: python setup_dataset.py")
            print("2. Reduce batch size or use CPU if memory issues")


if __name__ == "__main__":
    main()