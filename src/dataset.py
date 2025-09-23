import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from skimage import draw
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CarDamageDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        
        # Load annotations
        annotation_file = os.path.join(data_dir, 'via_region_data.json')
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Get image list
        self.image_keys = list(self.annotations.keys())
        
    def __len__(self):
        return len(self.image_keys)
    
    def __getitem__(self, idx):
        img_key = self.image_keys[idx]
        img_data = self.annotations[img_key]
        
        # Load image
        img_path = os.path.join(self.data_dir, 'images', img_data['filename'])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Get regions (damage annotations)
        regions = img_data.get('regions', {})
        
        masks = []
        boxes = []
        labels = []
        
        for region_id, region in regions.items():
            shape_attrs = region['shape_attributes']
            if shape_attrs['name'] == 'polygon':
                # Create mask from polygon
                x_points = shape_attrs['all_points_x']
                y_points = shape_attrs['all_points_y']
                
                # Create binary mask
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                rr, cc = draw.polygon(y_points, x_points, shape=mask.shape)
                mask[rr, cc] = 1
                
                # Get bounding box
                x_min, x_max = min(x_points), max(x_points)
                y_min, y_max = min(y_points), max(y_points)
                
                if x_max > x_min and y_max > y_min:
                    masks.append(mask)
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(1)  # damage class
        
        # Convert to tensors
        if len(masks) == 0:
            # No damage annotations
            masks = torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64) if len(boxes) > 0 else torch.tensor([])
        }
        
        if self.transforms:
            # Apply albumentation transforms
            transformed = self.transforms(image=image)
            image = transformed['image']
        else:
            # Convert to tensor if no transforms
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, target


def get_transforms(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def collate_fn(batch):
    return tuple(zip(*batch))


def validate_dataset(data_dir):
    """
    Validate dataset for missing images and annotation issues
    
    Args:
        data_dir (str): Path to dataset directory
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    print(f"Validating dataset: {data_dir}")
    print("-" * 40)
    
    # Check if annotation file exists
    annotation_file = os.path.join(data_dir, 'via_region_data.json')
    if not os.path.exists(annotation_file):
        print(f"❌ Annotation file not found: {annotation_file}")
        return False
    
    # Check if images directory exists
    images_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    # Load annotations
    try:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load annotations: {e}")
        return False
    
    # Check for missing images
    total_annotations = len(annotations)
    missing_images = []
    valid_images = []
    images_with_no_damage = []
    
    for key, img_data in annotations.items():
        filename = img_data['filename']
        img_path = os.path.join(images_dir, filename)
        
        if os.path.exists(img_path):
            valid_images.append(filename)
            # Check if image has damage annotations
            regions = img_data.get('regions', {})
            if len(regions) == 0:
                images_with_no_damage.append(filename)
        else:
            missing_images.append(filename)
    
    # Print validation results
    print(f"📊 Validation Results:")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Valid images found: {len(valid_images)}")
    print(f"  Missing images: {len(missing_images)}")
    print(f"  Images with no damage annotations: {len(images_with_no_damage)}")
    
    if missing_images:
        print(f"\n❌ Missing images:")
        for img in missing_images[:10]:  # Show first 10
            print(f"    - {img}")
        if len(missing_images) > 10:
            print(f"    ... and {len(missing_images) - 10} more")
    
    if images_with_no_damage:
        print(f"\n⚠️  Images with no damage annotations:")
        for img in images_with_no_damage[:5]:  # Show first 5
            print(f"    - {img}")
        if len(images_with_no_damage) > 5:
            print(f"    ... and {len(images_with_no_damage) - 5} more")
    
    is_valid = len(valid_images) > 0 and len(missing_images) == 0
    
    if is_valid:
        print(f"\n✅ Dataset validation passed!")
    else:
        print(f"\n❌ Dataset validation failed!")
        if len(missing_images) > 0:
            print("   Fix: Remove references to missing images from via_region_data.json")
            print("   Or add the missing image files to the images directory")
        if len(valid_images) == 0:
            print("   Fix: Add valid images and annotations")
    
    return is_valid