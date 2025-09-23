import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import load_model


class CarDamagePredictor:
    def __init__(self, model_path, confidence_threshold=0.1, device=None):  # Lower default threshold
        self.confidence_threshold = confidence_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = load_model(model_path, num_classes=2)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device} with confidence threshold {confidence_threshold}")
        
        # Preprocessing pipeline
        self.transforms = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, str):
            # Load from path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Apply transforms
        transformed = self.transforms(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor, image_np
    
    def predict(self, image, debug=False):
        """
        Make predictions on an image
        
        Args:
            image: PIL Image, numpy array, or path to image
            debug: Print debug information
            
        Returns:
            dict: Prediction results containing boxes, masks, scores
        """
        # Preprocess
        image_tensor, original_image = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # CRITICAL FIX: Ensure tensor is 3D for model input
        if len(image_tensor.shape) == 4:
            # Remove batch dimension: [1, 3, 512, 512] -> [3, 512, 512]
            image_tensor = image_tensor.squeeze(0)
        
        if debug:
            print(f"Input image shape: {image_tensor.shape}")
            print(f"Original image shape: {original_image.shape}")
        
        # Predict - pass 3D tensor in a list
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]  # Now image_tensor is [3, 512, 512]
        
        if debug:
            print(f"Raw predictions keys: {list(predictions.keys())}")
            print(f"Number of raw detections: {len(predictions['scores'])}")
            if len(predictions['scores']) > 0:
                print(f"Score range: {predictions['scores'].min().item():.4f} - {predictions['scores'].max().item():.4f}")
        
        # Get all predictions first (before filtering)
        all_boxes = predictions['boxes'].cpu().numpy()
        all_masks = predictions['masks'].cpu().numpy()
        all_scores = predictions['scores'].cpu().numpy()
        all_labels = predictions['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        keep = predictions['scores'] > self.confidence_threshold
        
        boxes = all_boxes[keep]
        masks = all_masks[keep]
        scores = all_scores[keep]
        labels = all_labels[keep]
        
        if debug:
            print(f"Detections above threshold {self.confidence_threshold}: {len(boxes)}")
            if len(scores) > 0:
                print(f"Kept scores: {scores}")
        
        # Scale boxes back to original image size
        original_h, original_w = original_image.shape[:2]
        scale_x = original_w / 512
        scale_y = original_h / 512
        
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x  # x coordinates
            boxes[:, [1, 3]] *= scale_y  # y coordinates
            
            # Resize masks to original image size
            resized_masks = []
            for mask in masks:
                if len(mask.shape) == 3:
                    mask = mask[0]  # Remove batch dimension if present
                mask_resized = cv2.resize(mask, (original_w, original_h))
                resized_masks.append(mask_resized)
            masks = np.array(resized_masks) if resized_masks else np.array([])
        
        result = {
            'boxes': boxes,
            'masks': masks,
            'scores': scores,
            'labels': labels,
            'num_detections': len(boxes),
            'all_scores': all_scores,  # For debugging
            'all_boxes': all_boxes,   # For debugging
        }
        
        if debug:
            print(f"Final result: {result['num_detections']} detections")
        
        return result
    
    def visualize_predictions(self, image, predictions, mask_alpha=0.3, box_thickness=3):
        """
        Visualize predictions on image with both bounding boxes and masks
        
        Args:
            image: PIL Image or numpy array
            predictions: Prediction results from predict()
            mask_alpha: Transparency of mask overlay
            box_thickness: Thickness of bounding box lines
            
        Returns:
            PIL Image: Image with visualizations
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Convert to cv2 format for better drawing
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        boxes = predictions['boxes']
        masks = predictions['masks']
        scores = predictions['scores']
        
        # Colors for different detections
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (box, mask, score) in enumerate(zip(boxes, masks, scores)):
            color = colors[i % len(colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image_cv2, (x1, y1), (x2, y2), color, box_thickness)
            
            # Add score text with background
            text = f'Damage: {score:.2f}'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image_cv2, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0] + 5, y1), color, -1)
            cv2.putText(image_cv2, text, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw mask overlay
            if len(mask.shape) == 2:
                mask_binary = (mask > 0.5).astype(np.uint8)
                
                # Create colored mask overlay
                colored_mask = np.zeros_like(image_cv2)
                colored_mask[mask_binary == 1] = color
                
                # Blend with original image
                image_cv2 = cv2.addWeighted(image_cv2, 1.0, colored_mask, mask_alpha, 0)
        
        # Convert back to PIL
        result_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_image)


def predict_damage(image_path, model_path='models/best_model.pth', confidence_threshold=0.1):
    """
    Simple function to predict damage on an image
    
    Args:
        image_path: Path to image or PIL Image
        model_path: Path to trained model
        confidence_threshold: Confidence threshold for detections
        
    Returns:
        tuple: (predictions dict, visualized image)
    """
    try:
        predictor = CarDamagePredictor(model_path, confidence_threshold)
        predictions = predictor.predict(image_path, debug=True)  # Enable debug by default
        visualized = predictor.visualize_predictions(image_path, predictions)
        
        return predictions, visualized
    except Exception as e:
        print(f"Error in predict_damage: {e}")
        # Return empty results on error
        empty_predictions = {
            'boxes': np.array([]),
            'masks': np.array([]),
            'scores': np.array([]),
            'labels': np.array([]),
            'num_detections': 0,
            'all_scores': np.array([]),
            'all_boxes': np.array([])
        }
        
        # Return original image on error
        if isinstance(image_path, str):
            original_img = Image.open(image_path)
        else:
            original_img = image_path
            
        return empty_predictions, original_img