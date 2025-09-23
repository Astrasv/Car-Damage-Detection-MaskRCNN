import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class CarDamageMaskRCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(CarDamageMaskRCNN, self).__init__()
        
        # Load pre-trained Mask R-CNN model
        self.model = maskrcnn_resnet50_fpn(pretrained=pretrained)
        
        # Get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Get number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        
        # Replace the mask predictor
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
    
    def forward(self, images, targets=None):
        return self.model(images, targets)


def get_model(num_classes=2, pretrained=True):
    """
    Create and return the Mask R-CNN model
    
    Args:
        num_classes (int): Number of classes (background + damage)
        pretrained (bool): Whether to use pre-trained weights
    
    Returns:
        CarDamageMaskRCNN: The model instance
    """
    model = CarDamageMaskRCNN(num_classes=num_classes, pretrained=pretrained)
    return model


def load_model(model_path, num_classes=2):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path (str): Path to the model checkpoint
        num_classes (int): Number of classes
    
    Returns:
        CarDamageMaskRCNN: Loaded model
    """
    model = get_model(num_classes=num_classes, pretrained=False)
    
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model