"""
Simple Explainable AI for Car Damage Detection
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class SimpleGradCAM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        
        # Hook into the backbone's last convolutional layer
        self.hook_layers()
    
    def hook_layers(self):
        """Hook into the model to capture gradients and activations"""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # Hook into ResNet backbone's last layer
        target_layer = self.model.model.backbone.body.layer4[-1].conv3
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx=1):
        """
        Generate Class Activation Map
        
        Args:
            input_tensor: Preprocessed image tensor [3, H, W] (3D tensor)
            class_idx: Class index (1 for damage)
            
        Returns:
            numpy array: Heatmap of same size as input image
        """
        self.model.eval()
        
        # Ensure input_tensor is 3D and on correct device
        if len(input_tensor.shape) != 3:
            print(f"Warning: Expected 3D tensor, got {input_tensor.shape}")
            return np.zeros((512, 512))
        
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()
        
        try:
            # Get predictions - pass 3D tensor in a list (correct format for Mask R-CNN)
            outputs = self.model([input_tensor])  # input_tensor is [3, H, W]
        except Exception as e:
            print(f"Model forward pass failed: {e}")
            return np.zeros((input_tensor.shape[1], input_tensor.shape[2]))
        
        if len(outputs[0]['scores']) == 0:
            # No detections, return empty heatmap
            print("No detections found for Grad-CAM")
            return np.zeros((input_tensor.shape[1], input_tensor.shape[2]))
        
        # Use the highest scoring detection
        max_score_idx = torch.argmax(outputs[0]['scores'])
        target_score = outputs[0]['scores'][max_score_idx]
        
        # Backward pass
        self.model.zero_grad()
        try:
            target_score.backward()
        except Exception as e:
            print(f"Backward pass failed: {e}")
            return np.zeros((input_tensor.shape[1], input_tensor.shape[2]))
        
        if self.gradients is None or self.activations is None:
            print("No gradients or activations captured")
            return np.zeros((input_tensor.shape[1], input_tensor.shape[2]))
        
        # Generate CAM
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        
        # Global average pooling on gradients
        weights = np.mean(gradients[0], axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations[0][0].shape, dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[0][i]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input image size
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[1]))
        
        return cam


class SimpleExplainer:
    def __init__(self, predictor):
        self.predictor = predictor
        self.gradcam = SimpleGradCAM(predictor.model, predictor.device)
    
    def generate_gradcam_heatmap(self, image, alpha=0.4):
        """
        Generate Grad-CAM heatmap overlay
        
        Args:
            image: PIL Image or numpy array
            alpha: Transparency of heatmap overlay
            
        Returns:
            PIL Image: Original image with heatmap overlay
        """
        # Preprocess image
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        try:
            # Get preprocessed tensor (should be 3D: [3, 512, 512])
            image_tensor, _ = self.predictor.preprocess_image(image)
            
            # Generate CAM - pass 3D tensor directly (no unsqueeze!)
            cam = self.gradcam.generate_cam(image_tensor)
            
            if cam.max() == 0:
                # No heatmap generated, return original image
                print("Warning: No Grad-CAM heatmap generated")
                return Image.fromarray(image_np)
            
            # Create heatmap
            heatmap = cm.jet(cam)[:, :, :3]  # Remove alpha channel
            heatmap = (heatmap * 255).astype(np.uint8)
            
            # Resize heatmap to match original image
            heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
            
            # Overlay heatmap on original image
            overlaid = cv2.addWeighted(image_np, 1-alpha, heatmap, alpha, 0)
            
            return Image.fromarray(overlaid)
            
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            return Image.fromarray(image_np)
    
    def occlusion_analysis(self, image, patch_size=50, stride=25):
        """
        Simple occlusion sensitivity analysis
        
        Args:
            image: PIL Image
            patch_size: Size of occlusion patches
            stride: Stride between patches
            
        Returns:
            tuple: (occlusion_map, original_score)
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        try:
            # Get baseline prediction
            baseline_pred = self.predictor.predict(image)
            original_score = baseline_pred['scores'].max() if len(baseline_pred['scores']) > 0 else 0.0
            
            # Initialize occlusion map
            h, w = image_np.shape[:2]
            occlusion_map = np.zeros((h, w), dtype=np.float32)
            
            print(f"Running occlusion analysis with {patch_size}x{patch_size} patches...")
            
            # Calculate total patches for progress
            total_patches = ((h - patch_size) // stride + 1) * ((w - patch_size) // stride + 1)
            processed_patches = 0
            
            # Slide occlusion patches
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    try:
                        # Create occluded image
                        occluded_img = image_np.copy()
                        occluded_img[y:y+patch_size, x:x+patch_size] = 128  # Gray patch
                        
                        # Get prediction for occluded image
                        occluded_pred = self.predictor.predict(Image.fromarray(occluded_img))
                        occluded_score = occluded_pred['scores'].max() if len(occluded_pred['scores']) > 0 else 0.0
                        
                        # Calculate importance (drop in confidence)
                        importance = max(0, original_score - occluded_score)
                        
                        # Assign importance to patch region
                        occlusion_map[y:y+patch_size, x:x+patch_size] = max(
                            occlusion_map[y:y+patch_size, x:x+patch_size].mean(), 
                            importance
                        )
                        
                        processed_patches += 1
                        if processed_patches % 10 == 0:
                            print(f"Progress: {processed_patches}/{total_patches} patches")
                            
                    except Exception as e:
                        print(f"Error processing patch at ({x}, {y}): {e}")
                        continue
            
            # Normalize occlusion map
            if occlusion_map.max() > 0:
                occlusion_map = occlusion_map / occlusion_map.max()
            
            print(f"Occlusion analysis completed")
            return occlusion_map, original_score
            
        except Exception as e:
            print(f"Error in occlusion analysis: {e}")
            # Return empty map on error
            return np.zeros((image_np.shape[0], image_np.shape[1])), 0.0
    
    def create_occlusion_visualization(self, image, occlusion_map, alpha=0.5):
        """
        Create occlusion sensitivity visualization
        
        Args:
            image: Original PIL Image
            occlusion_map: Occlusion sensitivity map
            alpha: Transparency of overlay
            
        Returns:
            PIL Image: Visualization
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        # Create heatmap from occlusion map
        heatmap = cm.hot(occlusion_map)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Overlay on original image
        overlaid = cv2.addWeighted(image_np, 1-alpha, heatmap, alpha, 0)
        
        return Image.fromarray(overlaid)
    
    def explain_prediction(self, image, method='gradcam', **kwargs):
        """
        Unified explanation interface
        
        Args:
            image: PIL Image
            method: 'gradcam' or 'occlusion'
            **kwargs: Method-specific parameters
            
        Returns:
            dict: Explanation results
        """
        results = {'method': method, 'original_image': image, 'error': False}
        
        try:
            if method == 'gradcam':
                alpha = kwargs.get('alpha', 0.4)
                heatmap_image = self.generate_gradcam_heatmap(image, alpha=alpha)
                results['explanation_image'] = heatmap_image
                results['description'] = "Heatmap shows areas the model focuses on for damage detection"
                
            elif method == 'occlusion':
                patch_size = kwargs.get('patch_size', 50)
                stride = kwargs.get('stride', 25)
                alpha = kwargs.get('alpha', 0.5)
                
                occlusion_map, original_score = self.occlusion_analysis(
                    image, patch_size=patch_size, stride=stride
                )
                explanation_image = self.create_occlusion_visualization(
                    image, occlusion_map, alpha=alpha
                )
                
                results['explanation_image'] = explanation_image
                results['occlusion_map'] = occlusion_map
                results['original_score'] = original_score
                results['description'] = "Heatmap shows how much each region affects the prediction"
                
            else:
                results['explanation_image'] = image
                results['description'] = f"Unknown method: {method}"
                results['error'] = True
        
        except Exception as e:
            print(f"Error in explain_prediction: {e}")
            results['explanation_image'] = image
            results['description'] = f"Explanation failed: {str(e)}"
            results['error'] = True
        
        return results


def create_simple_explanation(predictor, image, method='gradcam', **kwargs):
    """
    Simple function to create explanations
    
    Args:
        predictor: CarDamagePredictor instance
        image: PIL Image
        method: 'gradcam' or 'occlusion'
        **kwargs: Method-specific parameters
        
    Returns:
        dict: Explanation results with visualization
    """
    try:
        explainer = SimpleExplainer(predictor)
        return explainer.explain_prediction(image, method=method, **kwargs)
    except Exception as e:
        print(f"Error in create_simple_explanation: {e}")
        # Return a safe fallback result
        if isinstance(image, str):
            fallback_image = Image.open(image)
        else:
            fallback_image = image
            
        return {
            'method': method,
            'original_image': fallback_image,
            'explanation_image': fallback_image,
            'description': f"Explanation failed: {str(e)}",
            'error': True
        }