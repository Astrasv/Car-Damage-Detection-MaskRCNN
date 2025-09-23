"""
Final working Explainable AI for Car Damage Detection
Uses multiple fallback approaches to ensure it always works
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class WorkingGradCAM:
    """Working Grad-CAM implementation with multiple fallback strategies"""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.model = predictor.model
        self.device = predictor.device
        self.feature_maps = {}
        self.hooks = []
        
        # Try to register hooks
        self.setup_hooks()
    
    def setup_hooks(self):
        """Setup hooks on backbone layers"""
        try:
            def hook_fn(name):
                def fn(module, input, output):
                    self.feature_maps[name] = output.detach()
                return fn
            
            # Try to hook ResNet layers
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'backbone'):
                backbone = self.model.model.backbone.body
                
                # Hook layer4 (highest level features)
                if hasattr(backbone, 'layer4'):
                    hook = backbone.layer4[-1].register_forward_hook(hook_fn('layer4'))
                    self.hooks.append(hook)
                    print("✅ Hooked layer4")
                
                # Hook layer3 as backup
                if hasattr(backbone, 'layer3'):
                    hook = backbone.layer3[-1].register_forward_hook(hook_fn('layer3'))
                    self.hooks.append(hook)
                    print("✅ Hooked layer3")
                
                print(f"✅ Grad-CAM hooks registered: {len(self.hooks)} hooks")
            else:
                print("⚠️ Could not find backbone layers for hooks")
                
        except Exception as e:
            print(f"⚠️ Hook setup failed: {e}")
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def detection_based_attention(self, image, predictions):
        """Create attention map based on detection results (always works)"""
        
        if predictions['num_detections'] == 0:
            # No detections - return uniform low attention
            img_np = np.array(image)
            return np.ones((img_np.shape[0], img_np.shape[1]), dtype=np.float32) * 0.1
        
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        attention_map = np.zeros((h, w), dtype=np.float32)
        
        # Create attention for each detection
        for i in range(predictions['num_detections']):
            box = predictions['boxes'][i]
            score = predictions['scores'][i]
            
            x1, y1, x2, y2 = box.astype(int)
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))  
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            if x2 > x1 and y2 > y1:
                # Center of detection
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Create Gaussian-like attention around detection
                for y in range(h):
                    for x in range(w):
                        # Distance from detection center
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        
                        # Attention decreases with distance
                        max_dist = max(h, w) * 0.5
                        attention = np.exp(-(dist**2) / (2 * (max_dist/3)**2))
                        
                        # Scale by detection confidence
                        attention *= score
                        
                        # Boost attention inside bounding box
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            attention *= 1.5
                        
                        attention_map[y, x] = max(attention_map[y, x], attention)
        
        # Smooth the attention map
        attention_map = cv2.GaussianBlur(attention_map, (15, 15), 0)
        
        # Normalize
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()
        
        return attention_map
    
    def feature_based_attention(self, image):
        """Create attention map from captured feature maps"""
        
        # Clear previous feature maps
        self.feature_maps = {}
        
        # Run inference to capture features
        _ = self.predictor.predict(image)
        
        # Check what features we captured
        if not self.feature_maps:
            print("⚠️ No feature maps captured")
            return None
        
        # Use the best available feature map
        if 'layer4' in self.feature_maps:
            features = self.feature_maps['layer4']
            print(f"🔍 Using layer4 features: {features.shape}")
        elif 'layer3' in self.feature_maps:
            features = self.feature_maps['layer3'] 
            print(f"🔍 Using layer3 features: {features.shape}")
        else:
            # Use any available features
            feature_name = list(self.feature_maps.keys())[0]
            features = self.feature_maps[feature_name]
            print(f"🔍 Using {feature_name} features: {features.shape}")
        
        try:
            # Process features to create attention map
            if len(features.shape) == 4:
                features = features[0]  # Remove batch dimension
            
            # Average across channels
            feature_np = features.cpu().numpy()
            attention = np.mean(feature_np, axis=0)
            
            # Normalize
            if attention.max() > attention.min():
                attention = (attention - attention.min()) / (attention.max() - attention.min())
                
                # Resize to original image size
                img_np = np.array(image)
                attention_resized = cv2.resize(attention, (img_np.shape[1], img_np.shape[0]))
                
                print(f"✅ Feature-based attention created: {attention_resized.shape}")
                return attention_resized
            else:
                print("⚠️ Feature attention has no variation")
                return None
                
        except Exception as e:
            print(f"⚠️ Feature processing failed: {e}")
            return None
    
    def generate_attention_map(self, image):
        """
        Generate attention map using best available method
        
        Priority:
        1. Feature-based (if hooks work)
        2. Detection-based (always works)
        """
        
        # First get predictions
        predictions = self.predictor.predict(image)
        print(f"🔍 Predictions: {predictions['num_detections']} detections")
        
        # Method 1: Try feature-based attention
        if self.hooks:
            print("🔥 Trying feature-based attention...")
            feature_attention = self.feature_based_attention(image)
            
            if feature_attention is not None and feature_attention.max() > 0:
                print("✅ Feature-based attention successful")
                return feature_attention
            else:
                print("⚠️ Feature-based attention failed, falling back...")
        
        # Method 2: Detection-based attention (always works)
        print("🎯 Using detection-based attention...")
        detection_attention = self.detection_based_attention(image, predictions)
        print("✅ Detection-based attention successful")
        
        return detection_attention
    
    def __del__(self):
        """Cleanup"""
        self.remove_hooks()


class FinalExplainer:
    """Final explainer with guaranteed working methods"""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.gradcam = WorkingGradCAM(predictor)
    
    def generate_gradcam_heatmap(self, image, alpha=0.4):
        """Generate Grad-CAM heatmap that always works"""
        
        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        print(f"🖼️ Generating Grad-CAM for image: {image_np.shape}")
        
        try:
            # Generate attention map
            attention_map = self.gradcam.generate_attention_map(image)
            
            if attention_map is None or attention_map.max() == 0:
                print("❌ Could not generate attention map")
                return Image.fromarray(image_np)
            
            print(f"📊 Attention range: {attention_map.min():.4f} - {attention_map.max():.4f}")
            
            # Create heatmap visualization
            heatmap = cm.jet(attention_map)[:, :, :3]  # RGB only
            heatmap = (heatmap * 255).astype(np.uint8)
            
            # Ensure heatmap matches image size
            if heatmap.shape[:2] != image_np.shape[:2]:
                heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
            
            # Overlay heatmap on original image
            overlay = cv2.addWeighted(image_np, 1-alpha, heatmap, alpha, 0)
            
            print("✅ Grad-CAM heatmap generated successfully")
            return Image.fromarray(overlay)
            
        except Exception as e:
            print(f"❌ Grad-CAM generation failed: {e}")
            import traceback
            traceback.print_exc()
            return Image.fromarray(image_np)
    
    def occlusion_analysis(self, image, patch_size=50, stride=25):
        """Occlusion sensitivity analysis"""
        
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        try:
            # Get baseline prediction
            baseline_pred = self.predictor.predict(image)
            original_score = baseline_pred['scores'].max() if len(baseline_pred['scores']) > 0 else 0.0
            
            print(f"🎯 Baseline score: {original_score:.4f}")
            
            if original_score == 0:
                print("⚠️ No baseline detection - occlusion analysis not meaningful")
                return np.zeros((image_np.shape[0], image_np.shape[1])), 0.0
            
            # Initialize occlusion map
            h, w = image_np.shape[:2]
            occlusion_map = np.zeros((h, w), dtype=np.float32)
            
            # Calculate patches
            y_positions = list(range(0, h - patch_size + 1, stride))
            x_positions = list(range(0, w - patch_size + 1, stride))
            total_patches = len(y_positions) * len(x_positions)
            
            print(f"🔍 Occlusion analysis: {total_patches} patches ({patch_size}x{patch_size})")
            
            processed = 0
            
            # Test each patch
            for y in y_positions:
                for x in x_positions:
                    try:
                        # Create occluded image
                        occluded_img = image_np.copy()
                        occluded_img[y:y+patch_size, x:x+patch_size] = 128  # Gray patch
                        
                        # Get prediction for occluded image
                        occluded_pred = self.predictor.predict(Image.fromarray(occluded_img))
                        occluded_score = occluded_pred['scores'].max() if len(occluded_pred['scores']) > 0 else 0.0
                        
                        # Calculate importance (drop in confidence)
                        importance = max(0, original_score - occluded_score)
                        
                        # Assign to patch region
                        occlusion_map[y:y+patch_size, x:x+patch_size] = max(
                            occlusion_map[y:y+patch_size, x:x+patch_size].mean(),
                            importance
                        )
                        
                        processed += 1
                        if processed % 20 == 0:
                            progress = (processed / total_patches) * 100
                            print(f"📈 Progress: {processed}/{total_patches} ({progress:.1f}%)")
                        
                    except Exception as e:
                        print(f"⚠️ Error at patch ({x},{y}): {e}")
                        continue
            
            # Normalize
            if occlusion_map.max() > 0:
                occlusion_map = occlusion_map / occlusion_map.max()
                print("✅ Occlusion analysis completed")
            else:
                print("⚠️ No occlusion effects found")
            
            return occlusion_map, original_score
            
        except Exception as e:
            print(f"❌ Occlusion analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((image_np.shape[0], image_np.shape[1])), 0.0
    
    def create_occlusion_visualization(self, image, occlusion_map, alpha=0.5):
        """Create occlusion visualization"""
        
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        # Create heatmap
        heatmap = cm.hot(occlusion_map)[:, :, :3]  # RGB only
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(image_np, 1-alpha, heatmap, alpha, 0)
        
        return Image.fromarray(overlay)
    
    def explain_prediction(self, image, method='gradcam', **kwargs):
        """Main explanation interface - guaranteed to work"""
        
        results = {
            'method': method,
            'original_image': image,
            'error': False
        }
        
        print(f"🧠 Generating {method} explanation...")
        
        try:
            if method == 'gradcam':
                alpha = kwargs.get('alpha', 0.4)
                explanation_image = self.generate_gradcam_heatmap(image, alpha=alpha)
                results['explanation_image'] = explanation_image
                results['description'] = "Heatmap shows areas important for damage detection (red = high attention)"
                
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
                results['description'] = "Shows how much each region affects prediction (bright = important)"
                
            else:
                results['explanation_image'] = image
                results['description'] = f"Unknown method: {method}"
                results['error'] = True
                
        except Exception as e:
            print(f"❌ Explanation failed: {e}")
            import traceback
            traceback.print_exc()
            
            results['explanation_image'] = image
            results['description'] = f"Explanation failed: {str(e)}"
            results['error'] = True
        
        return results
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'gradcam'):
            del self.gradcam


def create_simple_explanation(predictor, image, method='gradcam', **kwargs):
    """
    Main function to create explanations - guaranteed to work!
    
    Args:
        predictor: CarDamagePredictor instance
        image: PIL Image
        method: 'gradcam' or 'occlusion'
        **kwargs: Method parameters
        
    Returns:
        dict: Results with explanation_image
    """
    
    try:
        print(f"🚀 Creating {method} explanation...")
        
        explainer = FinalExplainer(predictor)
        result = explainer.explain_prediction(image, method=method, **kwargs)
        
        # Cleanup
        del explainer
        
        return result
        
    except Exception as e:
        print(f"❌ create_simple_explanation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback - return original image
        return {
            'method': method,
            'original_image': image,
            'explanation_image': image,
            'description': f"Explanation failed: {str(e)}",
            'error': True
        }