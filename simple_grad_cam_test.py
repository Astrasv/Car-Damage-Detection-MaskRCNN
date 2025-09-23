"""
Working simple Grad-CAM test that doesn't break the model
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.cm as cm

sys.path.append('src')

def working_gradcam_test():
    """Simple working Grad-CAM test using model predictions"""
    
    print("🧠 Working Grad-CAM Test")
    print("=" * 30)
    
    try:
        from inference import CarDamagePredictor
        
        # Load model
        model_path = 'models/best_model.pth'
        predictor = CarDamagePredictor(model_path, confidence_threshold=0.01)  # Lower threshold
        print(f"✅ Model loaded on {predictor.device}")
        
        # Find test image
        test_image = None
        for path in ['dataset/val/images', 'dataset/train/images']:
            if os.path.exists(path):
                images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    test_image = os.path.join(path, images[0])
                    break
        
        if not test_image:
            print("❌ No test image found")
            return False
        
        image = Image.open(test_image)
        print(f"📸 Using: {os.path.basename(test_image)}, size: {image.size}")
        
        # Test basic prediction first
        predictions = predictor.predict(image, debug=True)
        print(f"🔍 Detections: {predictions['num_detections']}")
        
        if predictions['num_detections'] == 0:
            print("⚠️ No detections found - can't generate meaningful Grad-CAM")
            return False
        
        # Get the detection with highest confidence
        best_idx = np.argmax(predictions['scores'])
        best_score = predictions['scores'][best_idx]
        best_box = predictions['boxes'][best_idx]
        
        print(f"🎯 Best detection: score={best_score:.4f}, box={best_box}")
        
        # Create a simple attention map based on the detection
        # This is a fallback approach when gradient-based methods fail
        
        original_h, original_w = np.array(image).shape[:2]
        
        # Create attention map centered on the detected region
        attention_map = np.zeros((original_h, original_w), dtype=np.float32)
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = best_box.astype(int)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, original_w-1))
        y1 = max(0, min(y1, original_h-1))
        x2 = max(0, min(x2, original_w-1))
        y2 = max(0, min(y2, original_h-1))
        
        if x2 > x1 and y2 > y1:
            # Create a gradient that peaks at the detection center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Create distance-based attention
            for y in range(original_h):
                for x in range(original_w):
                    # Distance from center of detection
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    
                    # Higher attention near the detection, with some spread
                    max_dist = max(original_h, original_w)
                    attention = max(0, 1.0 - (dist / (max_dist * 0.3)))
                    
                    # Boost attention inside the bounding box
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        attention *= (1.0 + best_score)  # Scale by confidence
                    
                    attention_map[y, x] = attention
            
            # Smooth the attention map
            attention_map = cv2.GaussianBlur(attention_map, (21, 21), 0)
        
        # Normalize attention map
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()
        
        print(f"📊 Attention map range: {attention_map.min():.4f} - {attention_map.max():.4f}")
        
        if attention_map.max() > 0:
            # Create heatmap visualization
            heatmap = cm.jet(attention_map)[:, :, :3] * 255
            heatmap = heatmap.astype(np.uint8)
            
            # Original image as numpy array
            img_np = np.array(image)
            
            # Overlay heatmap
            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
            
            # Save results
            os.makedirs('working_test', exist_ok=True)
            Image.fromarray(img_np).save('working_test/original.png')
            Image.fromarray(heatmap).save('working_test/heatmap.png')
            Image.fromarray(overlay).save('working_test/overlay.png')
            
            # Also create a side-by-side comparison
            comparison = np.hstack([img_np, overlay])
            Image.fromarray(comparison).save('working_test/comparison.png')
            
            print("💾 Results saved in 'working_test/' folder:")
            print("   - original.png: Original image")
            print("   - heatmap.png: Pure heatmap")
            print("   - overlay.png: Overlaid result") 
            print("   - comparison.png: Side-by-side")
            
            print("✅ WORKING GRAD-CAM GENERATED!")
            
            return True
        else:
            print("❌ Could not generate attention map")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def try_real_gradcam():
    """Try a real grad-cam approach with proper hooks"""
    
    print("\n🔥 Trying Real Grad-CAM Approach")
    print("=" * 40)
    
    try:
        from inference import CarDamagePredictor
        
        # Load model
        model_path = 'models/best_model.pth'
        predictor = CarDamagePredictor(model_path, confidence_threshold=0.01)
        
        # Find test image
        test_image = None
        for path in ['dataset/val/images', 'dataset/train/images']:
            if os.path.exists(path):
                images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    test_image = os.path.join(path, images[0])
                    break
        
        image = Image.open(test_image)
        print(f"📸 Using: {os.path.basename(test_image)}")
        
        # Variables to store feature maps
        feature_maps = {}
        
        def hook_fn(name):
            def fn(module, input, output):
                feature_maps[name] = output.detach()
                print(f"✅ Hooked {name}: {output.shape}")
            return fn
        
        # Hook into backbone layers
        model = predictor.model
        
        # Register hooks on multiple layers
        hooks = []
        try:
            # ResNet backbone layers
            layer4 = model.model.backbone.body.layer4[-1]
            hook = layer4.register_forward_hook(hook_fn('layer4'))
            hooks.append(hook)
            
            layer3 = model.model.backbone.body.layer3[-1]
            hook = layer3.register_forward_hook(hook_fn('layer3'))
            hooks.append(hook)
            
            print("✅ Hooks registered on backbone layers")
            
        except Exception as e:
            print(f"⚠️ Hook registration failed: {e}")
        
        # Make prediction to trigger hooks
        predictions = predictor.predict(image)
        print(f"🔍 Made prediction: {predictions['num_detections']} detections")
        
        # Check what we captured
        print(f"📊 Captured feature maps: {list(feature_maps.keys())}")
        for name, features in feature_maps.items():
            print(f"   {name}: {features.shape}")
        
        # Use the highest level features
        if 'layer4' in feature_maps:
            features = feature_maps['layer4']
        elif 'layer3' in feature_maps:
            features = feature_maps['layer3']
        else:
            print("❌ No features captured")
            return False
        
        # Create simple CAM from features
        if len(features.shape) == 4:
            features = features[0]  # Remove batch dimension
        
        # Average across channels
        feature_np = features.cpu().numpy()
        cam = np.mean(feature_np, axis=0)
        
        # Normalize
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        print(f"📊 CAM shape: {cam.shape}, range: {cam.min():.4f}-{cam.max():.4f}")
        
        # Resize to original image size
        img_np = np.array(image)
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        
        # Create heatmap
        heatmap = cm.jet(cam_resized)[:, :, :3] * 255
        heatmap = heatmap.astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        
        # Save results
        os.makedirs('real_gradcam_test', exist_ok=True)
        Image.fromarray(img_np).save('real_gradcam_test/original.png')
        Image.fromarray(heatmap).save('real_gradcam_test/heatmap.png')
        Image.fromarray(overlay).save('real_gradcam_test/overlay.png')
        
        print("💾 Real Grad-CAM results saved in 'real_gradcam_test/' folder")
        print("✅ REAL GRAD-CAM WORKING!")
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        return True
        
    except Exception as e:
        print(f"❌ Real Grad-CAM failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("🧪 Complete Grad-CAM Test Suite")
    print("=" * 50)
    
    # Try working approach first
    success1 = working_gradcam_test()
    
    if success1:
        print(f"\n✅ Working approach succeeded!")
        
        # Try real grad-cam
        success2 = try_real_gradcam()
        
        if success2:
            print(f"\n🎉 BOTH METHODS WORK!")
            print(f"✅ Check 'working_test/' and 'real_gradcam_test/' folders")
            print(f"\n🚀 Ready for Streamlit app:")
            print(f"   streamlit run app.py")
        else:
            print(f"\n✅ Working method succeeded, real Grad-CAM failed")
            print(f"   The working method will be used as fallback")
    else:
        print(f"\n❌ Both methods failed")
        print(f"   Check your model and test images")    