"""
Test script to verify the fixed Grad-CAM implementation
"""

import sys
import os
from PIL import Image
import numpy as np

sys.path.append('src')

def test_fixed_gradcam():
    """Test the fixed Grad-CAM implementation"""
    
    print("🧠 Testing Fixed Grad-CAM Implementation")
    print("=" * 50)
    
    # Import modules
    try:
        from inference import CarDamagePredictor
        from explainable_ai import create_simple_explanation
        print("✅ Modules imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Check model
    model_path = 'models/best_model.pth'
    if not os.path.exists(model_path):
        print("❌ Model not found. Train first!")
        return False
    
    # Find test image
    test_image = None
    for path in ['dataset/val/images', 'dataset/train/images']:
        if os.path.exists(path):
            images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_image = os.path.join(path, images[0])
                break
    
    if not test_image:
        print("❌ No test images found")
        return False
    
    print(f"📸 Using test image: {test_image}")
    
    try:
        # Load predictor
        print("\n🤖 Loading predictor...")
        predictor = CarDamagePredictor(model_path, confidence_threshold=0.1)
        print(f"✅ Predictor loaded on {predictor.device}")
        
        # Load and test image
        print("\n🖼️ Loading image...")
        image = Image.open(test_image)
        print(f"✅ Image loaded: {image.size}")
        
        # Make prediction first
        print("\n🔍 Making prediction...")
        predictions = predictor.predict(image, debug=True)
        print(f"✅ Predictions: {predictions['num_detections']} detections")
        
        if predictions['num_detections'] == 0:
            print("⚠️ No detections found. Grad-CAM might not work well, but let's try...")
        
        # Test Grad-CAM with detailed output
        print("\n🔥 Testing Grad-CAM with detailed output...")
        print("-" * 30)
        
        gradcam_result = create_simple_explanation(
            predictor, image, method='gradcam', alpha=0.4
        )
        
        print("-" * 30)
        
        if gradcam_result['error']:
            print(f"❌ Grad-CAM failed: {gradcam_result['description']}")
            return False
        
        print("✅ Grad-CAM completed!")
        
        # Check if heatmap was actually generated
        original_array = np.array(image)
        gradcam_array = np.array(gradcam_result['explanation_image'])
        
        # Compare arrays to see if they're different
        if np.array_equal(original_array, gradcam_array):
            print("❌ Grad-CAM output is identical to input (no heatmap generated)")
            return False
        else:
            print("✅ Grad-CAM generated a different image (heatmap added!)")
            
            # Calculate difference statistics
            diff = np.abs(original_array.astype(float) - gradcam_array.astype(float))
            mean_diff = diff.mean()
            max_diff = diff.max()
            
            print(f"📊 Image difference stats:")
            print(f"   Mean difference: {mean_diff:.2f}")
            print(f"   Max difference: {max_diff:.2f}")
            
            if mean_diff < 1.0:
                print("⚠️ Very small differences - heatmap might be too subtle")
            else:
                print("✅ Good difference levels - heatmap is visible!")
        
        # Save results
        os.makedirs('gradcam_test', exist_ok=True)
        image.save('gradcam_test/original.png')
        gradcam_result['explanation_image'].save('gradcam_test/gradcam_result.png')
        
        print(f"\n💾 Results saved:")
        print(f"   Original: gradcam_test/original.png")
        print(f"   Grad-CAM: gradcam_test/gradcam_result.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_images_visually():
    """Display comparison if results exist"""
    
    import matplotlib.pyplot as plt
    
    if not os.path.exists('gradcam_test/original.png'):
        print("❌ No test results found. Run test first.")
        return
    
    try:
        original = Image.open('gradcam_test/original.png')
        gradcam = Image.open('gradcam_test/gradcam_result.png')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(original)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(gradcam)
        ax2.set_title('Grad-CAM Result')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('gradcam_test/comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visual comparison saved: gradcam_test/comparison.png")
        
    except Exception as e:
        print(f"❌ Visual comparison failed: {e}")


if __name__ == '__main__':
    print("🧪 Grad-CAM Fix Test")
    print("=" * 50)
    
    success = test_fixed_gradcam()
    
    if success:
        print(f"\n🎉 Test PASSED! Grad-CAM is working!")
        print(f"\n🚀 Now try the Streamlit app:")
        print(f"   streamlit run app.py")
        print(f"\n📊 Visual comparison:")
        compare_images_visually()
    else:
        print(f"\n❌ Test FAILED! Check the detailed output above.")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Ensure model is properly trained")
        print(f"   2. Try with images that have clear damage")
        print(f"   3. Check that PyTorch version supports hooks")
        print(f"   4. Verify GPU/CUDA setup if using GPU")