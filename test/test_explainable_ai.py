"""
Fixed test for explainable AI with better error handling
"""

import sys
import os
from PIL import Image
import traceback

sys.path.append('src')

def test_gradcam_only():
    """Test only Grad-CAM functionality with robust error handling"""
    
    print("🧠 Testing Grad-CAM Explainable AI")
    print("=" * 40)
    
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
        print("❌ Model not found. Train first with: python train_robust.py")
        return False
    
    # Find test images
    test_images = []
    for path in ['dataset/val/images', 'dataset/train/images']:
        if os.path.exists(path):
            images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            test_images.extend([os.path.join(path, img) for img in images[:2]])
            break
    
    if not test_images:
        print("❌ No test images found")
        return False
    
    print(f"✅ Found {len(test_images)} test images")
    
    # Create output directory
    output_dir = 'xai_test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load predictor
        predictor = CarDamagePredictor(model_path, confidence_threshold=0.1)
        print(f"✅ Model loaded on {predictor.device}")
        
        success_count = 0
        
        for i, img_path in enumerate(test_images):
            print(f"\n📸 Testing image {i+1}: {os.path.basename(img_path)}")
            print("-" * 30)
            
            try:
                # Load image
                image = Image.open(img_path)
                print(f"   Image size: {image.size}")
                
                # Make prediction
                predictions = predictor.predict(image)
                print(f"   Detections: {predictions['num_detections']}")
                
                # Test Grad-CAM
                print("   🔥 Testing Grad-CAM...")
                gradcam_result = create_simple_explanation(
                    predictor, image, method='gradcam', alpha=0.4
                )
                print("   ✅ Grad-CAM explanation generated successfully!")
                
                # Save results
                image.save(f'{output_dir}/image_{i+1}_original.png')
                
                pred_viz = predictor.visualize_predictions(image, predictions)
                pred_viz.save(f'{output_dir}/image_{i+1}_prediction.png')
                
                gradcam_result['explanation_image'].save(f'{output_dir}/image_{i+1}_gradcam.png')
                
                print(f"   💾 Results saved successfully")
                success_count += 1
                
            except Exception as e:
                print(f"   ❌ Error processing image: {e}")
                print(f"   🔍 Detailed error:")
                traceback.print_exc()
                continue
        
        print(f"\n{'='*40}")
        print(f"🎉 Test completed!")
        print(f"📊 Successfully processed: {success_count}/{len(test_images)} images")
        print(f"📁 Results saved in: {output_dir}/")
        
        if success_count > 0:
            print(f"\n🚀 Next steps:")
            print(f"   1. Check results in '{output_dir}/'")
            print(f"   2. Run: streamlit run app.py")
            print(f"   3. Test both Detection and Explainable AI tabs")
            return True
        else:
            return False
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        print(f"🔍 Detailed error:")
        traceback.print_exc()
        return False


def test_basic_model():
    """Test if model and predictor work correctly"""
    
    print("\n🔧 Testing Basic Model Functionality")
    print("-" * 40)
    
    try:
        from inference import CarDamagePredictor
        
        model_path = 'models/best_model.pth'
        predictor = CarDamagePredictor(model_path, confidence_threshold=0.1)
        
        print("✅ Predictor created successfully")
        print(f"   Device: {predictor.device}")
        print(f"   Confidence threshold: {predictor.confidence_threshold}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        print(f"🔍 Detailed error:")
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("🧪 XAI Fixed Test Suite")
    print("=" * 50)
    
    # Test basic model first
    if not test_basic_model():
        print("\n❌ Basic model test failed. Please fix model issues first.")
        sys.exit(1)
    
    # Test Grad-CAM
    if test_gradcam_only():
        print(f"\n✅ All tests passed!")
        
        # Ask about occlusion test
        test_occlusion = input("\n🎯 Test occlusion analysis too? (slower, y/n): ").lower() == 'y'
        
        if test_occlusion:
            print("\n🎯 Testing Occlusion Analysis...")
            try:
                from inference import CarDamagePredictor
                from explainable_ai import create_simple_explanation
                
                predictor = CarDamagePredictor('models/best_model.pth', confidence_threshold=0.1)
                
                # Find one test image
                for path in ['dataset/val/images', 'dataset/train/images']:
                    if os.path.exists(path):
                        images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if images:
                            test_img = os.path.join(path, images[0])
                            break
                
                image = Image.open(test_img)
                print(f"Testing occlusion on: {os.path.basename(test_img)}")
                
                occlusion_result = create_simple_explanation(
                    predictor, image, method='occlusion', patch_size=40, stride=25
                )
                
                occlusion_result['explanation_image'].save('xai_test_results/occlusion_test.png')
                print("✅ Occlusion analysis completed!")
                
            except Exception as e:
                print(f"⚠️ Occlusion test failed: {e}")
        
    else:
        print(f"\n❌ Tests failed. Check the errors above and:")
        print(f"   1. Ensure model is properly trained")
        print(f"   2. Check all dependencies are installed")
        print(f"   3. Verify dataset images exist")