import streamlit as st
import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
import sys
import io

# Add src to path for imports
sys.path.append('src')

from inference import CarDamagePredictor, predict_damage


def main():
    st.set_page_config(
        page_title="Car Damage Detection",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Car Damage Detection with Mask R-CNN")
    st.markdown("Upload an image of a car to detect damage areas")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model path
    model_path = st.sidebar.text_input(
        "Model Path",
        value="models/best_model.pth",
        help="Path to the trained model checkpoint"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.01,
        max_value=1.0,
        value=0.1,  # Lower default value
        step=0.01,
        help="Minimum confidence for damage detection"
    )
    
    # Sidebar evaluation
    st.sidebar.subheader("Model Evaluation")
    if st.sidebar.button("🔍 Run Comprehensive Evaluation"):
        if os.path.exists(model_path):
            with st.spinner("Running comprehensive evaluation..."):
                try:
                    from evaluation import run_comprehensive_evaluation
                    run_comprehensive_evaluation(model_path, 'dataset')
                    st.success("✅ Evaluation completed! Check 'evaluation_results' folder.")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
        else:
            st.error("Model not found for evaluation")
    
    # Quick model test button
    if st.sidebar.button("🧪 Quick Model Test"):
        if os.path.exists(model_path):
            with st.spinner("Testing model..."):
                try:
                    predictor = CarDamagePredictor(model_path, confidence_threshold=0.01)
                    st.success("✅ Model loads successfully!")
                    st.info(f"Model ready on {predictor.device}")
                except Exception as e:
                    st.error(f"Model test failed: {e}")
        else:
            st.error("Model not found for testing")
    
    # Sidebar evaluation
    st.sidebar.subheader("Model Evaluation")
    if st.sidebar.button("🔍 Run Comprehensive Evaluation"):
        if os.path.exists(model_path):
            with st.spinner("Running comprehensive evaluation..."):
                try:
                    from evaluation import run_comprehensive_evaluation
                    run_comprehensive_evaluation(model_path, 'dataset')
                    st.success("✅ Evaluation completed! Check 'evaluation_results' folder.")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
        else:
            st.error("Model not found for evaluation")
    
    # Quick model test button
    if st.sidebar.button("🧪 Quick Model Test"):
        if os.path.exists(model_path):
            with st.spinner("Testing model..."):
                try:
                    predictor = CarDamagePredictor(model_path, confidence_threshold=0.01)
                    st.success("✅ Model loads successfully!")
                    st.info(f"Model ready on {predictor.device}")
                except Exception as e:
                    st.error(f"Model test failed: {e}")
        else:
            st.error("Model not found for testing")
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("Debug Mode", help="Show detailed prediction information")
    
    # Visualization options
    st.sidebar.subheader("Visualization Options")
    show_bboxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
    show_masks = st.sidebar.checkbox("Show Masks", value=True)
    mask_alpha = st.sidebar.slider("Mask Transparency", 0.1, 1.0, 0.3, 0.1)
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.info("Please train the model first or provide a valid model path")
        
        with st.expander("How to train the model"):
            st.code("""
# First, organize your dataset in the following structure:
# dataset/
# ├── train/
# │   ├── images/
# │   └── via_region_data.json
# ├── val/
# │   ├── images/
# │   └── via_region_data.json
# └── test/
#     ├── images/
#     └── via_region_data.json

# Then run the training script:
python src/train.py
            """, language="bash")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of a car to detect damage"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Make prediction button
        if st.button("🔍 Detect Damage", type="primary"):
            with st.spinner("Analyzing image for damage..."):
                try:
                    # Initialize predictor
                    predictor = CarDamagePredictor(
                        model_path, 
                        confidence_threshold=confidence_threshold
                    )
                    
                    # Make prediction
                    predictions = predictor.predict(image, debug=debug_mode)
                    
                    # Debug information
                    if debug_mode:
                        st.subheader("🔍 Debug Information")
                        st.write(f"**Raw detections:** {len(predictions['all_scores'])}")
                        st.write(f"**Score range:** {predictions['all_scores'].min():.4f} - {predictions['all_scores'].max():.4f}")
                        st.write(f"**Detections above threshold:** {predictions['num_detections']}")
                        
                        if len(predictions['all_scores']) > 0:
                            st.write("**All detection scores:**")
                            scores_df = pd.DataFrame({
                                'Detection': range(1, len(predictions['all_scores']) + 1),
                                'Score': predictions['all_scores'],
                                'Above_Threshold': predictions['all_scores'] > confidence_threshold
                            })
                            st.dataframe(scores_df)
                    
                    # Create visualization based on options
                    if show_bboxes or show_masks:
                        # Custom visualization
                        visualized_image = predictor.visualize_predictions(
                            image, predictions, mask_alpha=mask_alpha
                        )
                    else:
                        visualized_image = image
                    
                    # Display results
                    with col2:
                        st.subheader("Damage Detection Results")
                        st.image(visualized_image, use_column_width=True)
                    
                    # Display detection summary
                    st.subheader("📊 Detection Summary")
                    
                    num_detections = predictions['num_detections']
                    
                    if num_detections > 0:
                        st.success(f"Found {num_detections} damage area(s)")
                        
                        # Create results table
                        results_data = []
                        for i in range(num_detections):
                            results_data.append({
                                'Detection': f"Damage {i+1}",
                                'Confidence': f"{predictions['scores'][i]:.2%}",
                                'Bounding Box': f"({predictions['boxes'][i][0]:.0f}, {predictions['boxes'][i][1]:.0f}, {predictions['boxes'][i][2]:.0f}, {predictions['boxes'][i][3]:.0f})"
                            })
                        
                        st.table(results_data)
                        
                        # Download button for result image
                        img_buffer = io.BytesIO()
                        visualized_image.save(img_buffer, format='PNG')
                        img_bytes = img_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Result Image",
                            data=img_bytes,
                            file_name="damage_detection_result.png",
                            mime="image/png"
                        )
                        
                    else:
                        st.info("No damage detected in the image")
                        
                    # Additional metrics
                    with st.expander("📈 Detection Details"):
                        st.write(f"**Confidence Threshold:** {confidence_threshold}")
                        st.write(f"**Total Detections:** {num_detections}")
                        st.write(f"**Image Size:** {image.size}")
                        st.write(f"**Device Used:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
                        
                        if num_detections > 0:
                            avg_confidence = np.mean(predictions['scores'])
                            st.write(f"**Average Confidence:** {avg_confidence:.2%}")
                            st.write(f"**Highest Confidence:** {np.max(predictions['scores']):.2%}")
                            st.write(f"**Lowest Confidence:** {np.min(predictions['scores']):.2%}")
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.error("Please check your model path and try again")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with Streamlit and PyTorch Mask R-CNN</p>
            <p>For best results, use clear images of cars with visible damage</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()