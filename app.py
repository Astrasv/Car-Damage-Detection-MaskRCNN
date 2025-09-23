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
from explainable_ai import create_simple_explanation


def main():
    st.set_page_config(
        page_title="Car Damage Detection",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Car Damage Detection with Mask R-CNN")
    st.markdown("Upload an image of a car to detect damage areas and get AI explanations")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["🔍 Damage Detection", "🧠 Explainable AI"])
    
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
    if st.sidebar.button("🔍 Run Comprehensive Evaluation", key="eval_button_1"):
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
    if st.sidebar.button("🧪 Quick Model Test", key="test_button_1"):
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
        image = Image.open(uploaded_file)
        
        # Tab 1: Damage Detection
        with tab1:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Make prediction button
            if st.button("🔍 Detect Damage", type="primary", key="detect_button"):
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
                            st.image(visualized_image, use_container_width=True)
                        
                        # Store predictions in session state for explainable AI tab
                        st.session_state.predictions = predictions
                        st.session_state.predictor = predictor
                        st.session_state.current_image = image
                        
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
                                mime="image/png",
                                key="download_detection"
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
        
        # Tab 2: Explainable AI
        with tab2:
            st.subheader("🧠 Explainable AI - Understanding Model Decisions")
            st.markdown("Explore why the model made its predictions using advanced explanation techniques.")
            
            # Check if predictions are available
            if 'predictions' not in st.session_state or 'predictor' not in st.session_state:
                st.warning("⚠️ Please run damage detection first in the 'Damage Detection' tab to enable explainable AI features.")
                st.info("💡 Upload an image and click 'Detect Damage' to get started!")
            else:
                # Display original image in explainable AI tab
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(st.session_state.current_image, use_container_width=True)
                
                # Explanation method selection
                st.subheader("🎯 Choose Explanation Method")
                explanation_method = st.radio(
                    "Select explanation technique:",
                    ["Grad-CAM", "Occlusion Analysis"],
                    help="Grad-CAM shows where the model focuses attention. Occlusion Analysis shows which areas are most important for the prediction."
                )
                
                if explanation_method == "Grad-CAM":
                    st.markdown("**Grad-CAM** highlights the regions the model focuses on when making predictions.")
                    
                    # Grad-CAM parameters
                    gradcam_alpha = st.slider(
                        "Heatmap Opacity", 
                        min_value=0.1, 
                        max_value=1.0, 
                        value=0.4, 
                        step=0.1,
                        help="Controls the transparency of the heatmap overlay"
                    )
                    
                    if st.button("🔥 Generate Grad-CAM Explanation", key="gradcam_button"):
                        with st.spinner("Generating Grad-CAM explanation..."):
                            try:
                                explanation_result = create_simple_explanation(
                                    st.session_state.predictor,
                                    st.session_state.current_image,
                                    method='gradcam',
                                    alpha=gradcam_alpha
                                )
                                
                                with col2:
                                    st.subheader("Grad-CAM Heatmap")
                                    st.image(explanation_result['explanation_image'], use_container_width=True)
                                
                                st.success("✅ Grad-CAM explanation generated!")
                                st.info(explanation_result['description'])
                                
                                # Download button for explanation
                                img_buffer = io.BytesIO()
                                explanation_result['explanation_image'].save(img_buffer, format='PNG')
                                img_bytes = img_buffer.getvalue()
                                
                                st.download_button(
                                    label="📥 Download Grad-CAM Image",
                                    data=img_bytes,
                                    file_name="gradcam_explanation.png",
                                    mime="image/png",
                                    key="download_gradcam"
                                )
                                
                            except Exception as e:
                                st.error(f"Error generating Grad-CAM: {str(e)}")
                
                elif explanation_method == "Occlusion Analysis":
                    st.markdown("**Occlusion Analysis** shows how much each region affects the model's prediction by systematically blocking parts of the image.")
                    
                    st.warning("⚠️ Occlusion analysis is computationally intensive and may take several minutes to complete.")
                    
                    # Occlusion parameters
                    col_param1, col_param2 = st.columns(2)
                    with col_param1:
                        patch_size = st.slider(
                            "Patch Size", 
                            min_value=20, 
                            max_value=100, 
                            value=50, 
                            step=10,
                            help="Size of the occlusion patches (larger = faster but less precise)"
                        )
                    
                    with col_param2:
                        stride = st.slider(
                            "Stride", 
                            min_value=10, 
                            max_value=50, 
                            value=25, 
                            step=5,
                            help="Step size between patches (larger = faster but less coverage)"
                        )
                    
                    occlusion_alpha = st.slider(
                        "Heatmap Opacity", 
                        min_value=0.1, 
                        max_value=1.0, 
                        value=0.5, 
                        step=0.1
                    )
                    
                    if st.button("🎯 Generate Occlusion Analysis", key="occlusion_button"):
                        with st.spinner(f"Running occlusion analysis (this may take a few minutes)..."):
                            try:
                                explanation_result = create_simple_explanation(
                                    st.session_state.predictor,
                                    st.session_state.current_image,
                                    method='occlusion',
                                    patch_size=patch_size,
                                    stride=stride,
                                    alpha=occlusion_alpha
                                )
                                
                                with col2:
                                    st.subheader("Occlusion Sensitivity Map")
                                    st.image(explanation_result['explanation_image'], use_container_width=True)
                                
                                st.success("✅ Occlusion analysis completed!")
                                st.info(explanation_result['description'])
                                
                                if 'original_score' in explanation_result:
                                    st.write(f"**Original prediction confidence:** {explanation_result['original_score']:.2%}")
                                
                                # Download button for explanation
                                img_buffer = io.BytesIO()
                                explanation_result['explanation_image'].save(img_buffer, format='PNG')
                                img_bytes = img_buffer.getvalue()
                                
                                st.download_button(
                                    label="📥 Download Occlusion Analysis",
                                    data=img_bytes,
                                    file_name="occlusion_analysis.png",
                                    mime="image/png",
                                    key="download_occlusion"
                                )
                                
                            except Exception as e:
                                st.error(f"Error generating occlusion analysis: {str(e)}")
                
                # Information about explanation methods
                with st.expander("ℹ️ About Explanation Methods"):
                    st.markdown("""
                    ### Grad-CAM (Gradient-weighted Class Activation Mapping)
                    - **Fast** and lightweight explanation method
                    - Shows where the model "looks" when making predictions
                    - Red/hot colors indicate high importance regions
                    - **Best for:** Quick understanding of model focus areas
                    
                    ### Occlusion Analysis
                    - **Thorough** but slower explanation method
                    - Tests how prediction changes when parts of image are blocked
                    - Bright colors indicate regions critical for the prediction
                    - **Best for:** Detailed understanding of feature importance
                    
                    ### Interpretation Tips:
                    - **Hot colors (red/yellow)**: High importance regions
                    - **Cool colors (blue/black)**: Low importance regions
                    - Compare explanations with actual damage locations
                    - Multiple explanation methods can provide different insights
                    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with Streamlit and PyTorch Mask R-CNN</p>
            <p>🔍 Damage Detection + 🧠 Explainable AI</p>
            <p>For best results, use clear images of cars with visible damage</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()