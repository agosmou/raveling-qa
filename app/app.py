# app.py
import streamlit as st
import pandas as pd
from io import BytesIO

from models.model_repository import ModelRepository
from services.feature_extractor import RavelingFeatureExtractor
from services.prediction_service import PredictionService
from utils.image_utils import resize_image

def main():
    st.title("Raveling Severity Prediction App")

    # Initialize Model Repository
    model_repo = ModelRepository()
    prediction_service = PredictionService(model=model_repo.model, scaler=model_repo.scaler)

    # Initialize Feature Extractor
    feature_extractor = RavelingFeatureExtractor()

    # --- Upload Ground Truth CSV First ---
    st.header("Upload Ground Truth CSV (Optional)")
    uploaded_csv = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_csv:
        try:
            ground_truth_df = pd.read_csv(uploaded_csv)
            st.session_state['ground_truth_df'] = ground_truth_df
            st.subheader("Ground Truth Data")
            st.dataframe(ground_truth_df)
        except Exception as e:
            st.error(f"Error loading CSV: {e}")

    # --- Upload Images ---
    st.header("Upload Images for Prediction")
    uploaded_images = st.file_uploader("Choose Image Files", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    if uploaded_images:
        images = []
        image_titles = []
        features = []
        ground_truth_values = []

        # Process each uploaded image
        for uploaded_file in uploaded_images:
            try:
                # Load and prepare image
                image = feature_extractor.load_and_prepare_image(uploaded_file)
                resized_image = resize_image(image)
                images.append(resized_image)
                image_titles.append(uploaded_file.name)

                # Extract features
                feature_vector = feature_extractor.extract_features(image)
                features.append(feature_vector)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

        if features:
            # Get predictions
            try:
                predictions = prediction_service.predict(features)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                predictions = [None] * len(features)

            # Create layout columns
            num_cols = 3  # Number of images per row
            cols = st.columns(num_cols)

            # Display images with predictions and ground truth inputs
            for idx, image in enumerate(images):
                col_idx = idx % num_cols
                with cols[col_idx]:
                    st.image(image, caption=image_titles[idx], use_column_width=True)
                    if predictions[idx] is not None:
                        st.write(f"**Predicted Severity:** {predictions[idx]}")
                    else:
                        st.write("**Predicted Severity:** Error")

                    # Get default value from ground truth CSV if it exists
                    default_value = 0
                    if 'ground_truth_df' in st.session_state:
                        gt_df = st.session_state['ground_truth_df']
                        matching_row = gt_df[gt_df['image_title'] == image_titles[idx]]
                        if not matching_row.empty:
                            default_value = int(matching_row['severity'].iloc[0])

                    # Add ground truth input with pre-filled value
                    ground_truth = st.number_input(
                        f"Ground Truth for {image_titles[idx]}",
                        min_value=0,
                        max_value=3,
                        value=default_value,
                        key=f"ground_truth_{idx}",
                        help="0=None, 1=Low, 2=Medium, 3=Severe"
                    )
                    ground_truth_values.append(ground_truth)
                    st.markdown("---")

            # Create and download ground truth CSV
            if st.button("Generate Ground Truth CSV"):
                try:
                    results_df = pd.DataFrame({
                        'image_title': image_titles,
                        'severity': ground_truth_values
                    })

                    csv = results_df.to_csv(index=False)

                    st.download_button(
                        label="Download Ground Truth CSV",
                        data=csv,
                        file_name="ground_truth.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error generating CSV: {e}")

            # Show comparison if ground truth exists
            if 'ground_truth_df' in st.session_state:
                try:
                    results_df = pd.DataFrame({
                        'Image Title': image_titles,
                        'Predicted Severity': predictions
                    })
                    merged_df = pd.merge(
                        st.session_state['ground_truth_df'],
                        results_df,
                        left_on='image_title',
                        right_on='Image Title',
                        how='inner'
                    )
                    st.subheader("Comparison of Ground Truth and Predictions")
                    st.dataframe(merged_df)
                except Exception as e:
                    st.error(f"Error merging data: {e}")

    # Add legend in the sidebar
    st.sidebar.markdown("""
    ### Severity Legend
    - **0** = None
    - **1** = Low
    - **2** = Medium
    - **3** = Severe
    """)

if __name__ == "__main__":
    main()
