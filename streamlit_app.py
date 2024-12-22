# app.py
import math
import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image

from models.model_repository import ModelRepository
from services.feature_extractor import RavelingFeatureExtractor
from services.prediction_service import PredictionService
from utils.image_utils import resize_image
from utils.database import (
    update_image_prediction,
    initialize_db, 
    clear_database,
    insert_csv,
    fetch_csv,
    update_ground_truth,
    insert_image,
    delete_image,
    fetch_images
)

def reset_app():
    """Clears the database and resets the session state."""
    clear_database()
    # Reset all relevant session state variables
    for key in list(st.session_state.keys()):
        if key not in ['upload_key']:  # Preserve 'upload_key' to allow resetting the uploader
            del st.session_state[key]
    st.session_state['images'] = {}
    # Increment the 'upload_key' to reset the file uploader
    st.session_state.upload_key += 1
    st.sidebar.success("All data has been cleared!")

def main():
    # Initialize session state variables
    if 'images' not in st.session_state:
        st.session_state['images'] = {}

    # Initialize the unique key for the file uploader if not present
    if 'upload_key' not in st.session_state:
        st.session_state.upload_key = 0

    initialize_db()
    st.title("Raveling Severity Prediction App")

    # Sidebar Reset Button with Red Styling
    with st.sidebar:
        if st.button("Reset App"):
            reset_app()

    # Initialize Model Repository and Prediction Service
    model_repo = ModelRepository()
    # debugging logging
    # st.write("Model and Scaler loaded successfully.")
    # st.write(f"Model Type: {type(model_repo.model)}")
    # st.write(f"Scaler Type: {type(model_repo.scaler)}")

    prediction_service = PredictionService(model=model_repo.model, scaler=model_repo.scaler)

    # Initialize Feature Extractor
    feature_extractor = RavelingFeatureExtractor()

    # --- Upload Ground Truth CSV First ---
    st.header("Upload Ground Truth CSV (Optional)")
    uploaded_csv = st.file_uploader(
        "Choose a CSV file. Please make sure to include the headers in your CSV file exactly as image_title and severity.",
        type=['csv'],
        key=f'csv_uploader_{st.session_state.upload_key}'
    )

    if uploaded_csv:
        try:
            csv_bytes = uploaded_csv.read()
            ground_truth_df = pd.read_csv(BytesIO(csv_bytes))
            st.session_state['ground_truth_df'] = ground_truth_df
            st.subheader("Ground Truth Data")
            st.dataframe(ground_truth_df)

            # Store CSV in database
            insert_csv(uploaded_csv.name, csv_bytes)

            # If images are already uploaded, update ground_truth_values accordingly
            if st.session_state['images']:
                for title in st.session_state['images']:
                    severity_row = ground_truth_df[ground_truth_df['image_title'] == title]
                    if not severity_row.empty:
                        severity = int(severity_row['severity'].iloc[0])
                        st.session_state['images'][title]['ground_truth'] = severity
                        update_ground_truth(severity, title) # was missing this... critical to update ground truth
                        # st.write(f"Updated ground truth for {title}: {severity}")  # debugging
                st.success("Ground truth values updated based on the uploaded CSV.")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
    else:
        # Fetch CSV data from database
        csv_records = fetch_csv()
        if csv_records:
            csv_id, csv_name, csv_data = csv_records[-1]  # Handle only the most recent CSV
            try:
                ground_truth_df = pd.read_csv(BytesIO(csv_data))
                st.session_state['ground_truth_df'] = ground_truth_df
                st.subheader("Loaded Ground Truth Data")
                st.dataframe(ground_truth_df)

                # If images are already uploaded, initialize ground_truth_values
                if st.session_state['images']:
                    for title in st.session_state['images']:
                        severity_row = ground_truth_df[ground_truth_df['image_title'] == title]
                        if not severity_row.empty:
                            severity = int(severity_row['severity'].iloc[0])
                            st.session_state['images'][title]['ground_truth'] = severity
                            # st.write(f"Loaded ground truth for {title}: {severity}")
            except Exception as e:
                st.error(f"Error loading CSV from database: {e}")
        else:
            st.info("No CSV data found. Please upload a CSV file.")

    # --- Upload Images ---
    st.header("Upload Images for Prediction")
    uploaded_images = st.file_uploader(
        "Choose Image Files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key=f'images_uploader_{st.session_state.upload_key}'
    )

    # Load images from the database if available and no new upload
    if not uploaded_images and not st.session_state['images']:
        db_images = fetch_images()
        if db_images:
            for img in db_images:
                _, image_title, image_data, prediction, ground_truth = img
                try:
                    image = Image.open(BytesIO(image_data)).convert('RGB')  # Now, Image is defined
                    resized_image = resize_image(image)
                    st.session_state['images'][image_title] = {
                        'image': resized_image,
                        'image_data': image_data,  # Store original image bytes
                        'features': None,          # Initialize as None; will be extracted when running predictions
                        'prediction': prediction if prediction is not None else "None",
                        'ground_truth': ground_truth if ground_truth is not None else 0
                    }
                    # st.write(f"Loaded image from DB: {image_title}, Prediction: {prediction}, Ground Truth: {ground_truth}")
                except Exception as e:
                    st.error(f"Error loading image {image_title} from database: {e}")

    # Process images only once and store in session_state
    if uploaded_images:
        for uploaded_file in uploaded_images:
            try:
                # Read image data as bytes
                image_data = uploaded_file.getvalue()  # bytes

                # Validate the image before processing
                try:
                    Image.open(BytesIO(image_data))
                except Exception:
                    st.error(f"The file {uploaded_file.name} is not a valid image.")
                    continue  # Skip to the next file

                # Load and resize image for display
                image = Image.open(BytesIO(image_data)).convert('RGB')  # Convert to RGB for display
                resized_image = resize_image(image)

                # Initialize ground truth
                if 'ground_truth_df' in st.session_state:
                    ground_truth_df = st.session_state['ground_truth_df']
                    severity_row = ground_truth_df[ground_truth_df['image_title'] == uploaded_file.name]
                    ground_truth = int(severity_row['severity'].iloc[0]) if not severity_row.empty else 0
                else:
                    ground_truth = 0

                # Store in session state without features but with image_data
                st.session_state['images'][uploaded_file.name] = {
                    'image': resized_image,
                    'image_data': image_data,  # Store original image bytes
                    'features': None,          # Placeholder for features to be extracted later
                    'prediction': "None",
                    'ground_truth': ground_truth
                }

                # st.write(f"Uploaded image: {uploaded_file.name}, Ground Truth: {ground_truth}")

                # Insert images into the database with prediction as None
                try:
                    insert_image(
                        image_title=uploaded_file.name,
                        image_data=image_data,  # Use the already read image_data
                        prediction=None,        # Set prediction to None initially
                        ground_truth=ground_truth
                    )
                    # st.write(f"Inserted/Updated image {uploaded_file.name} into the database.")
                except Exception as e:
                    st.error(f"Error inserting image {uploaded_file.name} into database: {e}")

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

    if st.button("Run Predictions"):
        if st.session_state['images']:
            try:
                # Initialize list to hold feature vectors
                features_list = []
                image_titles = []
                
                # Initialize progress spinner
                with st.spinner("Extracting image features and running raveling severity predictions. Please wait as this may take a few minutes..."):
                    for title, data in st.session_state['images'].items():
                        # Check if features are already extracted
                        if data['features'] is None:
                            # Extract features using original image bytes
                            feature_vector = feature_extractor.extract_features(data['image_data'])
                            st.session_state['images'][title]['features'] = feature_vector
                            # st.write(f"Extracted features for {title}.")
                        else:
                            feature_vector = data['features']
                            # st.write(f"Using cached features for {title}.")
                        
                        features_list.append(feature_vector)
                        image_titles.append(title)
                
                # Run predictions
                predictions = prediction_service.predict(features_list)
                # st.write(f"Predictions: {predictions}")

                # Convert predictions to integers if necessary
                predictions = [int(pred) for pred in predictions]

                # Update predictions in session state and database
                for idx, title in enumerate(image_titles):
                    pred = predictions[idx]
                    st.session_state['images'][title]['prediction'] = pred
                    update_image_prediction(title, pred) # updates here too
                    # st.write(f"Updated prediction for {title}: {pred}")
                
                st.success("Predictions have been successfully updated.")
            except Exception as e:
                st.error(f"Error running predictions: {e}")
        else:
            st.warning("No images available for prediction.")

    # Define pagination for displaying images
    if st.session_state['images']:
        image_titles = sorted(st.session_state['images'].keys())
        total_images = len(image_titles)
        num_cols = 4  # Number of images per row
        images_per_page = num_cols * 3  # 3 rows per page
        total_pages = math.ceil(total_images / images_per_page)

        # Initialize page number if not in session state
        if 'current_page' not in st.session_state:
            st.session_state['current_page'] = 0

        # Determine the images to display for the current page
        start_idx = st.session_state['current_page'] * images_per_page
        end_idx = start_idx + images_per_page
        page_image_titles = image_titles[start_idx:end_idx]

        # Create layout columns for the current page
        cols = st.columns(num_cols)

        # Display images with predictions, ground truth inputs, and delete buttons
        for idx, image_title in enumerate(page_image_titles):
            col_idx = idx % num_cols
            with cols[col_idx]:
                image_data = st.session_state['images'][image_title]['image']
                prediction = st.session_state['images'][image_title]['prediction']
                ground_truth = st.session_state['images'][image_title]['ground_truth']

                # Display the image
                st.image(image_data, caption=image_title, use_column_width=True)
                st.write(f"**Predicted Severity:** {prediction}")

                # Ground Truth Input
                ground_truth_input = st.number_input(
                    f"Ground Truth for {image_title}",
                    min_value=0,
                    max_value=3,
                    value=ground_truth,
                    key=f"ground_truth_{image_title}_",
                )

                # Update ground truth in session state and database
                st.session_state['images'][image_title]['ground_truth'] = ground_truth_input
                try:
                    update_ground_truth(
                        ground_truth=ground_truth_input,
                        image_title=image_title
                    )
                    # st.write(f"Ground truth for {image_title} set to {ground_truth_input}")
                except Exception as e:
                    st.error(f"Error updating ground truth for {image_title}: {e}")

                # Delete Button
                if st.button("Delete Image", key=f"delete_{image_title}"):
                    try:
                        # Delete from database
                        delete_image(image_title)
                        # Delete from session state
                        del st.session_state['images'][image_title]
                        st.success(f"Image '{image_title}' has been deleted successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting image '{image_title}': {e}")
                
                st.markdown("---")

        # Display page navigation buttons
        st.write(f"Showing page {st.session_state['current_page'] + 1} of {total_pages}")

        # Allow user to jump to a specific page
        st.markdown("### Jump to Page")
        page_input = st.number_input(
            "Enter page number:",
            min_value=1,
            max_value=total_pages,
            value=st.session_state['current_page'] + 1,
            step=1,
            key="page_input",
            help="Type the page number you want to navigate to."
        )

        # Update the current page based on user input
        if st.session_state['current_page'] != page_input - 1:
            st.session_state['current_page'] = page_input - 1
            st.rerun()

        # Previous and Next Buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            prev_clicked = st.button("◀ Previous")
            if prev_clicked and st.session_state['current_page'] > 0:
                st.session_state['current_page'] -= 1 
                st.rerun()
        with col3:
            next_clicked = st.button("Next ▶")
            if next_clicked and st.session_state['current_page'] < total_pages - 1:
                st.session_state['current_page'] += 1 
                st.rerun()

        st.markdown("---")

        # Create and download predictions CSV
        if st.button("Generate Ground Truth vs Predictions CSV"):
            try:
                # Create results dataframe using current values from session state
                results_df = pd.DataFrame([
                    {
                        'image_title': title,
                        'predicted_severity': st.session_state['images'][title]['prediction'] if st.session_state['images'][title]['prediction'] is not None else "N/A",
                        'ground_truth_severity': st.session_state['images'][title]['ground_truth']
                    }
                    for title in image_titles
                ])
                
                # Convert to numeric, handling any non-numeric values
                results_df['predicted_severity'] = pd.to_numeric(results_df['predicted_severity'], errors='coerce')
                results_df['ground_truth_severity'] = pd.to_numeric(results_df['ground_truth_severity'], errors='coerce')
                
                csv = results_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Ground Truth vs Predictions CSV",
                    data=csv,
                    file_name="gt_vs_pred.csv",
                    mime="text/csv"
                )
                st.success("CSV generated successfully.")
            except Exception as e:
                st.error(f"Error generating CSV: {e}")

        # Show comparison if ground truth exists
        if 'ground_truth_df' in st.session_state:
            try:
                # Create comparison dataframe using current values
                current_predictions = pd.DataFrame([
                    {
                        'image_title': title,
                        'Predicted Severity': st.session_state['images'][title]['prediction'],
                        'Ground Truth Severity': st.session_state['images'][title]['ground_truth']
                    }
                    for title in image_titles
                ])
                
                # Convert 'None' strings to NaN
                current_predictions['Predicted Severity'] = current_predictions['Predicted Severity'].replace('None', pd.NA)
                
                # Convert to numeric, handling any non-numeric values
                current_predictions['Predicted Severity'] = pd.to_numeric(current_predictions['Predicted Severity'], errors='coerce')
                current_predictions['Ground Truth Severity'] = pd.to_numeric(current_predictions['Ground Truth Severity'], errors='coerce')
                
                st.subheader("Comparison of Ground Truth and Predictions")
                
                # Create a styled dataframe for display
                styled_df = current_predictions.style.format({
                    'Predicted Severity': '{:.0f}',
                    'Ground Truth Severity': '{:.0f}'
                })
                
                st.dataframe(styled_df)
                
            except Exception as e:
                st.error(f"Error creating comparison table: {e}")

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
