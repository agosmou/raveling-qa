# import streamlit as st
# import pandas as pd
# import pickle
# from PIL import Image
# import numpy as np
# from io import BytesIO

# from feature_extractor import RavelingFeatureExtractor

# # --- Load Pre-trained Model and Scaler ---
# @st.cache_resource
# def load_model_and_scaler():
#     with open('rf_model.pkl', 'rb') as model_file:
#         model = pickle.load(model_file)
#     with open('scaler.pkl', 'rb') as scaler_file:
#         scaler = pickle.load(scaler_file)
#     return model, scaler

# # Initialize feature extractor
# @st.cache_resource
# def get_feature_extractor():
#     return RavelingFeatureExtractor()

# def resize_image(image, max_size=300):
#     """Resize image maintaining aspect ratio"""
#     ratio = max_size / max(image.size)
#     new_size = tuple([int(x * ratio) for x in image.size])
#     return image.resize(new_size, Image.Resampling.LANCZOS)

# rf_model, scaler = load_model_and_scaler()
# feature_extractor = get_feature_extractor()

# # --- App Title ---
# st.title("Raveling Severity Prediction App")

# # --- Upload Images ---
# st.header("Upload Images for Prediction")
# uploaded_images = st.file_uploader("Choose Image Files", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

# if uploaded_images:
#     images = []
#     image_titles = []
#     features = []
#     ground_truth_values = []
    
#     # Convert features to DataFrame and get predictions first
#     for uploaded_file in uploaded_images:
#         image = Image.open(uploaded_file).convert('RGB')
#         resized_image = resize_image(image)
#         images.append(resized_image)
#         image_titles.append(uploaded_file.name)
        
#         # Extract features using our feature extractor
#         feature_vector = feature_extractor.extract_features(image)  # Use original image for features
#         features.append(feature_vector)
    
#     X_new = pd.DataFrame(features)
#     X_new_scaled = scaler.transform(X_new)
#     predictions = rf_model.predict(X_new_scaled)
    
#     # Create columns for layout
#     num_cols = 3  # Number of images per row
#     cols = st.columns(num_cols)
    
#     # Display images with predictions and ground truth inputs
#     for idx, image in enumerate(images):
#         col_idx = idx % num_cols
#         with cols[col_idx]:
#             st.image(image, caption=image_titles[idx], use_column_width=True)
#             st.write(f"**Predicted:** {predictions[idx]}")
            
#             # Add ground truth input with numeric validation
#             ground_truth = st.number_input(
#                 f"Ground Truth for {image_titles[idx]}", 
#                 min_value=0, 
#                 max_value=3, 
#                 value=0, 
#                 key=f"ground_truth_{idx}",
#                 help="0=None, 1=Low, 2=Medium, 3=Severe"
#             )
#             ground_truth_values.append(ground_truth)
#             st.write("---")
    
#     # Create and download ground truth CSV
#     if st.button("Generate Ground Truth CSV"):
#         results_df = pd.DataFrame({
#             'image_title': image_titles,
#             'severity': ground_truth_values
#         })
        
#         # Convert dataframe to CSV string
#         csv = results_df.to_csv(index=False)
        
#         # Create download button
#         st.download_button(
#             label="Download Ground Truth CSV",
#             data=csv,
#             file_name="ground_truth.csv",
#             mime="text/csv"
#         )

# # --- Upload Existing Ground Truth CSV ---
# st.header("Upload Existing Ground Truth CSV")
# uploaded_csv = st.file_uploader("Choose a CSV file", type=['csv'])

# if uploaded_csv:
#     ground_truth_df = pd.read_csv(uploaded_csv)
#     st.subheader("Ground Truth Data")
#     st.dataframe(ground_truth_df)

#     # Merge Predictions with Ground Truth
#     if uploaded_images and 'image_titles' in locals():
#         results_df = pd.DataFrame({
#             'Image Title': image_titles,
#             'Predicted Severity': predictions
#         })
#         merged_df = pd.merge(ground_truth_df, results_df, 
#                            left_on='image_title', 
#                            right_on='Image Title', 
#                            how='inner')
#         st.subheader("Comparison of Ground Truth and Predictions")
#         st.dataframe(merged_df)

# # Add legend
# st.sidebar.markdown("""
# ### Severity Legend
# - 0 = None
# - 1 = Low
# - 2 = Medium
# - 3 = Severe
# """)

import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import numpy as np
from io import BytesIO

from feature_extractor import RavelingFeatureExtractor

# --- Load Pre-trained Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    with open('rf_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Initialize feature extractor
@st.cache_resource
def get_feature_extractor():
    return RavelingFeatureExtractor()

def resize_image(image, max_size=300):
    """Resize image maintaining aspect ratio"""
    ratio = max_size / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    return image.resize(new_size, Image.Resampling.LANCZOS)

rf_model, scaler = load_model_and_scaler()
feature_extractor = get_feature_extractor()

# --- App Title ---
st.title("Raveling Severity Prediction App")

# --- Upload Ground Truth CSV First ---
st.header("Upload Ground Truth CSV (Optional)")
uploaded_csv = st.file_uploader("Choose a CSV file", type=['csv'])

# Store ground truth data in session state if CSV is uploaded
if uploaded_csv:
    ground_truth_df = pd.read_csv(uploaded_csv)
    st.session_state['ground_truth_df'] = ground_truth_df
    st.subheader("Ground Truth Data")
    st.dataframe(ground_truth_df)

# --- Upload Images ---
st.header("Upload Images for Prediction")
uploaded_images = st.file_uploader("Choose Image Files", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_images:
    images = []
    image_titles = []
    features = []
    ground_truth_values = []
    
    # Convert features to DataFrame and get predictions first
    for uploaded_file in uploaded_images:
        image = Image.open(uploaded_file).convert('RGB')
        resized_image = resize_image(image)
        images.append(resized_image)
        image_titles.append(uploaded_file.name)
        
        # Extract features using our feature extractor
        feature_vector = feature_extractor.extract_features(image)  # Use original image for features
        features.append(feature_vector)
    
    X_new = pd.DataFrame(features)
    X_new_scaled = scaler.transform(X_new)
    predictions = rf_model.predict(X_new_scaled)
    
    # Create columns for layout
    num_cols = 3  # Number of images per row
    cols = st.columns(num_cols)
    
    # Display images with predictions and ground truth inputs
    for idx, image in enumerate(images):
        col_idx = idx % num_cols
        with cols[col_idx]:
            st.image(image, caption=image_titles[idx], use_column_width=True)
            st.write(f"**Predicted:** {predictions[idx]}")
            
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
            st.write("---")
    
    # Create and download ground truth CSV
    if st.button("Generate Ground Truth CSV"):
        results_df = pd.DataFrame({
            'image_title': image_titles,
            'severity': ground_truth_values
        })
        
        # Convert dataframe to CSV string
        csv = results_df.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="Download Ground Truth CSV",
            data=csv,
            file_name="ground_truth.csv",
            mime="text/csv"
        )

    # Show comparison if ground truth exists
    if 'ground_truth_df' in st.session_state:
        results_df = pd.DataFrame({
            'Image Title': image_titles,
            'Predicted Severity': predictions
        })
        merged_df = pd.merge(st.session_state['ground_truth_df'], results_df, 
                           left_on='image_title', 
                           right_on='Image Title', 
                           how='inner')
        st.subheader("Comparison of Ground Truth and Predictions")
        st.dataframe(merged_df)

# Add legend
st.sidebar.markdown("""
### Severity Legend
- 0 = None
- 1 = Low
- 2 = Medium
- 3 = Severe
""")