# services/feature_extractor.py
import math
import time
import streamlit as st
import numpy as np
from scipy.stats import iqr, skew, kurtosis, norm
from PIL import Image
from io import BytesIO

class RavelingFeatureExtractor:
    def __init__(self, p_size=25):
        self.p_size = p_size
        self.height = 417
        self.width = 520
        self.h_patch = math.ceil(self.height / self.p_size)
        self.w_patch = math.ceil(self.width / self.p_size)
        
        # Precompute linspace arrays for distribution features
        self.std_linspace = np.linspace(0, 30, 100)
        self.iqr_linspace = np.linspace(0, 30, 100)
        self.mean_linspace = np.linspace(100, 140, 100)
        self.rms_linspace = np.linspace(0, 20, 100)
        self.skew_linspace = np.linspace(-8, 5, 100)
        self.kurt_linspace = np.linspace(-5, 10, 100)
        
    def load_and_prepare_image(self, uploaded_file: bytes) -> Image.Image:
        """
        Load image from uploaded file bytes and convert to grayscale.
        
        Args:
            uploaded_file (bytes): Image file in bytes.
        
        Returns:
            Image.Image: Grayscale PIL Image.
        
        Raises:
            ValueError: If the image cannot be opened or converted.
        """
        try:
            image = Image.open(BytesIO(uploaded_file)).convert('L')  # Convert to grayscale
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
        
    def resize_image(self, img: Image.Image, target_size=(1250, 1040)) -> Image.Image:
        """
        Resize image to target size using PIL without maintaining aspect ratio.
        
        Args:
            img (Image.Image): PIL Image to resize.
            target_size (tuple): Desired size as (width, height).
        
        Returns:
            Image.Image: Resized PIL Image.
        """
        try:
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            return resized_img
        except Exception as e:
            raise ValueError(f"Failed to resize image: {e}")
        
    def split_into_smaller_images(self, img: Image.Image) -> list:
        """
        Split the larger image into 6 smaller images as per mentor's featurization.
        Assumes the image has been resized to (1250, 1040).
        
        Args:
            img (Image.Image): Resized PIL Image.
        
        Returns:
            list: List of 6 NumPy arrays representing smaller images.
        
        Raises:
            ValueError: If the image dimensions are incorrect.
        """
        if img.size != (1250, 1040):
            raise ValueError(f"Image has incorrect size: {img.size}. Expected (1250, 1040).")
        
        img_np = np.array(img)
        smaller_images = [
            img_np[0:417, 0:520],
            img_np[0:417, 520:1040],
            img_np[417:834, 0:520],
            img_np[417:834, 520:1040],
            img_np[834:1250, 0:520],
            img_np[834:1250, 520:1040]
        ]
        return smaller_images
        
    def calculate_global_features(self, img: np.ndarray) -> np.ndarray:
        """
        Calculate the 6 global statistical features for a given image patch.
        
        Args:
            img (np.ndarray): Grayscale image patch.
        
        Returns:
            np.ndarray: Array of 6 global features.
        """
        std_val = np.std(img)
        iqr_val = iqr(img)
        mean_val = np.mean(np.abs(img))
        rms_val = np.sqrt(np.mean(img ** 2))
        skew_val = skew(img.flatten())
        kurt_val = kurtosis(img.flatten())
        global_features = np.array([std_val, iqr_val, mean_val, rms_val, skew_val, kurt_val])
        return global_features
        
    def calculate_patch_features(self, img: np.ndarray) -> np.ndarray:
        """
        Calculate distribution features from patches using vectorized operations.
        
        Args:
            img (np.ndarray): Grayscale image patch.
        
        Returns:
            np.ndarray: Concatenated array of 600 distribution features.
        """
        # Initialize lists to store patch-wise features
        std_val = []
        iqr_val = []
        mean_val = []
        rms_val = []
        skw_val = []
        kurt_val = []
        
        # Efficiently iterate over patches without unnecessary computations
        for h in range(self.h_patch):
            for w in range(self.w_patch):
                start_h = h * self.p_size
                end_h = min((h + 1) * self.p_size, self.height)
                start_w = w * self.p_size
                end_w = min((w + 1) * self.p_size, self.width)
                
                patch = img[start_h:end_h, start_w:end_w]
                
                if patch.size == 0:
                    continue  # Skip empty patches
                
                # Append computed features to respective lists
                std_val.append(np.std(patch))
                iqr_val.append(iqr(patch))
                mean_val.append(np.mean(np.abs(patch)))
                rms_val.append(np.sqrt(np.mean(patch ** 2)))
                skw_val.append(skew(patch.flatten()))
                kurt_val.append(kurtosis(patch.flatten()))
        
        # Convert lists to NumPy arrays for vectorized operations
        std_val = np.array(std_val)
        iqr_val = np.array(iqr_val)
        mean_val = np.array(mean_val)
        rms_val = np.array(rms_val)
        skw_val = np.array(skw_val)
        kurt_val = np.array(kurt_val)
        
        # Initialize distribution feature arrays
        std_dist = np.zeros(100)
        iqr_dist = np.zeros(100)
        mean_dist = np.zeros(100)
        rms_dist = np.zeros(100)
        skew_dist = np.zeros(100)
        kurt_dist = np.zeros(100)
        
        # Vectorized PDF computations using list comprehensions
        if std_val.size > 0:
            std_pdf = [norm(xi).pdf(self.std_linspace) for xi in std_val]
            std_dist = np.sum(std_pdf, axis=0)
        
        if iqr_val.size > 0:
            iqr_pdf = [norm(xi).pdf(self.iqr_linspace) for xi in iqr_val]
            iqr_dist = np.sum(iqr_pdf, axis=0)
        
        if mean_val.size > 0:
            mean_pdf = [norm(xi).pdf(self.mean_linspace) for xi in mean_val]
            mean_dist = np.sum(mean_pdf, axis=0)
        
        if rms_val.size > 0:
            rms_pdf = [norm(xi).pdf(self.rms_linspace) for xi in rms_val]
            rms_dist = np.sum(rms_pdf, axis=0)
        
        if skw_val.size > 0:
            skew_pdf = [norm(xi).pdf(self.skew_linspace) for xi in skw_val]
            skew_dist = np.sum(skew_pdf, axis=0)
        
        if kurt_val.size > 0:
            kurt_pdf = [norm(xi).pdf(self.kurt_linspace) for xi in kurt_val]
            kurt_dist = np.sum(kurt_pdf, axis=0)
        
        # Concatenate all distribution features
        distribution_features = np.concatenate([
            std_dist,
            iqr_dist,
            mean_dist,
            rms_dist,
            skew_dist,
            kurt_dist
        ])
        
        # Ensure the concatenated distribution_features has size 600
        if distribution_features.shape[0] != 600:
            raise ValueError(f"Distribution features have incorrect shape: {distribution_features.shape}. Expected (600,)")
        
        return distribution_features
        
    def extract_features(self, uploaded_file: bytes) -> np.ndarray:
        """
        Extract 606 features from an uploaded image:
        - 6 global statistical features per smaller image
        - 600 distribution features (100 points x 6 features) per smaller image
        Aggregated across all 6 smaller images by taking the mean.
        
        Args:
            uploaded_file (bytes): Image file in bytes.
        
        Returns:
            np.ndarray: Aggregated feature vector of length 606.
        
        Raises:
            ValueError: If any step in the feature extraction fails.
        """
        start_time = time.time()
    
        # Load and prepare image
        img = self.load_and_prepare_image(uploaded_file)
        
        # Resize image to expected size
        img = self.resize_image(img)
        
        # Split into 6 smaller images
        smaller_images = self.split_into_smaller_images(img)
        
        all_features = []
        
        for j, smaller_img in enumerate(smaller_images):
            # Calculate global features
            global_features = self.calculate_global_features(smaller_img)
            
            # Calculate distribution features
            distribution_features = self.calculate_patch_features(smaller_img)
            
            # Combine global and distribution features
            features = np.concatenate([global_features, distribution_features])
            
            # Verify combined feature vector length
            if features.shape[0] != 606:
                raise ValueError(f"Combined features for smaller image {j} have incorrect shape: {features.shape}. Expected (606,)")
            
            all_features.append(features)
        
        # Convert to NumPy array and aggregate by taking the mean across the 6 smaller images
        try:
            all_features = np.array(all_features)
            if all_features.shape != (6, 606):
                raise ValueError(f"All features array has incorrect shape: {all_features.shape}. Expected (6, 606).")
            aggregated_features = np.mean(all_features, axis=0)
            end_time = time.time()
            # Optional: Log the feature extraction time
            # st.write(f"Feature extraction time: {end_time - start_time:.2f} seconds")
        except ValueError as ve:
            raise ValueError(f"Error aggregating features: {ve}")
        
        return aggregated_features
