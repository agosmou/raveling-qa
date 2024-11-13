# services/feature_extractor.py
import numpy as np
from scipy import stats
from PIL import Image
from io import BytesIO

class RavelingFeatureExtractor:
    def __init__(self, patch_size=75):
        self.patch_size = patch_size

    def load_and_prepare_image(self, uploaded_file) -> Image.Image:
        """Load image from uploaded file."""
        image = Image.open(BytesIO(uploaded_file.read())).convert('RGB')
        return image

    def preprocess_image(self, pil_image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to grayscale numpy array.
        """
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        img = np.array(pil_image).astype(np.float32)
        return img

    def calculate_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the 6 statistical features for a patch or whole image.
        """
        features = [
            np.mean(data),                             # Arithmetic mean
            np.std(data),                              # Standard deviation
            np.sqrt(np.mean(np.square(data))),         # Root mean square
            stats.skew(data.flatten()),                # Skewness
            stats.kurtosis(data.flatten()),            # Kurtosis
            np.percentile(data, 75) - np.percentile(data, 25)  # Interquartile range
        ]
        return np.array(features)

    def calculate_patch_distributions(self, img: np.ndarray) -> np.ndarray:
        """
        Calculate distributions of features across patches.
        """
        height, width = img.shape
        patch_features = []

        # Extract patches and calculate features
        for i in range(0, height - self.patch_size + 1, self.patch_size):
            for j in range(0, width - self.patch_size + 1, self.patch_size):
                patch = img[i:i + self.patch_size, j:j + self.patch_size]
                features = self.calculate_statistical_features(patch)
                patch_features.append(features)

        patch_features = np.array(patch_features)

        # Calculate distributions for each feature
        distributions = []
        for feature_idx in range(6):
            feature_values = patch_features[:, feature_idx]
            hist, _ = np.histogram(feature_values, bins=100, density=True)
            hist = hist / np.sum(hist)
            distributions.extend(hist)

        return np.array(distributions)

    def extract_features(self, pil_image: Image.Image) -> np.ndarray:
        """
        Extract all 606 features from a PIL Image:
        - 6 global statistical features
        - 600 distribution points (100 points x 6 features)
        """
        img = self.preprocess_image(pil_image)
        global_features = self.calculate_statistical_features(img)
        distribution_features = self.calculate_patch_distributions(img)
        all_features = np.concatenate([global_features, distribution_features])

        assert len(all_features) == 606, f"Expected 606 features, got {len(all_features)}"

        return all_features
