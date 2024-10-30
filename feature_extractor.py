import numpy as np
from scipy import stats
from PIL import Image

class RavelingFeatureExtractor:
    def __init__(self, patch_size=75):
        self.patch_size = patch_size
        
    def preprocess_image(self, pil_image):
        """
        Convert PIL Image to grayscale numpy array
        """
        # Convert to grayscale if not already
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
            
        # Convert to numpy array and float32
        img = np.array(pil_image).astype(np.float32)
        
        return img
        
    def calculate_statistical_features(self, data):
        """
        Calculate the 6 statistical features for a patch or whole image
        """
        features = []
        
        # 1. Arithmetic mean
        features.append(np.mean(data))
        
        # 2. Standard deviation
        features.append(np.std(data))
        
        # 3. Root mean square
        rms = np.sqrt(np.mean(np.square(data)))
        features.append(rms)
        
        # 4. Skewness
        features.append(stats.skew(data.flatten()))
        
        # 5. Kurtosis
        features.append(stats.kurtosis(data.flatten()))
        
        # 6. Interquartile range
        q75, q25 = np.percentile(data, [75, 25])
        features.append(q75 - q25)
        
        return np.array(features)
        
    def calculate_patch_distributions(self, img):
        """
        Calculate distributions of features across patches
        """
        height, width = img.shape
        patch_features = []
        
        # Extract patches and calculate features
        for i in range(0, height - self.patch_size + 1, self.patch_size):
            for j in range(0, width - self.patch_size + 1, self.patch_size):
                patch = img[i:i+self.patch_size, j:j+self.patch_size]
                features = self.calculate_statistical_features(patch)
                patch_features.append(features)
                
        patch_features = np.array(patch_features)
        
        # Calculate distributions for each feature
        distributions = []
        for feature_idx in range(6):
            feature_values = patch_features[:, feature_idx]
            # Calculate histogram with 100 bins
            hist, _ = np.histogram(feature_values, bins=100, density=True)
            # Normalize histogram
            hist = hist / np.sum(hist)
            distributions.extend(hist)
            
        return np.array(distributions)
    
    def extract_features(self, pil_image):
        """
        Extract all 606 features from a PIL Image:
        - 6 global statistical features
        - 600 distribution points (100 points x 6 features)
        """
        # Preprocess image
        img = self.preprocess_image(pil_image)
        
        # Calculate global features
        global_features = self.calculate_statistical_features(img)
        
        # Calculate patch distribution features
        distribution_features = self.calculate_patch_distributions(img)
        
        # Combine all features
        all_features = np.concatenate([global_features, distribution_features])
        
        # Verify we have exactly 606 features
        assert len(all_features) == 606, f"Expected 606 features, got {len(all_features)}"
        
        return all_features