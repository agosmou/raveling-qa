# utils/image_utils.py
from PIL import Image

def resize_image(image: Image.Image, max_size: int = 300) -> Image.Image:
    """Resize image maintaining aspect ratio."""
    ratio = max_size / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    return image.resize(new_size, Image.Resampling.LANCZOS)
