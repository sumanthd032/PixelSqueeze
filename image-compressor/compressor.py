import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    """
    Load an RGB image and convert it to NumPy arrays for each channel.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        tuple: Original image array, R, G, B channel matrices (2D).
    """
    # Load image using PIL
    img = Image.open(image_path).convert('RGB')  # Ensure RGB format
    img_array = np.array(img, dtype=np.float32)  # Convert to NumPy array (float for SVD)
    
    # Verify shape: (height, width, 3)
    print(f"Image shape: {img_array.shape}")
    
    # Split into R, G, B channels
    r_channel = img_array[:, :, 0]  # Red channel (2D)
    g_channel = img_array[:, :, 1]  # Green channel (2D)
    b_channel = img_array[:, :, 2]  # Blue channel (2D)
    
    # Ensure pixel values are in [0, 255] and float32 for numerical stability
    r_channel = np.clip(r_channel, 0, 255)
    g_channel = np.clip(g_channel, 0, 255)
    b_channel = np.clip(b_channel, 0, 255)
    
    return img_array, r_channel, g_channel, b_channel

def test_image_loading(image_path):
    """
    Test function to load and display the image and its channels.
    """
    # Load and preprocess
    img_array, r_channel, g_channel, b_channel = load_and_preprocess_image(image_path)
    
    # Display original and individual channels using matplotlib
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(img_array.astype(np.uint8))
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("Red Channel")
    plt.imshow(r_channel, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("Green Channel")
    plt.imshow(g_channel, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("Blue Channel")
    plt.imshow(b_channel, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace with path to your sample image
    sample_image_path = "/home/sumanthd032/Projects/PixelSqueeze/image-compressor/sample_image.jpg"  
    test_image_loading(sample_image_path)