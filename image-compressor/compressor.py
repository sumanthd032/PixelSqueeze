import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_and_preprocess_image(image_path):
    """
    Load an RGB image and convert it to NumPy arrays for each channel.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        tuple: Original image array, R, G, B channel matrices (2D).
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    print(f"Image shape: {img_array.shape}")
    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1]
    b_channel = img_array[:, :, 2]
    r_channel = np.clip(r_channel, 0, 255)
    g_channel = np.clip(g_channel, 0, 255)
    b_channel = np.clip(b_channel, 0, 255)
    return img_array, r_channel, g_channel, b_channel

def compress_image_svd(r_channel, g_channel, b_channel, k):
    """
    Compress image channels using SVD with k singular values.
    
    Args:
        r_channel, g_channel, b_channel: 2D NumPy arrays for each channel.
        k (int): Number of singular values to keep.
    
    Returns:
        tuple: Compressed image array, compressed R, G, B channels.
    """
    # Apply SVD to each channel
    U_r, S_r, Vt_r = np.linalg.svd(r_channel, full_matrices=False)
    U_g, S_g, Vt_g = np.linalg.svd(g_channel, full_matrices=False)
    U_b, S_b, Vt_b = np.linalg.svd(b_channel, full_matrices=False)
    
    # Keep only top k singular values and corresponding vectors
    U_r_k = U_r[:, :k]
    S_r_k = np.diag(S_r[:k])
    Vt_r_k = Vt_r[:k, :]
    
    U_g_k = U_g[:, :k]
    S_g_k = np.diag(S_g[:k])
    Vt_g_k = Vt_g[:k, :]
    
    U_b_k = U_b[:, :k]
    S_b_k = np.diag(S_b[:k])
    Vt_b_k = Vt_b[:k, :]
    
    # Reconstruct each channel
    r_compressed = np.dot(U_r_k, np.dot(S_r_k, Vt_r_k))
    g_compressed = np.dot(U_g_k, np.dot(S_g_k, Vt_g_k))
    b_compressed = np.dot(U_b_k, np.dot(S_b_k, Vt_b_k))
    
    # Clip values to [0, 255] and convert to uint8
    r_compressed = np.clip(r_compressed, 0, 255).astype(np.uint8)
    g_compressed = np.clip(g_compressed, 0, 255).astype(np.uint8)
    b_compressed = np.clip(b_compressed, 0, 255).astype(np.uint8)
    
    # Combine channels into compressed image
    compressed_image = np.stack([r_compressed, g_compressed, b_compressed], axis=2)
    
    return compressed_image, r_compressed, g_compressed, b_compressed

def test_compression(image_path, k_values=[10, 50, 100]):
    """
    Test compression with different k values and display results.
    
    Args:
        image_path (str): Path to input image.
        k_values (list): List of k values to test.
    """
    # Ensure output directory exists
    os.makedirs("frontend/static/uploads", exist_ok=True)
    
    # Load and preprocess image
    img_array, r_channel, g_channel, b_channel = load_and_preprocess_image(image_path)
    
    # Display original and compressed images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, len(k_values) + 1, 1)
    plt.title("Original Image")
    plt.imshow(img_array.astype(np.uint8))
    plt.axis('off')
    
    # Compress and display for each k
    for i, k in enumerate(k_values, 1):
        compressed_image, _, _, _ = compress_image_svd(r_channel, g_channel, b_channel, k)
        
        # Save compressed image
        compressed_path = f"frontend/static/uploads/compressed_k{k}.jpg"
        Image.fromarray(compressed_image).save(compressed_path)
        
        plt.subplot(1, len(k_values) + 1, i + 1)
        plt.title(f"Compressed (k={k})")
        plt.imshow(compressed_image)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sample_image_path = "image-compressor/sample_image.jpg" 
    test_compression(sample_image_path, k_values=[10, 50, 100])