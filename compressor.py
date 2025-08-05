import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_and_preprocess_image(image_source):
    """
    Load an RGB image from a path or file stream and convert it to NumPy arrays.
    
    Args:
        image_source (str or file-like object): Path to the input image or a file stream.
    
    Returns:
        tuple: Original image array, R, G, B channel matrices (2D).
    """
    # Image.open can handle both file paths and in-memory streams directly.
    img = Image.open(image_source).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    print(f"Image shape: {img_array.shape}")
    
    # Separate the channels
    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1]
    b_channel = img_array[:, :, 2]
    
    # Clipping values to ensure they are within the 0-255 range
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
        tuple: Compressed image array, compressed R, G, B channels, SVD components.
    """
    # Perform SVD on each channel
    U_r, S_r, Vt_r = np.linalg.svd(r_channel, full_matrices=False)
    U_g, S_g, Vt_g = np.linalg.svd(g_channel, full_matrices=False)
    U_b, S_b, Vt_b = np.linalg.svd(b_channel, full_matrices=False)
    
    # Truncate the matrices to the given k value
    U_r_k = U_r[:, :k]
    S_r_k = np.diag(S_r[:k])
    Vt_r_k = Vt_r[:k, :]
    
    U_g_k = U_g[:, :k]
    S_g_k = np.diag(S_g[:k])
    Vt_g_k = Vt_g[:k, :]
    
    U_b_k = U_b[:, :k]
    S_b_k = np.diag(S_b[:k])
    Vt_b_k = Vt_b[:k, :]
    
    # Reconstruct the image from the truncated matrices
    r_compressed = np.dot(U_r_k, np.dot(S_r_k, Vt_r_k))
    g_compressed = np.dot(U_g_k, np.dot(S_g_k, Vt_g_k))
    b_compressed = np.dot(U_b_k, np.dot(S_b_k, Vt_b_k))
    
    # Clip and convert to uint8 for image display
    r_compressed = np.clip(r_compressed, 0, 255).astype(np.uint8)
    g_compressed = np.clip(g_compressed, 0, 255).astype(np.uint8)
    b_compressed = np.clip(b_compressed, 0, 255).astype(np.uint8)
    
    # Stack the channels back into a single image array
    compressed_image = np.stack([r_compressed, g_compressed, b_compressed], axis=2)
    
    return compressed_image, r_compressed, g_compressed, b_compressed, (U_r_k, S_r_k, Vt_r_k), (U_g_k, S_g_k, Vt_g_k), (U_b_k, S_b_k, Vt_b_k)

def calculate_compression_ratio(original_shape, k, U_r_k, S_r_k, Vt_r_k, U_g_k, S_g_k, Vt_g_k, U_b_k, S_b_k, Vt_b_k):
    """
    Calculate compression ratio based on original and compressed data sizes.
    
    Args:
        original_shape (tuple): Shape of original image (height, width, 3).
        k (int): Number of singular values used.
        U_r_k, S_r_k, Vt_r_k, etc.: SVD components for each channel.
    
    Returns:
        float: Compression ratio (original size / compressed size).
    """
    height, width, channels = original_shape
    original_size = height * width * channels
    
    # Compressed size is the sum of the elements in the U, S, and V matrices for each channel
    compressed_size = 0
    for U_k, S_k, Vt_k in [(U_r_k, S_r_k, Vt_r_k), (U_g_k, S_g_k, Vt_g_k), (U_b_k, S_b_k, Vt_b_k)]:
        compressed_size += U_k.size + S_k.diagonal().size + Vt_k.size
    
    return original_size / compressed_size if compressed_size > 0 else float('inf')

def calculate_psnr(original_image, compressed_image):
    """
    Calculate PSNR between original and compressed images.
    
    Args:
        original_image (np.ndarray): Original image array.
        compressed_image (np.ndarray): Compressed image array.
    
    Returns:
        float: PSNR value in decibels (dB).
    """
    mse = np.mean((original_image.astype(np.float32) - compressed_image.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if images are identical
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def test_compression(image_path, k_values=[10, 50, 100]):
    """
    Test compression with different k values, compute metrics, and display results.
    
    Args:
        image_path (str): Path to input image.
        k_values (list): List of k values to test.
    
    Returns:
        list: List of dictionaries with k, compressed_path, compression_ratio, psnr.
    """
    os.makedirs("uploads", exist_ok=True)
    img_array, r_channel, g_channel, b_channel = load_and_preprocess_image(image_path)
    
    results = []
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(k_values) + 1, 1)
    plt.title("Original Image")
    plt.imshow(img_array.astype(np.uint8))
    plt.axis('off')
    
    for i, k in enumerate(k_values, 1):
        compressed_image, _, _, _, r_svd, g_svd, b_svd = compress_image_svd(r_channel, g_channel, b_channel, k)
        
        compressed_path = f"uploads/compressed_k{k}.jpg"
        Image.fromarray(compressed_image).save(compressed_path)
        
        compression_ratio = calculate_compression_ratio(img_array.shape, k, *r_svd, *g_svd, *b_svd)
        psnr = calculate_psnr(img_array, compressed_image)
        
        results.append({
            'k': k,
            'compressed_path': compressed_path,
            'compression_ratio': compression_ratio,
            'psnr': psnr
        })
        
        plt.subplot(1, len(k_values) + 1, i + 1)
        plt.title(f"k={k}\nRatio: {compression_ratio:.2f}\nPSNR: {psnr:.2f} dB")
        plt.imshow(compressed_image)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    # Ensure you have a sample image at this path or change it
    sample_image_path = "sample_image.jpg"
    if os.path.exists(sample_image_path):
        results = test_compression(sample_image_path, k_values=[10, 50, 100])
        for result in results:
            print(f"k={result['k']}: Compression Ratio={result['compression_ratio']:.2f}, PSNR={result['psnr']:.2f} dB")
    else:
        print(f"Sample image not found at '{sample_image_path}'. Please provide a valid image path.")

