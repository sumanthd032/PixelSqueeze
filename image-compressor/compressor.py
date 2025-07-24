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
    U_r, S_r, Vt_r = np.linalg.svd(r_channel, full_matrices=False)
    U_g, S_g, Vt_g = np.linalg.svd(g_channel, full_matrices=False)
    U_b, S_b, Vt_b = np.linalg.svd(b_channel, full_matrices=False)
    
    U_r_k = U_r[:, :k]
    S_r_k = np.diag(S_r[:k])
    Vt_r_k = Vt_r[:k, :]
    
    U_g_k = U_g[:, :k]
    S_g_k = np.diag(S_g[:k])
    Vt_g_k = Vt_g[:k, :]
    
    U_b_k = U_b[:, :k]
    S_b_k = np.diag(S_b[:k])
    Vt_b_k = Vt_b[:k, :]
    
    r_compressed = np.dot(U_r_k, np.dot(S_r_k, Vt_r_k))
    g_compressed = np.dot(U_g_k, np.dot(S_g_k, Vt_g_k))
    b_compressed = np.dot(U_b_k, np.dot(S_b_k, Vt_b_k))
    
    r_compressed = np.clip(r_compressed, 0, 255).astype(np.uint8)
    g_compressed = np.clip(g_compressed, 0, 255).astype(np.uint8)
    b_compressed = np.clip(b_compressed, 0, 255).astype(np.uint8)
    
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
    original_size = height * width * channels  # Total pixels in original image
    
    # Compressed size: sum of elements in U_k, S_k, Vt_k for each channel
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
    mse = np.mean((original_image - compressed_image) ** 2)
    if mse == 0:
        return float('inf')
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
    os.makedirs("frontend/static/uploads", exist_ok=True)
    img_array, r_channel, g_channel, b_channel = load_and_preprocess_image(image_path)
    
    results = []
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(k_values) + 1, 1)
    plt.title("Original Image")
    plt.imshow(img_array.astype(np.uint8))
    plt.axis('off')
    
    for i, k in enumerate(k_values, 1):
        compressed_image, _, _, _, r_svd, g_svd, b_svd = compress_image_svd(r_channel, g_channel, b_channel, k)
        
        compressed_path = f"frontend/static/uploads/compressed_k{k}.jpg"
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
    sample_image_path = "/home/sumanthd032/Projects/PixelSqueeze/image-compressor/sample_image.jpg"
    results = test_compression(sample_image_path, k_values=[10, 50, 100])
    for result in results:
        print(f"k={result['k']}: Compression Ratio={result['compression_ratio']:.2f}, PSNR={result['psnr']:.2f} dB")