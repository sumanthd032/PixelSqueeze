import numpy as np
from PIL import Image
import os
import io

# --- NEW: Import the requests library to handle URLs ---
import requests

def load_and_preprocess_image(image_source):
    """
    Load an RGB image from a path, file stream, or URL and convert it to NumPy arrays.
    
    Args:
        image_source (str or file-like object): Path, stream, or URL of the input image.
    
    Returns:
        tuple: Original image array, R, G, B channel matrices (2D).
    """
    # --- FIX: Added logic to handle remote URLs ---
    if isinstance(image_source, str) and image_source.startswith(('http://', 'https://')):
        try:
            # Use requests to get the image from the URL
            response = requests.get(image_source)
            response.raise_for_status()  # Raise an exception for bad status codes (like 404)
            
            # Read the image content from the response into an in-memory buffer
            image_bytes = io.BytesIO(response.content)
            img = Image.open(image_bytes).convert('RGB')
        except requests.exceptions.RequestException as e:
            # Handle network errors gracefully
            raise IOError(f"Could not fetch image from URL: {e}")
    else:
        # This handles local file paths and file-like objects for local testing
        img = Image.open(image_source).convert('RGB')
    # --- End FIX ---
    
    img_array = np.array(img, dtype=np.float32)
    
    # Separate the channels
    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1]
    b_channel = img_array[:, :, 2]
    
    # Clipping values is good practice but often not strictly necessary with float32
    r_channel = np.clip(r_channel, 0, 255)
    g_channel = np.clip(g_channel, 0, 255)
    b_channel = np.clip(b_channel, 0, 255)
    
    return img_array, r_channel, g_channel, b_channel

def compress_image_svd(r_channel, g_channel, b_channel, k):
    """
    Compress image channels using SVD with k singular values.
    This function remains the same as your version.
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
    
    # The return signature matches what app.py expects
    return compressed_image, r_compressed, g_compressed, b_compressed, (U_r_k, S_r_k, Vt_r_k), (U_g_k, S_g_k, Vt_g_k), (U_b_k, S_b_k, Vt_b_k)

def calculate_compression_ratio(original_shape, k, U_r_k, S_r_k, Vt_r_k, U_g_k, S_g_k, Vt_g_k, U_b_k, S_b_k, Vt_b_k):
    """
    Calculate compression ratio based on original and compressed data sizes.
    This function remains the same as your version.
    """
    height, width, channels = original_shape
    original_size = height * width * channels
    
    compressed_size = 0
    for U_k, S_k, Vt_k in [(U_r_k, S_r_k, Vt_r_k), (U_g_k, S_g_k, Vt_g_k), (U_b_k, S_b_k, Vt_b_k)]:
        compressed_size += U_k.size + S_k.diagonal().size + Vt_k.size
    
    return original_size / compressed_size if compressed_size > 0 else float('inf')

def calculate_psnr(original_image, compressed_image):
    """
    Calculate PSNR between original and compressed images.
    This function remains the same as your version.
    """
    mse = np.mean((original_image.astype(np.float32) - compressed_image.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
