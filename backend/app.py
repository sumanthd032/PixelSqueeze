from flask import Flask, request, render_template
import os
import sys
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64

# Add parent directory to path to find the compressor module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compressor import load_and_preprocess_image, compress_image_svd, calculate_compression_ratio, calculate_psnr

# Initialize Flask app
app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), '../frontend/templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '../frontend/static'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/compress', methods=['GET', 'POST'])
def compress():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('compress.html', error="No image file uploaded. Please select a file.")
        
        file = request.files['image']
        if file.filename == '':
            return render_template('compress.html', error="No image file selected. Please choose a file.")
        
        if file:
            try:
                # Read the uploaded file into memory
                original_image_bytes = file.read()
                
                # Process image from the in-memory bytes
                img_array, r_channel, g_channel, b_channel = load_and_preprocess_image(io.BytesIO(original_image_bytes))
            except Exception as e:
                return render_template('compress.html', error=f"Error loading image. It might be corrupted or in an unsupported format. Details: {str(e)}")
            
            k_value_str = request.form.get('k_value', '50')
            try:
                k_value = int(k_value_str)
                max_k = min(img_array.shape[0], img_array.shape[1])
                if not (0 < k_value <= max_k):
                    raise ValueError(f"Quality (k) must be between 1 and {max_k} for this image.")
            except ValueError as e:
                return render_template('compress.html', error=str(e) or "Invalid k value. Please enter a positive integer.")

            # --- VERCEL FIX: Process and serve images from memory using Base64 ---

            # 1. Compress the image
            compressed_image_array, _, _, _, r_svd, g_svd, b_svd = compress_image_svd(r_channel, g_channel, b_channel, k_value)
            
            # 2. Convert compressed numpy array to an image and save to a memory buffer
            compressed_pil_image = Image.fromarray(compressed_image_array)
            buffer = io.BytesIO()
            # Determine image format from original filename
            file_format = file.filename.split('.')[-1].upper()
            if file_format not in ['JPEG', 'PNG', 'WEBP']:
                file_format = 'JPEG' # Default to JPEG
            compressed_pil_image.save(buffer, format=file_format)
            compressed_image_bytes = buffer.getvalue()

            # 3. Encode both original and compressed images as Base64 strings
            original_base64 = base64.b64encode(original_image_bytes).decode('utf-8')
            compressed_base64 = base64.b64encode(compressed_image_bytes).decode('utf-8')
            
            # 4. Create data URIs for embedding in the HTML
            mime_type = f"image/{file_format.lower()}"
            original_data_uri = f"data:{mime_type};base64,{original_base64}"
            compressed_data_uri = f"data:{mime_type};base64,{compressed_base64}"

            # Calculate Metrics
            compression_ratio = calculate_compression_ratio(img_array.shape, k_value, *r_svd, *g_svd, *b_svd)
            psnr = calculate_psnr(img_array, compressed_image_array)
            
            results = [{
                'k': k_value,
                'compressed_path': compressed_data_uri,
                'compression_ratio': f"{compression_ratio:.2f}",
                'psnr': f"{psnr:.2f}"
            }]
            
            return render_template('compress.html', results=results, original_path=original_data_uri)

    return render_template('compress.html', results=None)

if __name__ == '__main__':
    app.run(debug=True)