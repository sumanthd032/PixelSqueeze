from flask import Flask, request, render_template, send_from_directory
import os
import sys
from werkzeug.utils import secure_filename
from PIL import Image

# Add parent directory to path to find the compressor module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compressor import load_and_preprocess_image, compress_image_svd, calculate_compression_ratio, calculate_psnr

# Initialize Flask app
app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), '../frontend/templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '../frontend/static'))

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
            filename = secure_filename(file.filename)
            
            try:
                # Process image from memory stream for efficiency
                file.stream.seek(0) 
                img_array, r_channel, g_channel, b_channel = load_and_preprocess_image(file.stream)
            except Exception as e:
                return render_template('compress.html', error=f"Error loading image. It might be corrupted or in an unsupported format. Details: {str(e)}")
            
            # Robust k-value validation
            k_value_str = request.form.get('k_value', '50')
            try:
                k_value = int(k_value_str)
                max_k = min(img_array.shape[0], img_array.shape[1])
                if not (0 < k_value <= max_k):
                    raise ValueError(f"Quality (k) must be between 1 and {max_k} for this image.")
            except ValueError as e:
                return render_template('compress.html', error=str(e) or "Invalid k value. Please enter a positive integer.")

            # Save the original image only after validation is successful
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.stream.seek(0)
            file.save(original_path)

            # Compression Logic
            compressed_image, _, _, _, r_svd, g_svd, b_svd = compress_image_svd(r_channel, g_channel, b_channel, k_value)
            
            # Save the compressed image
            compressed_filename = f"compressed_k{k_value}_{filename}"
            compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], compressed_filename)
            Image.fromarray(compressed_image).save(compressed_path)
            
            # Calculate Metrics
            compression_ratio = calculate_compression_ratio(img_array.shape, k_value, *r_svd, *g_svd, *b_svd)
            psnr = calculate_psnr(img_array, compressed_image)
            
            results = [{
                'k': k_value,
                'compressed_path': f"/uploads/{compressed_filename}",
                'compression_ratio': f"{compression_ratio:.2f}",
                'psnr': f"{psnr:.2f}"
            }]
            
            return render_template('compress.html', results=results, original_path=f"/uploads/{filename}")

    return render_template('compress.html', results=None)

if __name__ == '__main__':
    app.run(debug=True)
