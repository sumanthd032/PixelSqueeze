from flask import Flask, request, render_template, send_from_directory
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
from compressor import load_and_preprocess_image, compress_image_svd, calculate_compression_ratio, calculate_psnr
from werkzeug.utils import secure_filename

# Initialize Flask app with custom template folder
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '../frontend/templates'))
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')  # Relative path to uploads folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure static folder
app.static_folder = os.path.join(os.path.dirname(__file__), '../frontend/static')  # Serve frontend static files

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', results=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/compress', methods=['POST'])
def compress():
    if 'image' not in request.files:
        return render_template('index.html', results=None, error="No image uploaded")
    
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', results=None, error="No image selected")
    
    if file:
        # Save uploaded image
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Load and preprocess image
        try:
            img_array, r_channel, g_channel, b_channel = load_and_preprocess_image(upload_path)
        except Exception as e:
            return render_template('index.html', results=None, error=f"Error loading image: {str(e)}")
        
        # Define k values for compression
        k_values = [10, 50, 100]
        results = []
        
        # Compress for each k
        for k in k_values:
            compressed_image, _, _, _, r_svd, g_svd, b_svd = compress_image_svd(r_channel, g_channel, b_channel, k)
            compressed_filename = f"compressed_k{k}_{filename}"
            compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], compressed_filename)
            from PIL import Image
            Image.fromarray(compressed_image).save(compressed_path)
            
            # Calculate metrics
            compression_ratio = calculate_compression_ratio(img_array.shape, k, *r_svd, *g_svd, *b_svd)
            psnr = calculate_psnr(img_array, compressed_image)
            
            results.append({
                'k': k,
                'compressed_path': f"/uploads/{compressed_filename}",
                'compression_ratio': f"{compression_ratio:.2f}",
                'psnr': f"{psnr:.2f}"
            })
        
        return render_template('index.html', results=results, original_path=f"/uploads/{filename}")
    
    return render_template('index.html', results=None, error="Invalid file")

if __name__ == '__main__':
    app.run(debug=True)