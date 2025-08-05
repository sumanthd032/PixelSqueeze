from flask import Flask, request, render_template, jsonify
import os
import sys
from werkzeug.utils import secure_filename
from PIL import Image
import io
import time

# --- Cloudinary Integration ---
import cloudinary
import cloudinary.uploader
import cloudinary.api
# --- End Cloudinary Integration ---

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compressor import load_and_preprocess_image, compress_image_svd, calculate_compression_ratio, calculate_psnr

# Initialize Flask app
app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), '../frontend/templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '../frontend/static'))

# --- Cloudinary Configuration ---
# This will automatically use the environment variables on Vercel
cloudinary.config(
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"), 
  api_key = os.getenv("CLOUDINARY_API_KEY"), 
  api_secret = os.getenv("CLOUDINARY_API_SECRET"),
  secure = True
)
# --- End Cloudinary Configuration ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

# --- NEW: API Route to get a signature for direct client-side upload ---
@app.route('/api/get-upload-signature', methods=['POST'])
def get_upload_signature():
    try:
        # Generate a timestamp for the signature
        timestamp = int(time.time())
        # Define the parameters for the signature
        params_to_sign = {
            'timestamp': timestamp,
            'folder': 'pixelsqueeze/originals' # Optional: specify a folder
        }
        # Generate the signature using your API secret
        signature = cloudinary.utils.api_sign_request(params_to_sign, os.getenv("CLOUDINARY_API_SECRET"))
        
        return jsonify({
            'signature': signature,
            'timestamp': timestamp,
            'api_key': os.getenv("CLOUDINARY_API_KEY"),
            'cloud_name': os.getenv("CLOUDINARY_CLOUD_NAME")
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/compress', methods=['GET', 'POST'])
def compress():
    if request.method == 'POST':
        # The form now submits JSON data instead of a file
        data = request.get_json()
        original_url = data.get('imageUrl')
        k_value_str = data.get('kValue', '50')

        if not original_url:
            return render_template('compress.html', error="Missing image URL. Please try uploading again.")

        try:
            # --- Use the Cloudinary URL to process the image ---
            # Cloudinary can fetch the image directly from its own URL
            img_array, r_channel, g_channel, b_channel = load_and_preprocess_image(original_url)
            
            k_value = int(k_value_str)
            max_k = min(img_array.shape[0], img_array.shape[1])
            if not (0 < k_value <= max_k):
                raise ValueError(f"Quality (k) must be between 1 and {max_k} for this image.")

            # Compress the image
            compressed_image_array, _, _, _, r_svd, g_svd, b_svd = compress_image_svd(r_channel, g_channel, b_channel, k_value)
            
            # --- Instead of uploading, we generate a new Cloudinary URL with transformations ---
            # This is much faster and more efficient.
            # We get the public_id from the original URL
            public_id = '/'.join(original_url.split('/')[-2:]).split('.')[0]
            
            # Create a new URL with the compression effect (simulated, as SVD is custom)
            # For a real app, we would upload the compressed version. Here we just show the concept.
            # Let's upload the compressed version for accuracy.
            compressed_pil_image = Image.fromarray(compressed_image_array)
            buffer = io.BytesIO()
            compressed_pil_image.save(buffer, format='JPEG')
            compressed_image_bytes = buffer.getvalue()

            compressed_upload_result = cloudinary.uploader.upload(
                io.BytesIO(compressed_image_bytes),
                folder="pixelsqueeze/compressed"
            )
            compressed_url = compressed_upload_result.get('secure_url')

            # Calculate Metrics
            compression_ratio = calculate_compression_ratio(img_array.shape, k_value, *r_svd, *g_svd, *b_svd)
            psnr = calculate_psnr(img_array, compressed_image_array)
            
            results = [{
                'k': k_value,
                'compressed_path': compressed_url,
                'compression_ratio': f"{compression_ratio:.2f}",
                'psnr': f"{psnr:.2f}"
            }]
            
            return render_template('compress.html', results=results, original_path=original_url)

        except Exception as e:
            return render_template('compress.html', error=f"An error occurred during compression: {str(e)}")

    # GET request just shows the page
    return render_template('compress.html', results=None)

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv() # For local testing, loads .env file
    app.run(debug=True)

