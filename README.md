# PixelSqueeze

![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Cloudinary](https://img.shields.io/badge/Cloudinary-3448C5?style=for-the-badge&logo=cloudinary&logoColor=white)
[![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://pixelsqueeze-live.vercel.app)

PixelSqueeze is an innovative web application that leverages **Singular Value Decomposition (SVD)** to efficiently compress digital images while preserving quality. Built with **Flask and Python**, it allows users to upload images, adjust compression levels using a customizable `k` value, and download the compressed results with metrics like **compression ratio** and **Peak Signal-to-Noise Ratio (PSNR)**.

This project is ideal for optimizing images for web use, reducing storage needs, or experimenting with SVD-based image processing.

---

## Key Features

* **Customizable Compression**: Adjust the `k` value (e.g., 10, 50, 100, 200) to control compression quality and file size.
* **Real-Time Metrics**: View compression ratio and PSNR to evaluate the trade-off between size and quality.
* **Instant Downloads**: Download compressed images directly from the browser.
* **Responsive Design**: Fully functional on desktop and mobile devices with a modern UI.

---

## How to Use

### Prerequisites

* Python 3.9 or higher
* Git (for cloning the repository)

### Installation

#### Clone the Repository

```bash
git clone https://github.com/sumanthd032/PixelSqueeze.git
cd PixelSqueeze
```

#### Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application Locally

Navigate to the `backend/` directory and start the Flask server:

```bash
python app.py
```

Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000) to access the app.

---

## Usage Instructions

### Home Page

* Visit the homepage to learn about PixelSqueeze and its features.
* Click **"Get Started"** to navigate to the compression page.

### Compress an Image

1. Go to the `/compress` route.
2. Upload an image (JPEG or PNG) using the file input.
3. Select a `k` value to adjust compression quality (lower values = higher compression, higher values = better quality).
4. Click **"Compress"** to process the image.

### View Results

* After compression, view the original and compressed images side by side.
* Check the compression ratio and PSNR metrics.
* Download the compressed image using the **"Download"** button.

### Learn More

Visit the `/how-it-works` page to understand the SVD process and PixelSqueeze workflow.

---

## Supported Image Formats

* JPEG
* PNG
* WEBP

---

## Notes

* The app uses a temporary directory for uploads, so files are not persisted after the session.
* For large images, ensure your system has sufficient memory to handle SVD computations.


