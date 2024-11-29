from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import random
from datetime import datetime
from werkzeug.utils import secure_filename
import os
import numpy as np
import joblib
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops # The function name is 'graycomatrix', not 'greycomatrix'
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Inisialisasi aplikasi Flask
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './static/uploads'  # Lokasi penyimpanan file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # Format file yang diizinkan
# Kamus pengguna: username sebagai kunci, dan tuple (password, role) sebagai nilai
users = {
    'admin': ('admin', 'admin'),
    'user1': ('password1', 'user'),  # Contoh pengguna biasa
    'user2': ('password2', 'user')  # Tambah lebih banyak jika diperlukan
}
app.secret_key = 'supersecretkey'

model = joblib.load('models/knn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le = joblib.load('models/label_encoder.pkl')

# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_glcm_features(image):
    glcm = graycomatrix(image, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
    ]
    return features

# Preprocessing
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))  # Resize to 128x128
    return resized

# Extract HSV Features
def extract_hsv_features(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv_img, axis=(0, 1))
    std_hsv = np.std(hsv_img, axis=(0, 1))
    return list(mean_hsv) + list(std_hsv)

# Feature Extraction Pipeline
def extract_features_with_hsv(images):
    features = []
    for img in images:
        gray_img = preprocess_image(img)
        glcm_features = extract_glcm_features(gray_img)
        hsv_features = extract_hsv_features(img)
        combined_features = glcm_features + hsv_features
        features.append(combined_features)
    return np.array(features)

# Rute untuk halaman utama
@app.route('/')
def home():
    if 'username' in session:
        role = session.get('role')
        if role == 'admin':
            return redirect(url_for('dashboard'))
        elif role == 'user':
            return redirect(url_for('site'))
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username][0] == password:
            session['username'] = username
            session['role'] = users[username][1]  # Simpan role di sesi
            if users[username][1] == 'admin':
                return redirect(url_for('dashboard'))
            elif users[username][1] == 'user':
                return redirect(url_for('site'))
        else:
            return 'Invalid credentials, please try again!'
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' in session and session.get('role') == 'admin':
        return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/site')
def site():
    if 'username' in session and session.get('role') == 'user':
        return render_template('site.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('home'))


# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)

        # Process the image and predict
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({"error": "Image not found!"})

        # Preprocess the image (resize and convert to grayscale)
        gray_img = preprocess_image(img)

        # Extract features (GLCM + HSV)
        glcm_features = extract_glcm_features(gray_img)
        hsv_features = extract_hsv_features(img)
        combined_features = glcm_features + hsv_features

        # Normalize the features using the same scaler
        combined_features = np.array(combined_features).reshape(1, -1)  # Make sure it's 2D for the model
        combined_features = scaler.transform(combined_features)

        kode_random = random.randint(00, 99)  # 6-digit random code
        tanggal_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        # Predict using the trained model
        predicted_label = model.predict(combined_features)
        print(predicted_label)
        # Decode the label
        predicted_class = le.inverse_transform(predicted_label)

        # Percabangan
        if predicted_class[0] == "bad":
            hasil = "Tidak Berkualitas"
        elif predicted_class[0] == "good":
            hasil = "Berkualitas"
        else:
            hasil = "Tidak bisa diidentifikasi"

        return render_template('predict.html', features=glcm_features, label=hasil, image_file=filename, kode_random=kode_random, tanggal_sekarang=tanggal_sekarang)

    return render_template('dashboard.html', label="Error")

# Fungsi untuk memeriksa format file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Rute untuk halaman utama
@app.route('/dashboard/train', methods=['GET', 'POST'])
def index():
    return render_template('train.html')

# Rute untuk mengunggah file
@app.route('/dashboard/train/upload', methods=['POST'])
def upload_file():
    # Memeriksa apakah ada file dalam permintaan
    if 'file' not in request.files:
        flash('No file part')  # Flash pesan kesalahan
        return redirect(request.url)
    
    file = request.files['file']
    
    # Jika tidak ada file yang dipilih
    if file.filename == '':
        flash('No selected file')  # Flash pesan kesalahan
        return redirect(request.url)

    # Memeriksa apakah input radio telah dipilih
    quality = request.form.get('quality')
    if not quality:
        flash('Please select the quality of the image.')  # Flash pesan kesalahan
        return redirect(request.url)
    
    # Memeriksa apakah file valid
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Mengamankan nama file
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], quality)
        
        # Membuat subfolder berdasarkan kualitas jika belum ada
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        file.save(os.path.join(save_path, filename))  # Simpan file
        flash(f'File successfully uploaded as {quality}')  # Flash pesan keberhasilan
        return redirect(url_for('index'))
    else:
        flash('Invalid file type')  # Flash pesan kesalahan tipe file
        return redirect(request.url)

FOLDER_BERKUALITAS = 'static/uploads/iya'
FOLDER_TIDAKBERKUALITAS = 'static/uploads/tidak'

@app.route('/dashboard/data')
def show_images():
    # Ambil daftar gambar dari folder berkualitas dan tidakberkualitas
    berkualitas_images = os.listdir(FOLDER_BERKUALITAS)
    tidakberkualitas_images = os.listdir(FOLDER_TIDAKBERKUALITAS)

    # Gabungkan semua gambar dengan statusnya
    images = []
    nomor = 1
    
    # Menambahkan gambar berkualitas dengan status 'Berkualitas'
    for img in berkualitas_images:
        images.append({
            'nomor': nomor,
            'subfolder': 'iya',
            'status': 'Berkualitas',
            'image': img
        })
        nomor += 1

    # Menambahkan gambar tidak berkualitas dengan status 'Tidak Berkualitas'
    for img in tidakberkualitas_images:
        images.append({
            'nomor': nomor,
            'subfolder': 'tidak',
            'status': 'Tidak Berkualitas',
            'image': img
        })
        nomor += 1

    return render_template('show.html', images=images)

# Menjalankan aplikasi
if __name__ == '__main__':
    # Membuat folder jika belum ada
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    port = int(os.getenv("PORT", 4000))  # Default port 5000 jika FLASK_RUN_PORT tidak diatur
    app.run(host='0.0.0.0', port=port)
