# Face Recognition Assignment

**Nama**: Bagus Cipta Pratama  
**NIM**: 23/516539/PA/22097

## Deskripsi
Proyek ini mengimplementasikan sistem deteksi dan pengenalan wajah (face detection & recognition) menggunakan metode Eigenface dan Support Vector Machine (SVM). Sistem dapat dijalankan dalam dua mode:

1. **Training & Evaluasi**: Memuat dataset wajah, melakukan preprocessing (deteksi & cropping, resize, flatten), pelatihan pipeline (mean centering → PCA → SVM), evaluasi pada test set, dan menyimpan model.
2. **Real-Time Recognition**: Menggunakan webcam untuk mendeteksi dan mengenali wajah secara langsung, menampilkan bounding box, label, dan skor kepercayaan.

## Struktur Direktori
```
face_recognition_project/
├─ images/                    # Dataset: subfolder per individu, minimal 10 gambar tiap folder
│   ├─ George_W_Bush/
│   ├─ Laura_Bush/
│   ├─ New_Person1/
│   └─ Your_Name/
├─ face_recognition.py        # Skrip utama untuk training & real-time recognition
├─ eigenface_pipeline.pkl     # Model pipeline yang disimpan setelah training          
└─ README.md                  # Petunjuk ini
```

## Instalasi
1. Clone repository ini:
   ```bash
   git clone https://github.com/username/face_recognition_project.git
   cd face_recognition_project
   ```
2. Buat virtual environment (opsional tetapi direkomendasikan):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Menjalankan
1. **Training & Evaluasi**
   ```bash
   python face_recognition.py
   ```
   - Jika file `eigenface_pipeline.pkl` belum ada, skrip akan otomatis melatih model pada dataset di folder `images/`.
   - Setelah selesai, classification report akan ditampilkan, dan model akan disimpan sebagai `eigenface_pipeline.pkl`.

2. **Real-Time Recognition**
   - Setelah training selesai atau saat file `eigenface_pipeline.pkl` sudah ada, skrip akan otomatis memulai mode webcam.
   - Jendela video akan menampilkan bounding box, label, dan skor pengenalan.
   - Tekan **`q`** untuk keluar.

## Result and Real Test
- Console menampilkan classification report (precision, recall, f1-score).
  
- Jendela video menampilkan deteksi dan pengenalan wajah secara real-time.
