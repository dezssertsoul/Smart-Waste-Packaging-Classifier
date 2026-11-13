# Proyek Klasifikasi Gambar Sampah (Garbage Classification)

Proyek ini membangun model *Deep Learning* berbasis *Convolutional Neural Network* (CNN) untuk mengklasifikasikan gambar sampah ke dalam 6 kategori utama (`cardboard`, `glass`, `metal`, `paper`, `plastic`, dan `trash`).

## 1. Dataset & Pembagian Data
* **Dataset:** Garbage Classification (6 Kelas, total ±2.527 gambar).
* **Pembagian Data (`split-folders`):**
  * **Train Set:** 80% (dilengkapi augmentasi *RandomFlip*, *RandomRotation*, *RandomZoom*, dan *RandomTranslation*).
  * **Validation Set:** 10%
  * **Test Set:** 10%

## 2. Arsitektur Model
Model dibangun menggunakan kerangka **`tf.keras.Sequential`** dengan pendekatan *Hybrid Transfer Learning*:
* **Input & Augmentasi:** Resolusi gambar `256x256` piksel RGB dengan lapisan augmentasi bawaan Keras.
* **Backbone Pre-trained:** `EfficientNetV2B0` (bobot ImageNet).
* **Lapisan Ekstraksi Tambahan:** `Conv2D(128, (3, 3))` dan `MaxPooling2D(pool_size=(2, 2), strides=1)`.
* **Klasifikasi Akhir:** `GlobalAveragePooling2D`, `Dropout`, `Dense(128, relu)`, dan lapisan keluaran `Dense(6, softmax)`.

## 3. Hasil Evaluasi
* **Validation Accuracy:** 94.82%
* **Test Accuracy:** **95.00%** (Macro F1-Score: 0.94, Weighted F1-Score: 0.95).

## 4. Struktur Direktori Submission
```text
├── notebook.ipynb          # Kode lengkap pelatihan, evaluasi, dan inferensi
├── README.md               # Dokumentasi proyek
├── requirements.txt        # Daftar pustaka dan versi yang digunakan
├── saved_model/            # Model dalam format TensorFlow SavedModel
├── tflite/                 # Model teroptimasi untuk mobile (model.tflite & label.txt)
└── tfjs_model/             # Model untuk web browser (model.json & file .bin)
```

## 5. Unduh Bobot Model TFLITE (*Model Weights*)
Karena ukuran file bobot model melebihi batas unggah langsung GitHub (>25 MB), seluruh file hasil ekspor model (`model.tflite`) disimpan di Google Drive:
- 📦 **[Download Model Weights (Google Drive)] https://drive.google.com/drive/folders/1NyKZdAs3oz7pIkomBeH9fI3tmYCig3wa?usp=sharing

## 6. Cara Menjalankan Inferensi (TF-Lite)
1. Instal seluruh dependensi melalui perintah:
   ```bash
   pip install -r requirements.txt
   ```
2. Muat file `tflite/model.tflite` menggunakan `tf.lite.Interpreter`, ubah ukuran gambar input menjadi `256x256` piksel (`float32`), lalu jalankan prediksi sesuai urutan indeks pada `tflite/label.txt`.
