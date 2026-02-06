# Laporan Tugas Klasifikasi Data Heart Disease

## Deskripsi Masalah

Masalah yang diangkat dalam tugas ini adalah klasifikasi penyakit jantung (*heart disease*) berdasarkan berbagai fitur klinis pasien. Dataset yang digunakan adalah `heart_desease.xlsx`. Tujuan utamanya adalah membangun model *machine learning* yang dapat memprediksi apakah seseorang memiliki indikasi penyakit jantung atau tidak (target variabel: `num_predicted`).

Dalam eksperimen ini, kami membandingkan performa dari lima algoritma klasifikasi yang berbeda untuk menentukan mana yang memberikan hasil prediksi terbaik. Kelima algoritma tersebut adalah:

1.  **Support Vector Machine (SVM)**: Algoritma yang bekerja dengan mencari *hyperplane* terbaik yang memisahkan dua kelas data dengan margin maksimal. SVM sangat efektif pada ruang dimensi tinggi.
2.  **Gaussian Naive Bayes**: Metode klasifikasi probabilistik yang didasarkan pada Teorema Bayes dengan asumsi bahwa fitur-fitur bersifat independen satu sama lain (*naive*).
3.  **Decision Tree**: Algoritma berbentuk struktur pohon di mana setiap *node* internal merepresentasikan tes pada atribut, setiap cabang merepresentasikan hasil tes, dan setiap *leaf node* merepresentasikan kelas target.
4.  **Random Forest**: Metode *ensemble* yang membangun banyak *decision trees* pada saat pelatihan dan menghasilkan kelas yang merupakan modus (paling sering muncul) dari kelas-kelas yang dihasilkan oleh pohon-pohon individu.
5.  **Multi-layer Perceptron (Neural Network)**: Salah satu arsitektur jaringan saraf tiruan (*artificial neural network*) sederhana yang terdiri dari setidaknya tiga lapisan *node*: *input layer*, *hidden layer*, dan *output layer*.

## Exploratory Data Analysis (EDA)

Tahap *Exploratory Data Analysis* (EDA) dilakukan untuk memahami karakteristik data sebelum dilakukan pemrosesan lebih lanjut.

1.  **Pemeriksaan Dataset**: Dataset terdiri dari 14 atribut, di mana `num_predicted` adalah variabel target (0 = sehat, 1 = sakit). Tipe data awal bervariasi antara *integer*, *float*, dan *object*.
2.  **Penanganan Nilai '?'**: Ditemukan bahwa beberapa kolom seperti `ca`, `thal`, dan `slope` mengandung nilai placeholder '?' yang bukan merupakan format numerik standar. Nilai ini kemudian dikonversi menjadi `NaN` agar dapat dihitung jumlah *missing values*-nya.
3.  **Analisis Missing Values**:
    -   Kolom `ca` memiliki *missing values* tertinggi (291 data), diikuti oleh `thal` (266 data) dan `slope` (190 data).
    -   Kolom lain seperti `chol`, `fbs`, `trestbps`, `restecg`, `thalach`, dan `exang` juga memiliki beberapa data yang hilang namun dalam jumlah yang jauh lebih sedikit.
    -   Informasi ini krusial untuk menentukan strategi imputasi yang tepat (menggunakan median) pada tahap *Preprocessing*.
4.  **Distribusi Kelas Target**: Visualisasi dilakukan menggunakan *countplot* untuk melihat sebaran data pasien yang sehat dan sakit. Dari statistik deskriptif, terlihat rata-rata `num_predicted` adalah 0.36, yang mengindikasikan bahwa sekitar 36% pasien dalam dataset didiagnosis memiliki penyakit jantung (kelas 1), sementara sisanya (64%) adalah sehat (kelas 0). Proporsi ini menunjukkan dataset yang sedikit *imbalanced* namun masih cukup wajar untuk diklasifikasikan tanpa teknik *sampling* khusus.

## Preprocessing Data

Sebelum melakukan pemodelan, data mentah diproses terlebih dahulu:
-   **Handling Missing Values**: Nilai '?' pada dataset diganti dengan `NaN`, kemudian diisi (*imputation*) menggunakan nilai median dari masing-masing kolom.
-   **Scaling**: Fitur-fitur numerik dinormalisasi menggunakan `StandardScaler` agar memiliki skala yang seragam, yang penting untuk performa model seperti SVM dan Neural Network.
-   **Data Splitting**: Data dibagi menjadi *training set* (80%) dan *test set* (20%) dengan `random_state=42` untuk menjaga konsistensi hasil.

## Modeling

Berikut adalah penjelasan singkat dan hasil evaluasi dari masing-masing model yang digunakan.

### 1. Support Vector Machine (SVM)
**Konsep**: SVM mencoba menemukan batas pemisah (*decision boundary*) yang optimal antar kelas. Kami menggunakan kernel linear untuk percobaan ini.

**Hasil Evaluasi**:
-   **Accuracy**: 85%
-   **Precision (Class 0 / Sehat)**: 0.87
-   **Recall (Class 0 / Sehat)**: 0.89
-   **Precision (Class 1 / Sakit)**: 0.79
-   **Recall (Class 1 / Sakit)**: 0.76

Model SVM menunjukkan performa yang cukup baik dan seimbang dalam memprediksi kedua kelas.

### 2. Gaussian Naive Bayes
**Konsep**: Model ini menghitung probabilitas posterior untuk setiap kelas dan memilih kelas dengan probabilitas tertinggi. Asumsi utamanya adalah distribusi data mengikuti distribusi Gaussian (normal).

**Hasil Evaluasi**:
-   **Accuracy**: 85%
-   **Precision (Class 0)**: 0.87
-   **Recall (Class 0)**: 0.89
-   **Precision (Class 1)**: 0.79
-   **Recall (Class 1)**: 0.76

Menariknya, hasil Naive Bayes pada dataset ini identik dengan SVM, menunjukkan bahwa asumsi independensi fitur mungkin cukup valid atau distribusi data cukup normal.

### 3. Decision Tree
**Konsep**: Model ini memecah data menjadi himpunan-himpunan bagian yang lebih kecil berdasarkan aturan keputusan yang semakin spesifik hingga mencapai prediksi akhir.

**Hasil Evaluasi**:
-   **Accuracy**: 75%
-   **Precision (Class 0)**: 0.81
-   **Recall (Class 0)**: 0.79
-   **Precision (Class 1)**: 0.64
-   **Recall (Class 1)**: 0.67

Performa Decision Tree tunggal terlihat paling rendah dibandingkan model lain, kemungkin karena kecenderungan *overfitting* pada data latih.

### 4. Random Forest
**Konsep**: Menggabungkan prediksi dari banyak *decision trees* untuk mengurangi varians dan risiko *overfitting*.

**Hasil Evaluasi**:
-   **Accuracy**: 86%
-   **Precision (Class 0)**: 0.89
-   **Recall (Class 0)**: 0.89
-   **Precision (Class 1)**: 0.81
-   **Recall (Class 1)**: 0.81

Random Forest memberikan akurasi tertinggi (86%) dan keseimbangan terbaik antara *precision* dan *recall* untuk kedua kelas.

### 5. Neural Network (MLP Classifier)
**Konsep**: Model ini belajar memetakan input ke output melalui lapisan-lapisan *neuron* tersembunyi (*hidden layers*) dengan fungsi aktivasi non-linear.

**Hasil Evaluasi**:
-   **Accuracy**: 81%
-   **Precision (Class 0)**: 0.89
-   **Recall (Class 0)**: 0.82
-   **Precision (Class 1)**: 0.71
-   **Recall (Class 1)**: 0.81

Meskipun cukup baik, Neural Network sedikit di bawah Random Forest dan SVM dalam hal akurasi pada dataset dan konfigurasi ini.

## Kesimpulan
Berdasarkan eksperimen ini, **Random Forest** terbukti menjadi model terbaik untuk dataset penyakit jantung ini dengan akurasi 86%, diikuti oleh SVM dan Naive Bayes (85%). Decision Tree memiliki performa terendah (75%).
