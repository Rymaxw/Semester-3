# Penjelasan Lengkap Program Klasifikasi Penyakit Jantung (`heart.ipynb`)

Dokumen ini berisi penjelasan mendetail mengenai alur program, logika kode yang digunakan, serta cara membaca hasil dan grafik (visualisasi) yang ada di dalam file `heart.ipynb`. Penjelasan dibuat dengan bahasa yang mudah dipahami agar Anda mengerti sepenuhnya apa yang dikerjakan oleh program ini.

---

## 1. Tujuan Program
Program ini bertujuan untuk **memprediksi apakah seseorang terkena penyakit jantung atau tidak** berdasarkan data medis (seperti umur, tekanan darah, kolesterol, dll) menggunakan **Machine Learning**.

Program ini membandingkan 5 algoritma kecerdasan buatan (AI) yang berbeda untuk mencari mana yang paling pintar dan akurat dalam mendiagnosa.

---

## 2. Persiapan Data (Preprocessing & EDA)

Sebelum membuat model AI, data harus disiapkan dan "dibersihkan" terlebih dahulu.

### A. Exploratory Data Analysis (EDA) - Memahami Data
Di bagian awal, program membuat beberapa grafik untuk melihat karakteristik data pasien.

1.  **Grafik Distribusi Target (Bar Chart)**
    *   **Bentuk**: Diagram batang dengan dua warna.
    *   **Fungsi**: Melihat apakah jumlah pasien sakit jantung (`1`) seimbang dengan yang sehat (`0`).
    *   **Cara Baca**: Jika tingginya hampir sama, berarti data seimbang. Jika salah satu jauh lebih tinggi, AI mungkin akan bias ke kelas yang lebih banyak.

2.  **Heatmap Korelasi (Kotak-kotak Merah/Biru)**
    *   **Bentuk**: Tabel warna-warni dengan angka di dalamnya.
    *   **Fungsi**: Melihat hubungan antar fitur.
    *   **Cara Baca**:
        *   Warna **Merah/Panas (Angka mendekati 1)**: Hubungan kuat searah (misal: makin tua umur, makin tinggi tekanan darah).
        *   Warna **Biru/Dingin (Angka mendekati -1)**: Hubungan berlawanan.
        *   Warna **Pudat (Mendekati 0)**: Tidak ada hubungan.

3.  **Boxplot (Kotak dengan Garis)**
    *   **Fungsi**: Mencari **Outlier** (data aneh/ekstrem).
    *   **Cara Baca**: Titik-titik hitam di luar garis kotak adalah "data aneh" (misal: kolesterol 600, padahal rata-rata orang cuma 200). Ini penting diketahui karena bisa mengganggu belajar AI.

4.  **KDE Plot (Distribusi Umur & Kolesterol)**
    *   **Bentuk**: Grafik gunung/bukit yang tumpang tindih.
    *   **Fungsi**: Membandingkan sebaran data antara orang sakit vs sehat.
    *   **Cara Baca**: Jika "bukit" orang sakit (misal warna oranye) bergeser ke kanan dibanding orang sehat, artinya orang yang sakit rata-rata umurnya lebih tua atau kolesterolnya lebih tinggi.

### B. Membersihkan Data (Preprocessing)
Agar AI bisa belajar dengan baik, data harus bersih:

1.  **Mengganti Tanda '?'**: Data asli memiliki tanda tanya `?` untuk data yang hilang. Program mengubahnya menjadi `NaN` (Not a Number) agar dikenali komputer sebagai data kosong.
2.  **Median Imputation**: Data kosong tadi diisi dengan **nilai tengah (median)** dari kolom tersebut. Kenapa? Agar data tidak terbuang, dan median lebih aman daripada rata-rata jika ada nilai ekstrem.
3.  **Normalisasi (StandardScaler)**:
    *   **Masalah**: Umur rangenya 29-77, Kolesterol 100-500. Angkanya beda jauh.
    *   **Solusi**: Semua angka diubah ke skala yang sama (sekitar -3 sampai 3). Ini penting agar AI tidak menganggap Kolesterol lebih penting dari Umur hanya karena angkanya lebih besar.
4.  **Training & Testing Split**:
    *   Data dibagi menjadi dua: **80% untuk Latian (Training)** dan **20% untuk Ujian (Testing)**.
    *   AI belajar dari 80% data, lalu kita uji kepintarannya pakai 20% data sisanya yang belum pernah ia lihat.

---

## 3. Pemodelan (Modeling)

Kita menggunakan 5 "otak" buatan yang berbeda untuk belajar dari data ini. Berikut penjelasannya:

### 1. SVM (Support Vector Machine)
*   **Logika**: Membayangkan data sebagai titik-titik di ruangan, lalu mencari **garis atau tembok pemisah terbaik** yang membedakan pasien sakit dan sehat sejelas mungkin.
*   **Kernel RBF**: Tembok pemisahnya bisa melengkung fleksibel, tidak harus lurus kaku.

### 2. Naive Bayes
*   **Logika**: Menggunakan **matematika peluang (probabilitas)**. Dia menghitung "Berapa persen peluang orang sakit jantung JIKA kolesterolnya tinggi?". Metode ini sederhana tapi seringkali sangat efektif.

### 3. ID3 Decision Tree (Pohon Keputusan)
*   **Logika**: Membuat aturan seperti kuis "Ya/Tidak".
    *   *Contoh*: "Apakah Umur > 50?" -> (Ya) -> "Apakah Kolesterol > 250?" -> (Ya) -> SAKIT.
*   **Grafik Pohon (Tree Visualization)**:
    *   **Cara Baca**: Lihat kotak paling atas (akar). Itu adalah pertanyaan pertama yang paling penting untuk membedakan sakit/sehat. Ikuti panah ke bawah sampai ujung (daun) untuk dapat keputusannya.

### 4. Random Forest
*   **Logika**: Jika satu pohon (Decision Tree) bisa salah, mari buat **100 pohon** (Hutan). Kita tanya ke 100 pohon itu, lalu ambil **suara terbanyak (Voting)**. Ini biasanya jauh lebih akurat.
*   **Grafik Feature Importance**:
    *   **Fungsi**: Memberitahu kita faktor apa yang paling dipertimbangkan oleh AI.
    *   **Cara Baca**: Batang paling panjang adalah faktor penentu utama. (Misal: `thal` atau `cp` biasanya paling tinggi pengaruhnya terhadap penyakit jantung).

### 5. Neural Network (Jaringan Syaraf Tiruan)
*   **Logika**: Meniru cara kerja otak manusia dengan neuron-neuron buatan yang saling terhubung. Dia belajar pola rumit dengan cara "coba-coba dan koreksi kesalahan" berulang kali.
*   **Grafik Loss Curve (Garis Merah)**:
    *   **X-axis (Iterations)**: Berapa kali dia belajar.
    *   **Y-axis (Loss)**: Tingkat kesalahan/error.
    *   **Cara Baca**: Garis harus **turun drastis** lalu mendatar di bawah. Artinya AI makin lama makin pintar dan kesalahannya makin kecil. Jika garis naik turun tidak jelas, berarti AI gagal belajar.

---

## 4. Evaluasi Hasil (Cara Membaca Nilai)

Setiap model dinilai menggunakan 4 kriteria utama.

### A. Confusion Matrix (Kotak 4 Warna)
Ini adalah tabel rapor detail prediksi AI:
*   **Kotak Kiri-Atas (True Negative)**: Prediksi SEHAT, Ternyata BENAR SEHAT. (Bagus!)
*   **Kotak Kanan-Bawah (True Positive)**: Prediksi SAKIT, Ternyata BENAR SAKIT. (Bagus!)
*   **Kotak Kanan-Atas (False Positive)**: Prediksi SAKIT, Ternyata SEHAT. (Salah Lapor/Alarm Palsu).
*   **Kotak Kiri-Bawah (False Negative)**: Prediksi SEHAT, Ternyata SAKIT. (Bahaya! Orang sakit dibilang sehat, bisa telat diobati).

### B. Metrik Performa (Bar Chart)
1.  **Accuracy (Akurasi)**:
    *   Berapa persen tebakan yang benar secara keseluruhan?
    *   *Contoh 0.88 artinya 88% tebakannya benar.*
2.  **Precision (Presisi)**:
    *   Dari semua orang yang dibilang SAKIT oleh AI, berapa yang beneran sakit?
    *   *Penting agar orang sehat tidak panik divonis sakit.*
3.  **Recall (Daya Ingat)**:
    *   Dari semua orang yang SAKIT beneran di dunia nyata, berapa yang berhasil ditemukan AI?
    *   *Ini paling penting di dunia medis.* Kita tidak mau melewatkan orang sakit (False Negative).
4.  **F1-Score**:
    *   Nilai gabungan (rata-rata harmonis) antara Precision dan Recall. Angka ini menunjukkan keseimbangan model.

---

## 5. Kesimpulan Akhir
Di bagian akhir program, ada grafik batang yang membandingkan kelima model sekaligus.

*   Model dengan batang paling tinggi di semua warna (terutama metrik **Recall**) adalah model terbaik.
*   Berdasarkan hasil analisis, **SVM (RBF)** dan **Random Forest** biasanya menjadi juara di data ini karena mampu menangani pola rumit dengan baik.

Semoga penjelasan ini membantu Anda memahami seluruh isi program!
