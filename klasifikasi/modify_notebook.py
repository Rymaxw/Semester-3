
import json
import os

notebook_path = r'c:\Users\jmjur\Documents\IPSD\klasifikasi\heart.ipynb'

new_cells = [
   {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
     "# Kesimpulan\n",
     "\n",
     "Berdasarkan hasil evaluasi, **Support Vector Machine (SVM)** terpilih sebagai model terbaik dengan akurasi 88%. Model ini menunjukkan keseimbangan yang baik antara precision dan recall.\n",
     "\n",
     "Oleh karena itu, kita akan menyimpan model SVM ini untuk digunakan dalam prediksi di masa depan."
    ]
   },
   {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
     "import joblib\n",
     "\n",
     "# Menyimpan model terbaik (SVM)\n",
     "joblib.dump(svm_model, 'best_heart_disease_model_svm.pkl')\n",
     "print(\"Model SVM terbaik berhasil disimpan sebagai 'best_heart_disease_model_svm.pkl'\")"
    ]
   }
]

with open(notebook_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Append new cells
data['cells'].extend(new_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)

print("Notebook updated successfully.")
