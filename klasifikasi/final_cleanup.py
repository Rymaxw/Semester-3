
import os

files_to_delete = [
    r"c:\Users\jmjur\Documents\IPSD\klasifikasi\extract_structure.py",
    r"c:\Users\jmjur\Documents\IPSD\klasifikasi\notebook_structure.txt"
]

for file_path in files_to_delete:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"Not found: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
