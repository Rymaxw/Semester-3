import json
import os

notebook_path = r"c:\Users\jmjur\Documents\IPSD\PSO TUBES\prediksisaham.ipynb"
output_path = r"c:\Users\jmjur\Documents\IPSD\PSO TUBES\notebook_code.py"

def export_code():
    if not os.path.exists(notebook_path):
        print(f"Error: File not found at {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, cell in enumerate(data['cells']):
            if cell['cell_type'] == 'code':
                f.write(f"\n\n# Cell {i}\n")
                source = "".join(cell['source'])
                f.write(source)

    print(f"Exported code to {output_path}")

if __name__ == "__main__":
    export_code()
