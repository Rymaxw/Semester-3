import json
import os

notebook_path = r"c:\Users\jmjur\Documents\IPSD\PSO TUBES\prediksisaham.ipynb"

def fix_notebook():
    if not os.path.exists(notebook_path):
        print(f"Error: File not found at {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified = False
    for i, cell in enumerate(data['cells']):
        if cell['cell_type'] == 'code':
            source_list = cell['source']
            source_str = "".join(source_list)
            
            if "hasil_black_swan = jalankan_simulasi_saham" in source_str:
                if "harga_akhir_bs =" in source_str:
                    print(f"Variable 'harga_akhir_bs' already defined in Cell {i}.")
                    continue
                
                print(f"Found target Cell {i}. Injecting variable definition.")
                
                new_source = []
                injected = False
                for line in source_list:
                    if "=== RINGKASAN STATISTIK: BLACK SWAN ===" in line and not injected:
                         new_source.append("\nharga_akhir_bs = hasil_black_swan[-1, :]\n")
                         injected = True
                    new_source.append(line)
                
                if injected:
                    cell['source'] = new_source
                    modified = True
                    print("Cell modified successfully.")
                    break
    
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Notebook saved.")
    else:
        print("Target cell not found or no changes needed.")

if __name__ == "__main__":
    fix_notebook()
