import json
import os
import re

def clean_notebook(file_path):
    print(f"Cleaning {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping {file_path} - Invalid JSON")
            return

    cleaned_cells = []
    filename = os.path.basename(file_path)

    for cell in nb.get('cells', []):
        # Skip empty cells
        if not cell.get('source'):
            continue
        
        # Clear outputs for code cells
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            
            source = cell['source']
            if isinstance(source, list):
                new_source = []
                for line in source:
                    # 1. Standardize pip install
                    if line.strip().startswith("pip install"):
                        line = "!" + line.lstrip()
                    
                    # 2. Fix Noordin_Top.ipynb download logic
                    if "wget -O tmp/terrorist_nodes.zip" in line:
                        # We'll replace this whole block in a bit if we detect it's the download cell
                        pass
                    
                    # 3. Fix Graph_json.ipynb syntax errors (JSON in code cells)
                    if filename == "Graph_json.ipynb":
                        if line.strip().startswith('{"type":'):
                            line = "# " + line
                    
                    new_source.append(line.rstrip() + '\n')
                
                # Specialized fix for the download block in Noordin_Top.ipynb
                full_source = "".join(new_source)
                if "wget -O tmp/terrorist_nodes.zip" in full_source and "subprocess.call" in full_source:
                    full_source = full_source.replace(
                        'subprocess.call(get_nodes_zip.split())',
                        'import urllib.request\nos.makedirs("tmp", exist_ok=True)\nurllib.request.urlretrieve(get_nodes_zip.split()[-1], "tmp/terrorist_nodes.zip")'
                    )
                    new_source = [l + '\n' for l in full_source.split('\n')]

                if new_source:
                    new_source[-1] = new_source[-1].rstrip('\n')
                cell['source'] = new_source

        cleaned_cells.append(cell)

    nb['cells'] = cleaned_cells
    
    # Minimize metadata
    nb['metadata'] = {
        "kernelspec": nb.get('metadata', {}).get('kernelspec', {}),
        "language_info": nb.get('metadata', {}).get('language_info', {})
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Done cleaning {file_path}")

def main():
    notebook_dir = r"C:\Users\USER\Graph\notebooks"
    for filename in os.listdir(notebook_dir):
        if filename.endswith(".ipynb"):
            clean_notebook(os.path.join(notebook_dir, filename))

if __name__ == "__main__":
    main()
