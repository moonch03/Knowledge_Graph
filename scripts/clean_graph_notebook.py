import json
import os

def clean_graph_network_notebook():
    nb_path = r"C:\Users\USER\Graph\notebooks\Graph_Network.ipynb"
    output_path = r"C:\Users\USER\Graph\notebooks\Graph_Network_Clean.ipynb"
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    
    # Track if we've added the loading code
    added_noordin_loading = False
    added_montreal_loading = False

    for cell in nb.get('cells', []):
        source_text = "".join(cell.get('source', []))
        
        # Detect Noordin raw data cell
        if '{"type": "node", "id": "Abdul_Malik"' in source_text:
            if not added_noordin_loading:
                # Replace with loading code
                cell['source'] = [
                    "# Loading Noordin Top data from external file\n",
                    "with open('../data/noordin.jsonl', 'r', encoding='utf-8') as f:\n",
                    "    noordin_raw_data = f.read()\n",
                    "print(f'Loaded {len(noordin_raw_data.splitlines())} lines of Noordin data')"
                ]
                new_cells.append(cell)
                added_noordin_loading = True
            continue # Skip the rest of the raw data if it repeats
            
        # Detect Montreal raw data variable
        if 'montreal_raw_data = """' in source_text:
            if not added_montreal_loading:
                cell['source'] = [
                    "# Loading Montreal Gang data from external file\n",
                    "with open('../data/montreal.jsonl', 'r', encoding='utf-8') as f:\n",
                    "    montreal_raw_data = f.read()\n",
                    "print(f'Loaded {len(montreal_raw_data.splitlines())} lines of Montreal data')"
                ]
                new_cells.append(cell)
                added_montreal_loading = True
            continue

        # Fix the D: drive paths to be more generic or relative
        if 'pd.read_csv(r"D:\\My_Doc\\' in source_text:
            # We don't have these files locally yet, but let's make them placeholders
            cell['source'] = [line.replace('r"D:\\\\My_Doc\\\\graph_know\\\\Montreal\\\\', 'r"data/') for line in cell['source']]

        new_cells.append(cell)

    nb['cells'] = new_cells
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Cleaned notebook saved to {output_path}")

if __name__ == "__main__":
    clean_graph_network_notebook()
