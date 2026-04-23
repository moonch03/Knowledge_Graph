import json
import os

def extract_data():
    nb_path = r"C:\Users\USER\Graph\notebooks\Graph_Network.ipynb"
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    noordin_data = []
    montreal_data = []

    for cell in nb.get('cells', []):
        source = "".join(cell.get('source', []))
        
        # Look for Noordin Top lines
        if '{"type": "node", "id": "Abdul_Malik"' in source:
            lines = source.split('\n')
            for line in lines:
                if line.strip().startswith('{"type":'):
                    noordin_data.append(line.strip())
        
        # Look for Montreal data
        if 'montreal_raw_data = """' in source:
            # Extract content between triple quotes
            import re
            match = re.search(r'montreal_raw_data = """(.*?)"""', source, re.DOTALL)
            if match:
                data_str = match.group(1).strip()
                montreal_data.extend([l.strip() for l in data_str.split('\n') if l.strip()])

    if noordin_data:
        with open(r"C:\Users\USER\Graph\data\noordin.jsonl", "w", encoding='utf-8') as f:
            f.write("\n".join(noordin_data))
        print(f"Extracted {len(noordin_data)} lines for Noordin")

    if montreal_data:
        with open(r"C:\Users\USER\Graph\data\montreal.jsonl", "w", encoding='utf-8') as f:
            f.write("\n".join(montreal_data))
        print(f"Extracted {len(montreal_data)} lines for Montreal")

if __name__ == "__main__":
    extract_data()
