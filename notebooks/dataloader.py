import os

def load_text_files(folder_path):
    data = {"pos": [], "neg": []}
    
    for subdir, _, files in os.walk(folder_path):
        if subdir.endswith("pos") or subdir.endswith("neg"):
            label = "pos" if subdir.endswith("pos") else "neg"
            for filename in files:
                if filename.endswith(".txt"):
                    file_path = os.path.join(subdir, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        data[label].append(text)
    
    return data

