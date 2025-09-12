import os
import shutil

# === CONFIGURATION ===
text_file = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/MDPI code-base/multi30k-2016/image_test_2016_flickr.txt"             # text file containing image info
source_folders = ["/home/sami/mmt-eval/eval-datasets/flicker30k/flickr30k/Images"]
                 # folder where images are stored
dest_folder = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/MDPI code-base/multi30k-2016/images"        # local folder to copy into

# Create destination folder if not exists
os.makedirs(dest_folder, exist_ok=True)




    # Read the text file line by line
with open(text_file, "r") as f:
    for line in f:
        filename = line.strip()
        if not filename:
            continue  # skip empty lines

 

        # Try to find the file in the given source folders
        found = False
        for folder in source_folders:
            source_path = os.path.join(folder, filename)
            if os.path.exists(source_path):
                shutil.copy(source_path, dest_folder)
                #print(f"✅ Copied: {filename} from {folder}")
                found = True
                break  # stop after first match

        if not found:
            print(f"⚠️ File not found: {filename}")
