import os
import shutil

# === CONFIGURATION ===
text_file = "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/mscoco-2017-testset-text/image_index.txt"             # text file containing image info
source_folders = ["/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/train2017-img",
                "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/val2017-img",
                "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/test2017-img"]
                 # folder where images are stored
dest_folder = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/MDPI code-base/mscoco-test2017"        # local folder to copy into

# Create destination folder if not exists
os.makedirs(dest_folder, exist_ok=True)

# Create destination folder if not exists
os.makedirs(dest_folder, exist_ok=True)


    # Read the text file line by line
with open(text_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # skip empty lines

        # Remove the part after '#'
        clean_line = line.split("#")[0]

        # Extract filename (last part after underscore)
        # e.g. COCO_train2014_000000117071.jpg → 000000117071.jpg
        filename = clean_line.split("_")[-1]

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
