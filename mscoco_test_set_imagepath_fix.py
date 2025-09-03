import os
import shutil

# === CONFIGURATION ===
text_file = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/MDPI code-base/mscoco-test-2017/image_index.txt"             # text file containing image info
dest_folder = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/MDPI code-base/mscoco-test-2017"  # destination folder to copy images to
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
        #save the filenames in a file
        with open(os.path.join(dest_folder, "image_filenames.txt"), "a") as out_f:
            out_f.write(filename + "\n")
        # Try to find the file in the given source folders
 