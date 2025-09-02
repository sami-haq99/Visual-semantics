import os
import shutil

# === CONFIGURATION ===
text_file = [
            "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/mlt-lexical-test-sets/ende_test2016.txt",
            # "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/mlt-lexical-test-sets/ende_test2017mscoco.txt",
            # "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/mlt-lexical-test-sets/enfr_test2017mscoco.txt"
            ]             # text file containing image info
source_folders = [
                "/home/sami/mmt-eval/eval-datasets/flicker30k/flickr30k/Images",
               # "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/train2017-img",
                #"/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/val2017-img",
                #"/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/test2017-img"
                ]
                 # folder where images are stored
dest_folder = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/MDPI code-base/mlt-lexical-test"        # local folder to copy into

# Create destination folder if not exists
os.makedirs(dest_folder, exist_ok=True)

# Create destination folder if not exists
os.makedirs(dest_folder, exist_ok=True)


    # Read the text file line by line
for text_file in text_file:
    with open(text_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Remove the part after '#'
            full_filename = line.split(" | ")[-1].strip()

            # Extract only the numeric filename part
            filename = full_filename.split("_")[-1]

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
