import os
import shutil

# === CONFIGURATION ===
text_files = [
            "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/mlt-lexical-test-sets/ende_test2016.txt",
            "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/mlt-lexical-test-sets/ende_test2017mscoco.txt",
            "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/mlt-lexical-test-sets/enfr_test2017mscoco.txt"
            ]             # text file containing image info
source_folders = [
                "/home/sami/mmt-eval/eval-datasets/flicker30k/flickr30k/Images",
               # "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/train2017-img",
                #"/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/val2017-img",
                #"/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/test2017-img"
                ]
                 # folder where images are stored
dest_folder = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/MDPI code-base/mlt-lexical-test-sets"        # local folder to copy into

# Create destination folder if not exists
os.makedirs(dest_folder, exist_ok=True)


def create_reference_file():    
    # Read the text file line by line
    for text_file in text_files:
        with open(text_file, "r") as f:
            name_text_file = os.path.basename(text_file).split(".")[0]
            output_ref_file = os.path.join(dest_folder, f"{name_text_file}.ref")
            with open(output_ref_file, "w") as out_f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue  # skip empty lines
                    line = line.split(" | ")[-2] # get the reference text
                    out_f.write(line + "\n")
        print(f"Reference file created: {output_ref_file}")


create_reference_file()