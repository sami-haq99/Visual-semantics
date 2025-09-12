import os
import shutil

# === CONFIGURATION ===
text_file = [
            "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/mlt-lexical-test-sets/ende_test2016.txt",
             "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/mlt-lexical-test-sets/ende_test2017mscoco.txt",
             "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/mlt-lexical-test-sets/enfr_test2017mscoco.txt"
            ]             # text file containing image info

                 # folder where images are stored
dest_folder = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/MDPI code-base/mlt-lexical-test-sets/"        # local folder to copy into



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
            #save the filename to a text file
            #extract the text_file name without the path and extension
            text_file_name = os.path.splitext(os.path.basename(text_file))[0]
            #create a folder with the text_file name if it doesn't exist
            output_file_name = text_file_name + "_images_path.txt"
            with open(dest_folder + output_file_name, "a") as f:
                f.write(filename + "\n")