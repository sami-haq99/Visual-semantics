import os
import shutil
import json
# === CONFIGURATION ===
json_file = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/MDPI code-base/thumb/mscoco_THumB-1.0.jsonl"             # text file containing image info

source_folders = ["/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/train2017-img",
                "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/val2017-img",
                "/home/sami/mmt-eval/eval-datasets/coco2017-images_testset/test2017-img"]
                 # folder where images are stored
dest_folder = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/MDPI code-base/thumb/images/"        # local folder to copy into

# Create destination folder if not exists
os.makedirs(dest_folder, exist_ok=True)

#save image names to a list but not dublicates
image_names = set()

with open(json_file, "r") as f:
        for line in f:
            record = json.loads(line)
            image_ = record["image"]
            # get part after the last underscore (with extension)
            image_filename = image_.split("_")[-1]
            # Try to find the file in the given source folders
            found = False
            for folder in source_folders:
                source_path = os.path.join(folder, image_filename)
                if os.path.exists(source_path):
                    shutil.copy(source_path, dest_folder)
                    #print(f"✅ Copied: {filename} from {folder}")
                    image_names.add(image_filename)
                    found = True
                    break  # stop after first match

            if not found:
                print(f"⚠️ File not found: {image_filename}")
        
        with open(dest_folder+"images.txt", "w") as out_f:
            for name in list(image_names):
                out_f.write(name + "\n")

