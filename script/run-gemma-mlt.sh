#!/usr/bin/env bash

#SBATCH --gres=gpu:rtxa6000:1
#SBATCH -p compute
#SBATCH -J cap_far
#SBATCH -t 23:59:59
#SBATCH -o imagecap-%j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster

source /home/shaq/image-caption/env-ic/bin/activate

image_name_file="mlt-lexical-test-sets/ende_test2017mscoco_images_path.txt"
output_file="mlt-lexical-test-sets/"
language=("de")  # Options: de, fr, cs, fi, ro, zh

image_dir="mlt-lexical-test-sets/images/"

for lang in "${language[@]}"; do
    python ic-gemma3b.py $image_name_file $output_file $lang $image_dir
done




