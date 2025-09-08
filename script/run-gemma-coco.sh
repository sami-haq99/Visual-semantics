#!/usr/bin/env bash

#SBATCH --gres=gpu:a100:1
#SBATCH -p compute
#SBATCH -J cap_far
#SBATCH -t 23:59:59
#SBATCH -o imagecap-%j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster

source /home/shaq/image-caption/env-ic/bin/activate

image_name_file="mscoco-test-2017/image_filenames.txt"
output_file="mscoco-test-2017/"
language=("de" "fr")  # Options: de, fr, cs, fi, ro, zh

image_dir="mscoco-test-2017/images/"

for lang in "${language[@]}"; do
    python ic-gemma3b.py $image_name_file $output_file $lang $image_dir
done
