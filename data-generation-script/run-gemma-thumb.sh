#!/usr/bin/env bash

#SBATCH --gres=gpu:a600:1
#SBATCH --mem=100000
#SBATCH -p compute
#SBATCH -J cap_far
#SBATCH -t 23:59:59
#SBATCH -o imagecap-%j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster

source /home/shaq/image-caption/env-ic/bin/activate

image_name_file="../thumb/images/images.txt"
output_file="../thumb/"
language=("en")  # Options: de, fr, cs, fi, ro, zh

image_dir="../thumb/images/"

for lang in "${language[@]}"; do
    python gemma-cap-thumb.py $image_name_file $output_file $lang $image_dir
done



