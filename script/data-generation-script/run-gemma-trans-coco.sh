#!/usr/bin/env bash

#SBATCH --gres=gpu:a100:1
#SBATCH -p compute
#SBATCH -J translation
#SBATCH -t 23:59:59
#SBATCH -o %j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster

source /home/shaq/image-caption/env-ic/bin/activate

root_dir="mscoco-test-2017/"
image_name_file="${root_dir}image_filenames.txt"
output_file="${root_dir}"
language=("de" "fr")  # Options: de, fr, cs, fi, ro, zh
source_file="${root_dir}test_2017_mscoco.en"
image_dir="${root_dir}images/"

for lang in "${language[@]}"; do
    python gemma-3-12b-trans.py $image_name_file $output_file $lang $image_dir $source_file
done
