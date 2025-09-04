#!/usr/bin/env bash

#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --nodelist=g122
#SBATCH -p compute
#SBATCH -J captions
#SBATCH -t 23:59:59
#SBATCH -o comet-%j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster

source /home/shaq/image-caption/env-ic/bin/activate

root_dir="multi30k-2016/"
image_name_file="${root_dir}image_test_2016_flickr.txt"
output_file="${root_dir}"
language=("cs")  # Options: de, fr, cs, fi, ro, zh
source_file="${root_dir}test_2016_flickr.cs"
image_dir="${root_dir}images/"

for lang in "${language[@]}"; do
    python gemma-3-12b-trans.py $image_name_file $output_file $lang $image_dir $source_file
done
