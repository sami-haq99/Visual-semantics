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

root_dir="../COMUTE/"

language=("cs" "de" "fr")  # Options: de, fr, cs, fi, ro, zh

image_dir="${root_dir}images/"

for lang in "${language[@]}"; do
    source_file="${root_dir}en-${lang}/src.en"
    image_name_file="${root_dir}en-${lang}/img.order"
    output_file="${root_dir}en-${lang}/"
    
    python gemma-3-12b-trans.py $image_name_file $output_file $lang $image_dir $source_file
done
