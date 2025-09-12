#!/usr/bin/env bash

#SBATCH --gres=gpu:a100:1
#SBATCH -p compute
#SBATCH -J captions
#SBATCH -t 23:59:59
#SBATCH -o comet-%j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster

source /home/shaq/image-caption/env-ic/bin/activate

root_dir="../COMUTE/"

language=("cs" "de" "fr")  # Options: de, fr, cs, fi, ro, zh


for lang in "${language[@]}"; do
    source_file="${root_dir}en-${lang}/src.en"
    output_file="${root_dir}en-${lang}/"
    
    python aya-23-8b-trans.py $output_file $lang $source_file
done
