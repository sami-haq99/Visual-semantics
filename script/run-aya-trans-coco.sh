#!/usr/bin/env bash

#SBATCH --gres=gpu:rtxa6000:1
#SBATCH -p compute
#SBATCH -J translation
#SBATCH -t 23:59:59
#SBATCH -o %j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster

source /home/shaq/image-caption/env-ic/bin/activate

root_dir="mscoco-test-2017/"
output_file="${root_dir}"
language=("de" "fr")  # Options: de, fr, cs, fi, ro, zh
source_file="${root_dir}test_2017_mscoco.en"

for lang in "${language[@]}"; do
    python aya-23-8b-trans.py $output_file $lang $source_file
done
