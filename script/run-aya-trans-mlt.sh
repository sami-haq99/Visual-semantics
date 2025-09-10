#!/usr/bin/env bash

#SBATCH --gres=gpu:a100:1
#SBATCH -p compute
#SBATCH -J translation
#SBATCH -t 23:59:59
#SBATCH -o %j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster

source /home/shaq/image-caption/env-ic/bin/activate

root_dir="../mlt-lexical-test-sets/"
output_file="${root_dir}"
language=("de")  # Options: de, fr, cs, fi, ro, zh
source_file="${root_dir}ende_test2017mscoco.en"

for lang in "${language[@]}"; do
    python aya-23-8b-trans.py $output_file $lang $source_file
done

source_file="${root_dir}enfr_test2017mscoco.en"
language=("fr")
for lang in "${language[@]}"; do
    python aya-23-8b-trans.py $output_file $lang $source_file
done


