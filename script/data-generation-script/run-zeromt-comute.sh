#!/usr/bin/env bash

#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100000
#SBATCH -p compute
#SBATCH -J translation
#SBATCH -t 23:59:59
#SBATCH -o %j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster

source /home/shaq/zero-mt/mmt-env/bin/activate

root_dir="/home/shaq/Visual-semantics/COMUTE/"

language=("de" "fr" "cs")  # Options: de, fr, cs, fi, ro, zh

image_dir="${root_dir}images/"

for lang in "${language[@]}"; do
    source_file="${root_dir}en-${lang}/src.en"
    image_name_file="${root_dir}en-${lang}/img.order"
    output_file="${root_dir}en-${lang}/"
    lang_latn=""
    if [ "$lang" == "de" ]; then
        lang_latn="deu_Latn"
    elif [ "$lang" == "fr" ]; then
        lang_latn="fra_Latn"
    elif [ "$lang" == "cs" ]; then
        lang_latn="ces_Latn"
    else
        echo "Language not supported"
        exit 1
    fi
    python zeromt.py $image_name_file $output_file $lang_latn $image_dir $source_file
done
