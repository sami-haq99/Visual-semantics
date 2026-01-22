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

root_dir="/home/shaq/Visual-semantics/mlt-lexical-test-sets/"


image_dir="${root_dir}images/"
output_file="${root_dir}"
image_dir="${root_dir}images/"
language=("deu_Latn")
source_file="${root_dir}ende_test2017mscoco.en"
image_name_file="${root_dir}ende_test2017mscoco_images.txt"
 
python zeromt.py $image_name_file $output_file $language $image_dir $source_file


language=("fra_Latn")
source_file="${root_dir}enfr_test2017mscoco.en"
image_name_file="${root_dir}enfr_test2017mscoco_images.txt"
 
python zeromt.py $image_name_file $output_file $language $image_dir $source_file


language=("deu_Latn")
source_file="${root_dir}ende_test2016.en"
image_name_file="${root_dir}ende_test2016_images.txt"
 
python zeromt.py $image_name_file $output_file $language $image_dir $source_file


