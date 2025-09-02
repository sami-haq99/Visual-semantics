#!/usr/bin/env bash

#SBATCH --gres=gpu:rtxa6000:2
# SBATCH --nodelist=g126
#SBATCH -p compute
#!/usr/bin/env bash

#SBATCH --gres=gpu:rtxa6000:2
# SBATCH --nodelist=g126
#SBATCH -p compute
#SBATCH -J captions
#SBATCH -t 23:59:59
#SBATCH -o comet-%j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster

source /home/shaq/image-caption/env-ic/bin/activate

image_name_file="multi30k-2016/image_test_2016_flickr.txt"
output_file="multi30k-2016/"
language="de"
image_dir="multi30k-2016/images/"

python ic-gemma3b.py $image_name_file $output_file $language $image_dir

