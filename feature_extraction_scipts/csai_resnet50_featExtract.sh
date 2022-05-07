#!/bin/bash
#SBATCH --job-name=res_clustering
#SBATCH -A research
#SBATCH -p long
#SBATCH -c 4
#SBATCH -w gnode62
#SBATCH --mem-per-cpu=3048
#SBATCH --time=4-00:00:00
#SBATCH --output csai_res_cluters.log
#SBATCH --mail-type=ALL

cd /scratch/CSAI
pwd
cp /home2/akshit.garg/Resnet50_feat_extract.py /scratch/CSAI/.
pip3 install tensorflow
echo "Starting python"
python3 /scratch/CSAI/Resnet50_feat_extract.py
echo "Ending python"