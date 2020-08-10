#!/bin/bash
#
#SBATCH -p cpl
#SBATCH --time=7-0
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=40G
#

python main.py --save_name='cosine_similarity_distractor' --similarity_experiments_metric='cosine_similarity' --data='distractor_similarity_experiments.txt'
#python main.py --save_name='linear_similarity_distractor' --similarity_experiments_metric='linear_classifier' --data='distractor_similarity_experiments.txt'
#python main.py --save_name='cosine_similarity' --similarity_experiments_metric='cosine_similarity' --data='similarity_experiments.txt'
#python main.py --save_name='linear_similarity' --similarity_experiments_metric='linear_classifier' --data='similarity_experiments.txt'
#python main.py --save_name='prediction' --data='prediction_experiments.txt'
