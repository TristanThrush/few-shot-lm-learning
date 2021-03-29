#!/bin/bash
#
#SBATCH -p cpl
#SBATCH --time=7-0
#SBATCH --gres=gpu:QUADRORTX6000:1
#SBATCH --mem=40G
#

#python main.py --save_name='cosine_similarity_distractor' --similarity_experiments_metric='cosine_similarity' --data='distractor_similarity_experiments.txt'
#python main.py --save_name='cosine_similarity' --similarity_experiments_metric='cosine_similarity' --data='similarity_experiments.txt'
#python main.py --save_name='linear_similarity_distractor' --similarity_experiments_metric='linear_classifier' --data='distractor_similarity_experiments.txt'
#python main.py --save_name='linear_similarity' --similarity_experiments_metric='linear_classifier' --data='similarity_experiments.txt'
#python main.py --save_name='prediction' --data='prediction_experiments.txt'
#python main.py --save_name='prediction_seen_vs_unseen_in_class' --data='prediction_experiments_seen_vs_unseen_in_class.txt'
#python main.py --save_name='levin_prediction_experiments_sanity_check' --data='levin_prediction_experiments.txt'

#python main.py --save_name='linear_similarity_distractor_revised' --similarity_experiments_metric='linear_classifier' --data='distractor_similarity_revised.txt'
#python main.py --save_name='linear_similarity_revised' --similarity_experiments_metric='linear_classifier' --data='similarity_revised.txt'
python main.py --save_name='levin_prediction_revised2' --data='levin_prediction_revised.txt'
