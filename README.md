This repo contains python3 code for the paper [Investigating Novel Verb Learning in BERT: Selectional Preference Classes and Alternation-Based Syntactic Generalization](https://www.aclweb.org/anthology/2020.blackboxnlp-1.25.pdf), by Tristan Thrush, Ethan Wilcox, and Roger Levy.

Setup:

`pip install -r requirements.txt`

Experiments:

These will take a while unless you have a GPU.

The code produces a png with a graph for each experiment. Blue-colored results
represent the correct judgement with a significant p-value. Red-colored results
represent the incorrect judgement with a significant p-value. There are also
generated csv files for each experiment.

3 Selectional Preferences:

`python main.py --save_name='prediction' --data='prediction_experiments.txt'`

`python main.py --save_name='prediction_seen_vs_unseen_in_class' --data='prediction_experiments_seen_vs_unseen_in_class.txt'`

4.1 Psycholinguistic Assessment Results

`python main.py --use_levin_prediction_parser --save_name='levin_prediction' --data='levin_prediction_experiments.txt'`

4.2 Classificaiton Assessment Results

`python main.py --save_name='linear_similarity' --similarity_experiments_metric='linear_classifier' --data='similarity_experiments.txt'`

`python main.py --save_name='linear_similarity_distractor' --similarity_experiments_metric='linear_classifier' --data='distractor_similarity_experiments.txt'`
