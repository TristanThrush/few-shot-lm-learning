This repo contains python3 code for the paper [Investigating Novel Verb Learning in BERT: Selectional Preference Classes and Alternation-Based Syntactic Generalization](https://www.aclweb.org/anthology/2020.blackboxnlp-1.25.pdf), by Tristan Thrush, Ethan Wilcox, and Roger Levy.

Setup:

`pip install -r requirements.txt`

Experiments:

These will take a while unless you have a GPU.

3 Selectional Preferences:

`python main.py --save_name='prediction' --data='prediction_experiments.txt'`

`python main.py --save_name='prediction_seen_vs_unseen_in_class' --data='prediction_experiments_seen_vs_unseen_in_class.txt'`

4.1 Psycholinguistic Assessment Results

`python main.py --save_name='levin_prediction_revised' --data='levin_prediction_revised.txt'`

4.2 Classificaiton Assessment Results

`python main.py --save_name='linear_similarity_revised' --similarity_experiments_metric='linear_classifier' --data='similarity_revised.txt'`

`python main.py --save_name='linear_similarity_distractor_revised' --similarity_experiments_metric='linear_classifier' --data='distractor_similarity_revised.txt'`
