from utils import VerbLearningExperimentParser, BertExperiments,\
    SimilarityExperiment, LevinPredictionExperimentParser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_levin_prediction_parser', type=bool)
parser.add_argument('--epochs', type=int, default=10,
                    help='number of training epochs')
parser.add_argument('--seeds', type=int, default=20,
                    help='number of random seeds')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--save_name', type=str, default='results',
                    help='save name for figures and data file')
parser.add_argument('--similarity_experiments_metric', type=str,
                    default='cosine_similarity',
                    help='similarity metric for similarity experiments')
parser.add_argument('--data', type=str,
                    help='experiments file')
args = parser.parse_args()

if args.use_levin_prediction_parser:
    experiment_parser = LevinPredictionExperimentParser()
    unparsed = eval(open(args.data).read())
else:
    experiment_parser = VerbLearningExperimentParser()
    unparsed = open(args.data).read()

experiment_parser.feed(unparsed)


for experiment in experiment_parser.experiments:
    if isinstance(experiment, SimilarityExperiment):
        experiment.metric = args.similarity_experiments_metric
bert_experiments = BertExperiments(experiment_parser.experiments,
                                   args.save_name)
bert_experiments.experiment(range(args.seeds), args.epochs, args.lr)
bert_experiments.significance_test_on_results()
bert_experiments.plot_results()
bert_experiments.save_results_as_csv(args.seeds)
