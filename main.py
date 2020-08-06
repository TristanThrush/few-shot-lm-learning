from utils import VerbLearningExperimentParser, BertExperiments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10,
                    help='number of training epochs')
parser.add_argument('--seeds', type=int, default=100,
                    help='random seed')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='random seed')
args = parser.parse_args()

experiment_parser = VerbLearningExperimentParser()
experiment_parser.feed(open('experiments.txt').read())

bert_experiments = BertExperiments(experiment_parser.experiments)
#bert_experiments.experiment(range(args.seeds), args.epochs, args.lr)
bert_experiments.significance_test_on_results()
bert_experiments.plot_results()
