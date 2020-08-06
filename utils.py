from html.parser import HTMLParser
from models import BERT
import matplotlib.pyplot as plt
import random
import scipy
import torch
import pickle
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BertExperiments:
    def __init__(self, experiments):
        self.experiments = experiments

    def experiment(self, seeds, epoch_num, learning_rate):
        for seed in seeds:
            torch.manual_seed(seed)
            random.seed(seed)

            novel_tokens = []
            for experiment in self.experiments:
                novel_tokens += experiment.novel_tokens

            train_data = []
            for experiment in self.experiments:
                train_data += experiment.train_data

            model = BERT(novel_tokens, learning_rate)

            print('Training')
            for epoch in range(epoch_num):
                model.model.train()
                model.optimizer.zero_grad()
                loss = model.get_loss(train_data)
                loss.backward()
                model.optimizer.step()
                print('loss:', loss.item())

            for experiment in self.experiments:
                experiment.run(model)

        pickle.dump(self.experiments, open('results.pkl', 'wb'))

    def significance_test_on_results(self):
        self.experiments = pickle.load(open('results.pkl', 'rb'))
        for experiment in self.experiments:
            p_value = scipy.stats.wilcoxon(experiment.in_class_results,
                experiment.out_class_results)[1]
            in_class_sum = sum(experiment.in_class_results)
            out_class_sum = sum(experiment.out_class_results)
            print(experiment.info)
            print(p_value)
            print(in_class_sum)
            print(out_class_sum)

    def plot_results(self):
        self.experiments = pickle.load(open('results.pkl', 'rb'))
        similarity_experiments = []
        prediction_experiments = []
        for experiment in self.experiments:
            if isinstance(experiment, SimilarityExperiment):
                similarity_experiments.append(experiment)
            elif isinstance(experiment, PredictionExperiment):
                prediction_experiments.append(experiment)
        plot_number = 0
        for experiments in (similarity_experiments, prediction_experiments):
            x = []
            y = []
            colors = []
            index = 0
            max_x = max(set().union(*[experiment.in_class_results
                + experiment.out_class_results for experiment in experiments]))
            for experiment in experiments:
                y += [index+0.05]*len(experiment.in_class_results) + [
                      index-0.05]*len(experiment.out_class_results)
                x += experiment.in_class_results + experiment.out_class_results
                colors += ['skyblue']*len(experiment.in_class_results) + [
                    'salmon']*len(experiment.out_class_results)
                p_value = round(scipy.stats.wilcoxon(
                    experiment.in_class_results,
                    experiment.out_class_results)[1], 2)
                in_class_mean = sum(experiment.in_class_results)/len(
                    experiment.in_class_results)
                out_class_mean = sum(experiment.out_class_results)/len(
                    experiment.out_class_results)
                color = None
                if p_value < 0.05:
                    if in_class_mean > out_class_mean:
                        color = 'b'
                    else:
                        color = 'r'
                annotation_in_class_mean = plt.annotate(
                    '|', (in_class_mean, index), horizontalalignment='center')
                annotation_out_class_mean = plt.annotate(
                    '|', (out_class_mean, index), horizontalalignment='center')
                annotation_info = plt.annotate(experiment.novel_verb
                    + ' (p=' + str(p_value) + ')', (max_x, index+0.3),
                    horizontalalignment='right')
                annotation_in_class_mean.set_fontsize(7)
                annotation_in_class_mean.set_color('b')
                annotation_out_class_mean.set_fontsize(7)
                annotation_out_class_mean.set_color('r')
                annotation_info.set_fontsize(4)
                if color:
                    annotation_info.set_color(color)
                index += 1
            plt.scatter(x, y, c=colors, alpha=0.5, s=[0.5]*len(x))
            plt.yticks([])
            axes = plt.gca()
            axes.set_ylim([-1,30])
            left, right = plt.xlim()
            print(abs(left-right))
            plt.axes().set_aspect(0.05*abs(left-right))
            plt.savefig('results' + str(plot_number) + '.png',
                        bbox_inches='tight', dpi=1000)
            plt.clf()
            plot_number += 1

class Experiment:
    def __init__(self, info, novel_tokens, novel_verb, train_data, in_class,
                 out_class):
        self.info = info
        self.novel_tokens = novel_tokens
        self.novel_verb = novel_verb
        self.train_data = train_data
        self.in_class = in_class
        self.out_class = out_class
        self.in_class_results = []
        self.out_class_results = []

    def run(self, model):
        raise NotImplementedError()

class SimilarityExperiment(Experiment):
    def __init__(self, info, novel_tokens, novel_verb, train_data,
                 in_class_verbs, out_class_verbs):
        super().__init__(info, novel_tokens, novel_verb, train_data,
                         in_class_verbs, out_class_verbs)

    def run(self, model):
        in_class_embeddings = []
        out_class_embeddings = []
        for verb in self.in_class:
            embedding = model.get_embedding(verb)
            if embedding.shape[1] == 1:
                in_class_embeddings.append(embedding)
        for verb in self.out_class:
            embedding = model.get_embedding(verb)
            if embedding.shape[1] == 1:
                out_class_embeddings.append(embedding)
        novel_verb_embedding = model.get_embedding(self.novel_verb)
        in_class_similarity = torch.nn.functional.cosine_similarity(
            novel_verb_embedding.squeeze(0),
            torch.mean(torch.cat(in_class_embeddings), dim=0), dim=1).item()
        out_class_similarity = torch.nn.functional.cosine_similarity(
            novel_verb_embedding.squeeze(0),
            torch.mean(torch.cat(out_class_embeddings), dim=0), dim=1).item()
        self.in_class_results.append(in_class_similarity)
        self.out_class_results.append(out_class_similarity)



class PredictionExperiment(Experiment):
    def __init__(self, info, novel_tokens, novel_verb, train_data,
                 in_class_sentences, out_class_sentences):
        super().__init__(info, novel_tokens, novel_verb, train_data,
                         in_class_sentences, out_class_sentences)

    def run(self, model):
        in_class_predictions = []
        out_class_predictions = []
        for utterance in self.in_class:
            in_class_predictions.append(
                model.get_probability_at(self.novel_verb, utterance))
        for utterance in self.out_class:
            out_class_predictions.append(
                model.get_probability_at(self.novel_verb, utterance))
        mean_in_class_prediction = sum(in_class_predictions)/len(
            in_class_predictions)
        mean_out_class_prediction = sum(out_class_predictions)/len(
            out_class_predictions)
        self.in_class_results.append(mean_in_class_prediction)
        self.out_class_results.append(mean_out_class_prediction)



class VerbLearningExperimentParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.experiments = []

    def handle_starttag(self, tag, attrs):
        if tag == 'prediction' or tag == 'similarity':
            self.current_section = tag
        else:
            self.current_subsection = tag

    def handle_endtag(self, tag):
        if tag == 'prediction':
            self.current_novel_tokens.remove('')
            self.current_train_data.remove('')
            self.experiments.append(PredictionExperiment(
                self.current_info, self.current_novel_tokens,
                self.current_novel_verb, self.current_train_data,
                self.current_in_class, self.current_out_class))
        elif tag == 'similarity':
            self.current_novel_tokens.remove('')
            self.current_train_data.remove('')
            self.experiments.append(SimilarityExperiment(
                self.current_info, self.current_novel_tokens,
                self.current_novel_verb, self.current_train_data,
                self.current_in_class, self.current_out_class))

    def handle_data(self, data):
        if data != '\n':
            if self.current_subsection == 'info':
                self.current_info = data
            elif self.current_subsection == 'novel_verb':
                self.current_novel_verb = data.replace('\n', '')
            elif self.current_subsection == 'novel_tokens':
                self.current_novel_tokens = set(data.split('\n'))
            elif self.current_subsection == 'train':
                self.current_train_data = set(data.split('\n'))
            elif self.current_subsection == 'test':
                self.current_in_class = []
                self.current_out_class = []
                for datum in data.split('\n'):
                    if datum != '':
                        if self.current_section == 'prediction':
                            if datum.startswith('*'):
                                    self.current_out_class.append(
                                        datum.split(':')[1])
                            else:
                                    self.current_in_class.append(
                                        datum.split(':')[1])
                        elif self.current_section == 'similarity':
                            if datum.startswith('*'):
                                    self.current_out_class +=\
                                        datum.split(':')[1].replace(
                                            ' ', '').split(',')
                            else:
                                    self.current_in_class +=\
                                        datum.split(':')[1].replace(
                                            ' ', '').split(',')
                        else:
                            raise ValueError('unexpected tag: '
                                + self.current_subsection)
            else:
                raise ValueError('unexpected tag: ' + self.current_subsection)
