from html.parser import HTMLParser
from models import BERT
import matplotlib.pyplot as plt
import random
import scipy
import torch
import csv
import pickle
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BertExperiments:
    def __init__(self, experiments, save_name):
        self.experiments = experiments
        self.save_name = save_name

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

        pickle.dump(self.experiments, open(self.save_name + '.pkl', 'wb'))

    def significance_test_on_results(self):
        self.experiments = pickle.load(open(self.save_name + '.pkl', 'rb'))
        for experiment in self.experiments:
            p_value = scipy.stats.wilcoxon(experiment.in_class_results,
                experiment.out_class_results)[1]
            in_class_sum = sum(experiment.in_class_results)
            out_class_sum = sum(experiment.out_class_results)
            print(experiment.info)
            if isinstance(experiment, SimilarityExperiment):
                if experiment.metric == 'linear_classifier':
                    print(experiment.classifier_accuracy)
            print(p_value)
            print(in_class_sum)
            print(out_class_sum)

    def save_results_as_csv(self, seeds):
        lines = open('word_frequencies.txt', 'r').readlines()
        word_frequencies = {}
        for line in lines:
            word = line.split()[1]
            frequency = float(line.split()[2])
            word_frequencies[word] = frequency
        self.experiments = pickle.load(open(self.save_name + '.pkl', 'rb'))
        max_in_class_words = max([len(experiment.in_class) for experiment in self.experiments])
        max_out_class_words = max([len(experiment.out_class) for experiment in self.experiments])
        fieldnames = ['experiment_info', 'novel_verb', 'linear_classifier_train_accuracy']\
            + ['in_class_word_frequency_' + str(index) for index in range(max_in_class_words)]\
            + ['out_class_word_frequency_' + str(index) for index in range(max_out_class_words)]\
            + ['linear_classification_on_seed_' + str(seed) for seed in range(seeds)]\
            + ['in_class_cosine_similarity_on_seed_' + str(seed) for seed in range(seeds)]\
            + ['out_class_cosine_similarity_on_seed_' + str(seed) for seed in range(seeds)]\
            + ['in_class_prediction_probability_on_seed_' + str(seed) for seed in range(seeds)]\
            + ['out_class_prediction_probability_on_seed_' + str(seed) for seed in range(seeds)]
        writer = csv.DictWriter(open(self.save_name + '.csv', 'w', newline=''), fieldnames=fieldnames)
        writer.writeheader()
        for experiment in self.experiments:
            csv_dict = {}
            csv_dict['experiment_info'] = experiment.info
            csv_dict['novel_verb'] = experiment.novel_verb
            if isinstance(experiment, SimilarityExperiment):
                for index in range(len(experiment.in_class)):
                    word = experiment.in_class[index]
                    if word in word_frequencies:
                        csv_dict['in_class_word_frequency_' + str(index)] = word_frequencies[word]
                    else:
                        csv_dict['in_class_word_frequency_' + str(index)] = 0
                for index in range(len(experiment.out_class)):
                    word = experiment.out_class[index]
                    if word in word_frequencies:
                        csv_dict['out_class_word_frequency_' + str(index)] = word_frequencies[word]
                    else:
                        csv_dict['out_class_word_frequency_' + str(index)] = 0
                if experiment.metric == 'linear_classifier':
                    csv_dict['linear_classifier_train_accuracy'] = experiment.classifier_accuracy
                    for seed in range(seeds):
                        csv_dict['linear_classification_on_seed_' + str(seed)] = experiment.in_class_results[seed]

                    #Write weights of linear classifier to their own csv.
                    classifier_fieldnames = ['out_dimension', 'bias'] + ['weight_' + str(in_dim) for in_dim in range(experiment.classifier.weight[0].shape[0])]
                    weights_writer = csv.DictWriter(open(self.save_name + '_' + experiment.novel_verb + '_classifier_weights.csv', 'w', newline=''), fieldnames=classifier_fieldnames)
                    weights_writer.writeheader()
                    for out_dim in [0,1]:
                        classifier_csv_dict = {}
                        classifier_csv_dict['out_dimension'] = out_dim
                        classifier_csv_dict['bias'] = experiment.classifier.bias[out_dim].item()
                        for index in range(experiment.classifier.weight[out_dim].shape[0]):
                            classifier_csv_dict['weight_' + str(index)] = experiment.classifier.weight[out_dim][index].item()
                        weights_writer.writerow(classifier_csv_dict)

                if experiment.metric == 'cosine_similarity':
                    for seed in range(seeds):
                        csv_dict['in_class_cosine_similarity_on_seed_' + str(seed)] = experiment.in_class_results[seed]
                        csv_dict['out_class_cosine_similarity_on_seed_' + str(seed)] = experiment.out_class_results[seed]
            if isinstance(experiment, PredictionExperiment):
                for seed in range(seeds):
                    csv_dict['in_class_prediction_probability_on_seed_' + str(seed)] = experiment.in_class_results[seed]
                    csv_dict['out_class_prediction_probability_on_seed_' + str(seed)] = experiment.out_class_results[seed]
            writer.writerow(csv_dict)


    def plot_results(self):
        self.experiments = pickle.load(open(self.save_name + '.pkl', 'rb'))
        similarity_experiments = []
        prediction_experiments = []
        for experiment in self.experiments:
            if isinstance(experiment, SimilarityExperiment):
                similarity_experiments.append(experiment)
            elif isinstance(experiment, PredictionExperiment):
                prediction_experiments.append(experiment)
        plot_number = 0
        for experiments in (similarity_experiments, prediction_experiments):
            if len(experiments) > 0:
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
                axes.set_ylim([-1,300])
                left, right = plt.xlim()
                plt.axes().set_aspect(0.05*abs(left-right))
                plt.savefig(self.save_name + str(plot_number) + '.png',
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
        self.classifier_trained = False
        self.learning_rate = 1e-1
        self.epochs = 20
        self.metric = 'linear_classifier'

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
        if self.metric == 'cosine_similarity':
            in_class_similarity = torch.nn.functional.cosine_similarity(
                novel_verb_embedding.squeeze(0),
                torch.mean(torch.cat(in_class_embeddings), dim=0), dim=1).item()
            out_class_similarity = torch.nn.functional.cosine_similarity(
                novel_verb_embedding.squeeze(0),
                torch.mean(torch.cat(out_class_embeddings), dim=0), dim=1).item()
            self.in_class_results.append(in_class_similarity)
            self.out_class_results.append(out_class_similarity)
        if self.metric == 'linear_classifier':
            if not self.classifier_trained:
                self.classifier = torch.nn.Linear(
                    novel_verb_embedding.shape[-1], 2).to(device)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(self.classifier.parameters(),
                                             self.learning_rate)
                self.classifier.train()
                batch = torch.cat(in_class_embeddings
                    + out_class_embeddings).squeeze(1)
                batch_labels = torch.tensor([1]*len(in_class_embeddings)
                    + [0]*len(out_class_embeddings)).to(device)
                print('training classifier for:', self.info)
                for epoch in range(self.epochs):
                    optimizer.zero_grad()
                    loss = criterion(self.classifier(batch), batch_labels)
                    loss.backward(retain_graph=True)
                    print('loss:', loss.item())
                    optimizer.step()
                self.classifier.eval()
                self.classifier_trained = True
                self.classifier_accuracy = (batch_labels.shape[0] - torch.sum(torch.abs(torch.max(self.classifier(batch), 1)[1].float() - batch_labels.float())).item())/batch_labels.shape[0]
                print('out:', self.classifier(novel_verb_embedding))
            result = torch.max(self.classifier(novel_verb_embedding), 2)[1].item()
            self.in_class_results.append(result)
            self.out_class_results.append(1-result)


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

class LevinPredictionExperimentParser:
    def __init__(self):
        self.experiments = []

    '''
    def feed(self, string):
        lines = string.split('\n')
        lines.remove('')
        n = 2
        alternations = [lines[i * n:(i + 1) * n] for i in range((len(lines) + n - 1) // n )]
        novel_verb_counter = 1
        for alternation in alternations:
            novel_verb_1 = '[V' + str(novel_verb_counter) + '.1]'
            novel_verb_2 = '[V' + str(novel_verb_counter) + '.2]'
            train_data_1 = [alternation[0].replace('[V]', novel_verb_1)]
            train_data_2 = [alternation[1].replace('[V]', novel_verb_2)]
            in_class_test_data_1 = [alternation[1].replace('[V]', novel_verb_1)]
            in_class_test_data_2 = [alternation[0].replace('[V]', novel_verb_2)]
            out_class_test_data_1 = []
            out_class_test_data_2 = []
            for index in range(len(lines)):
                if [alternation[0], lines[index]] not in alternations and [lines[index], alternation[0]] not in alternations and alternation[0] != lines[index]:
                    out_class_test_data_1.append(lines[index].replace('[V]', novel_verb_1))
                if [alternation[1], lines[index]] not in alternations and [lines[index], alternation[1]] not in alternations and alternation[1] != lines[index]:
                    out_class_test_data_2.append(lines[index].replace('[V]', novel_verb_2))
            self.experiments.append(PredictionExperiment('', [novel_verb_1], novel_verb_1, train_data_1, in_class_test_data_1, out_class_test_data_1))
            self.experiments.append(PredictionExperiment('', [novel_verb_2], novel_verb_2, train_data_2, in_class_test_data_2, out_class_test_data_2))
            novel_verb_counter += 1
    '''

    def feed(self, experiment_dict):
        all_alternations = []
        for value in experiment_dict.values():
            all_alternations += value
        novel_verb_counter = 1
        for info, alternations in experiment_dict.items():
            index = 0
            for frame1, frame2 in alternations:
                novel_verb_1 = '[V' + str(novel_verb_counter) + '.1.' + str(index) + ']'
                novel_verb_2 = '[V' + str(novel_verb_counter) + '.2.' + str(index) + ']'
                train_data_1 = [frame1.replace('[V]', novel_verb_1)]
                train_data_2 = [frame2.replace('[V]', novel_verb_2)]
                in_class_test_data_1 = [frame2.replace('[V]', novel_verb_1)]
                in_class_test_data_2 = [frame1.replace('[V]', novel_verb_2)]
                out_class_test_data_1 = []
                out_class_test_data_2 = []
                for frame1sub, frame2sub in all_alternations:
                    if frame1sub != frame1:
                        if (frame1, frame1sub) not in all_alternations and (frame1sub, frame1) not in all_alternations:
                            out_class_test_data_1.append(frame1sub.replace('[V]', novel_verb_1))
                    if frame2sub != frame1:
                        if (frame1, frame2sub) not in all_alternations and (frame2sub, frame1) not in all_alternations:
                            out_class_test_data_1.append(frame2sub.replace('[V]', novel_verb_1))
                    if frame1sub != frame2:
                        if (frame2, frame1sub) not in all_alternations and (frame1sub, frame2) not in all_alternations:
                            out_class_test_data_2.append(frame1sub.replace('[V]', novel_verb_2))
                    if frame2sub != frame2:
                        if (frame2, frame2sub) not in all_alternations and (frame2sub, frame2) not in all_alternations:
                            out_class_test_data_2.append(frame2sub.replace('[V]', novel_verb_2))
                index += 1
                self.experiments.append(PredictionExperiment(info, [novel_verb_1], novel_verb_1, train_data_1, in_class_test_data_1, out_class_test_data_1))
                self.experiments.append(PredictionExperiment(info, [novel_verb_2], novel_verb_2, train_data_2, in_class_test_data_2, out_class_test_data_2))
            novel_verb_counter += 1
