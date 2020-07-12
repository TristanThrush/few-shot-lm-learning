from transformers import BertTokenizer, BertForMaskedLM,\
    RobertaTokenizer, RobertaForMaskedLM
import torch
import re
import random
import ast
import scipy
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

epochs = 20
seeds = range(10)
model_name = 'bert-large-uncased'

test = open('test.txt').readlines()

inputs_to_scores = {}
for seed in seeds:
    torch.manual_seed(seed)
    random.seed(seed)

    train = open('train.txt').readlines()
    random.shuffle(train)

    new_tokens = re.findall('\[[N|V][0-9]+\]', open('train.txt').read())
    random.shuffle(new_tokens)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name).to(device)

    num_added_toks = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    tokenized_input = tokenizer.batch_encode_plus(
        train, pad_to_max_length=True)
    masked_input_ids = []
    input_ids = []
    for item in tokenized_input['input_ids']:
        masked_input_ids.append([])
        input_ids.append([])
        for id in item:
            if id < 30522:
                input_ids[-1].append(-100)
                masked_input_ids[-1].append(id)
            else:
                input_ids[-1].append(id)
                if tokenizer.decode([id]).startswith('[V'):
                    masked_input_ids[-1].append(
                        tokenizer.encode(tokenizer.mask_token)[1])
                else:
                    masked_input_ids[-1].append(id)
    masked_input_ids = torch.tensor(masked_input_ids).to(device)
    input_ids = torch.tensor(input_ids).to(device)
    attention_mask = torch.tensor(
        tokenized_input['attention_mask']).to(device)

    def encode(utterance):
        return torch.tensor(tokenizer.encode(
            utterance, add_special_tokens=True)).unsqueeze(0).to(device)
    optimizer = torch.optim.Adam(
        list(model.cls.predictions.decoder.parameters())
        + list(model.bert.embeddings.word_embeddings.parameters()), 1e-3)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model(
            masked_input_ids,
            attention_mask=attention_mask, masked_lm_labels=input_ids)[0]
        loss.backward()
        optimizer.step()
        print('loss:', loss.item())

    for string_input in test:
        tuple_input = ast.literal_eval(string_input)
        for tuple_index in range(len(tuple_input)):
            input = tuple_input[tuple_index].replace('\n', '')
            scores = []
            nonce_verb_range_tags = re.findall(
                '\[V[0-9]+\]-\[V[0-9]+\]', input)
            inputs_to_average = []
            if len(nonce_verb_range_tags) == 1:
                nonce_verb_range_tags[0]
                range_numbers = re.findall(
                    '\[V[0-9]+\]', nonce_verb_range_tags[0])
                for item in range(
                        int(range_numbers[0][2:-1]),
                        int(range_numbers[1][2:-1]) + 1):
                    inputs_to_average.append(
                        input.replace(nonce_verb_range_tags[0],
                        '[V' + str(item) + ']'))
            else:
                inputs_to_average.append(input)
            for processed_input in inputs_to_average:
                nonce_verb = re.findall('\[V[0-9]+\]', processed_input)[0]
                encoded_list = tokenizer.encode(processed_input,
                    add_special_tokens=True)
                score_index = encoded_list.index(
                    tokenizer.encode(nonce_verb, add_special_tokens=True)[1])
                encoded_list = tokenizer.encode(
                    processed_input.replace(nonce_verb, tokenizer.mask_token),
                    add_special_tokens=True)
                scores.append(
                    model(torch.tensor(encoded_list).unsqueeze(0).to(device))
                        [0][0][score_index][tokenizer.encode(
                            nonce_verb, add_special_tokens=True)[1]].item())
            if tuple_input in inputs_to_scores:
                if len(inputs_to_scores[tuple_input]) > tuple_index:
                    inputs_to_scores[tuple_input][tuple_index].append(
                        (sum(scores)/len(scores))/len(seeds))
                else:
                    inputs_to_scores[tuple_input].append(
                        [(sum(scores)/len(scores))/len(seeds)])
            else:
                inputs_to_scores[tuple_input] = [
                    [(sum(scores)/len(scores))/len(seeds)]]
for item in inputs_to_scores.items():
    print(item[0])
    print('p-value:', scipy.stats.median_test(
        np.array(item[1][0]), np.array(item[1][1]))[1])
    print('sampled median 1:', np.median(np.array(item[1][0])))
    print('sampled median 2:', np.median(np.array(item[1][1])))
