from transformers import BertTokenizer, BertForMaskedLM
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BidirectionalTransformer:
    def __init__(self, novel_tokens):
        self.mask_token = '[MASK]'

    def get_probability_at(self, word, utterance):
        raise NotImplementedError()

    def get_loss(utterances):
        raise NotImplementedError()

    def get_embedding(utterance):
        raise NotImplementedError()


class BERT(BidirectionalTransformer):
    def __init__(self, novel_tokens, learning_rate):
        super().__init__(novel_tokens)
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.model = BertForMaskedLM.from_pretrained(
            'bert-large-uncased').to(device)
        self.tokenizer.add_tokens(novel_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.optimizer = torch.optim.Adam(
            list(self.model.cls.predictions.decoder.parameters())
            + list(self.model.bert.embeddings.word_embeddings.parameters()),
            learning_rate)

    def get_probability_at(self, word, utterance):
        encoded_word = self.tokenizer.encode(word, add_special_tokens=True)[1]
        utterance = utterance.replace(
            self.mask_token, self.tokenizer.mask_token)
        encoded_list = self.tokenizer.encode(utterance,
                                             add_special_tokens=True)
        score_index = encoded_list.index(encoded_word)
        encoded_list = self.tokenizer.encode(
            utterance.replace(word, self.tokenizer.mask_token),
            add_special_tokens=True)
        input = torch.tensor(encoded_list).unsqueeze(0).to(device)
        probs = torch.nn.LogSoftmax()(self.model(input)[0][0][score_index])
        return probs[encoded_word].item()

    def get_loss(self, utterances):
        char = '[v'
        utterances = [utterance.replace(
            self.mask_token, self.tokenizer.mask_token).replace('\n','')
            for utterance in utterances]
        tokenized_input = self.tokenizer.batch_encode_plus(
            utterances, pad_to_max_length=True)
        masked_input_ids = []
        input_ids = []
        for item in tokenized_input['input_ids']:
            masked_input_ids.append([])
            input_ids.append([])
            for id in item:
                if self.tokenizer.decode([id]).startswith(char):
                    masked_input_ids[-1].append(
                        self.tokenizer.encode(self.tokenizer.mask_token)[1])
                    input_ids[-1].append(id)
                else:
                    input_ids[-1].append(-100)
                    masked_input_ids[-1].append(id)
        masked_input_ids = torch.tensor(masked_input_ids).to(device)
        input_ids = torch.tensor(input_ids).to(device)
        attention_mask = torch.tensor(
            tokenized_input['attention_mask']).to(device)
        loss = self.model(
            masked_input_ids,
            attention_mask=attention_mask, masked_lm_labels=input_ids)[0]
        return loss

    def get_embedding(self, word):
        encoded_list = self.tokenizer.encode(word)[1:-1]
        encoded = torch.tensor(encoded_list).unsqueeze(0).to(device)
        embedding = self.model.bert.embeddings.word_embeddings(encoded)
        return embedding
