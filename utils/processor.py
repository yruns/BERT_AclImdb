import json

from transformers import BertTokenizer
from utils.hyperParams import get_parser
from utils.tools import InputFeature, LstmFeatures


class Processor(object):

    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_path)
        self.max_seq_length = args.max_seq_length
        self.train_file = args.data_dir + '/train.json'
        self.test_file = args.data_dir + '/test.json'
        # self.dev_file = args.data_dir + '/dev.txt'
        # self.all_file = args.data_dir + '/hotel.txt'

        self.word2idx = None
        self.idx2word = None

    def tokenize(self, text):
        """
        Tokenize text.
        """
        if self.tokenizer is None or not isinstance(self.tokenizer, BertTokenizer):
            raise ValueError('tokenizer is None or is not BertTokenizer')
        return self.tokenizer.convert_tokens_to_ids(text)

    def detokenize(self, ids):
        """
        Detokenize ids.
        """
        if self.tokenizer is None or not isinstance(self.tokenizer, BertTokenizer):
            raise ValueError('tokenizer is None or is not BertTokenizer')
        return self.tokenizer.convert_ids_to_tokens(ids)


    def get_map(self):
        """
        Get the map of word to index and index to word
        """
        self.word2idx = {'[PAD]': 0, '[UKN]': 1}
        self.idx2word = {0: '[PAD]', 1: '[UKN]'}
        word_count = {}
        with open(self.train_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                text = line['text']

                for word in self.tokenizer.tokenize(text):
                    if word not in word_count:
                        word_count[word] = 1
                    else:
                        word_count[word] += 1

        # sort the word according to the frequency
        word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:15000]
        for word, _ in word_count:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word

        return self.word2idx, self.idx2word


    def convert_examples_to_features(self, mode='train'):
        """
        Convert examples to features.
        """
        if mode == 'train':
            file = self.train_file
        elif mode == 'test':
            file = self.test_file
        # elif mode == 'dev':
        #     file = self.dev_file
        else:
            raise ValueError('mode must be train or test')

        features = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                text, label = line['text'], line['label']

                tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
                tokens = ['[CLS]'] + tokens[:self.max_seq_length - 2] + ['[SEP]']

                seq_len = len(tokens)
                tokens = tokens + ['[PAD]'] * (self.max_seq_length - seq_len)
                token_ids = self.tokenize(tokens)
                mask_ids = [1] * seq_len + [0] * (self.max_seq_length - seq_len)
                type_ids = [0] * self.max_seq_length

                assert len(token_ids) == self.max_seq_length
                assert len(mask_ids) == self.max_seq_length
                assert len(type_ids) == self.max_seq_length

                features.append(
                    InputFeature(
                        input_ids=token_ids,
                        mask_ids=mask_ids,
                        type_ids=type_ids,
                        label=label,
                        text=text
                    )
                )

        return features

    def convert_file_to_lstm(self, mode='train'):
        """
        Convert file to data.
        """
        if mode == 'train':
            file = self.train_file
        elif mode == 'test':
            file = self.test_file
        # elif mode == 'dev':
        #     file = self.dev_file
        else:
            raise ValueError('mode must be train or test')

        features = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                text, label = line['text'], line['label']

                tokens = self.tokenizer.tokenize(text)
                token_idx = self.tokenizer.convert_tokens_to_ids(tokens)
                token_idx = token_idx[:self.max_seq_length]
                token_idx = token_idx + [0] * (self.max_seq_length - len(token_idx))

                assert len(token_idx) == self.max_seq_length

                features.append(LstmFeatures(
                    input_ids=token_idx,
                    label=label,
                    text=text)
                )

        return features





if __name__ == '__main__':
    args = get_parser()
    processor = Processor(args)
    processor.convert_file_to_lstm('train')