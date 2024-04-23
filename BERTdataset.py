
import torch

import random
import tqdm
from pathlib import Path

from torch.utils.data import Dataset, DataLoader


VOCAB_SIZE = 512


class BERTDataset(Dataset):
    def __init__(self, data_pair, special_tokens, seq_len=64):
        self.tokens = special_tokens
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        # Step 1: get random sentence pair, either negative or positive (saved as is_next_label)
        pair, is_next = self.get_sent(item)

        # Step 2: replace random words in sentence with mask / random words
        s1_random, s1_label = self.masker(pair[0])
        s2_random, s2_label = self.masker(pair[1])


        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
        # Adding PAD token for labels
        t1 = [self.tokens["bos"]] + s1_random + [self.tokens["pad"]]
        t2 = s2_random + [self.tokens["eos"]]
        s1_label = [self.tokens["pad"]] + s1_label + [self.tokens["pad"]]
        s2_label = s2_label + [self.tokens["pad"]]

        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len
        segment_label = ([1]*len(t1) + [2]*len(t2))[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (s1_label + s2_label)[:self.seq_len]
        padding = [self.tokens["pad"]] * (self.seq_len - len(bert_input))
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)


        assert len(bert_input) == len(bert_label) == len(segment_label) == self.seq_len

        output = {"bert_input": torch.tensor(bert_input), #input to bert model
                  "bert_label": torch.tensor(bert_label), #masked words real token for calculating loss
                  "segment_label": torch.tensor(segment_label), #segment token for pair sentences (1 on first sentence tokens and 2 for second sentence tokens)
                  "is_next": torch.tensor(is_next)} #if the second sentence logically follows the first sentence
        

        return output

    def masker(self, sentence):
        tokens = sentence
        output_label = []

        # 15% of the tokens would be replaced
        for i in range(len(tokens)):
            prob = random.random()

            if prob < 0.15:
                prob /= 0.15 #normalize prob to 1
                #save original token
                output_label.append(sentence[i])

                if prob < 0.8: # 80% chance change token to mask token
                    tokens[i] = self.tokens['mask']

                elif prob < 0.9: # 10% chance change token to random token
                    tokens[i] = random.randrange(self.tokens['mask']+1, VOCAB_SIZE)
                else:
                    pass #keep the original token and try to predict it

            else: #not changed token
                output_label.append(0)

        
        return tokens, output_label

    def get_sent(self, index):
        '''return random sentence pair, labels and isNext'''
        prob = random.random()
        if prob > 0.5:
            return self.lines[index], 1
        else:
            random_index = random.randrange(self.corpus_lines)
            return (self.lines[index][0], self.lines[random_index][1]), 0
            