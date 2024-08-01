import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from transformers import Trainer, TrainingArguments
from torch import cuda
device = 'mps'
data = pd.read_csv("ner_datasetreference.csv", encoding='unicode_escape')

print("Number of tags: {}".format(len(data.Tag.unique())))
frequencies = data.Tag.value_counts()
print(frequencies)

tags = {}
for tag, count in zip(frequencies.index, frequencies):
    if tag != "O":
        if tag[2:5] not in tags.keys():
            tags[tag[2:5]] = count
        else:
            tags[tag[2:5]] += count
    continue

print(sorted(tags.items(), key=lambda x: x[1], reverse=True))

entities_to_remove = ["B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"]
data = data[~data.Tag.isin(entities_to_remove)]
print(data.head())
# pandas has a very handy "forward fill" function to fill missing values based on the last upper non-nan value
data = data.fillna(method='ffill')
data['sentence'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
# let's also create a new column called "word_labels" which groups the tags by sentence
data['word_labels'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Tag'].transform(lambda x: ','.join(x))
label2id = {k: v for v, k in enumerate(data.Tag.unique())}
id2label = {v: k for v, k in enumerate(data.Tag.unique())}
print(label2id)
data = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
print(data.head())
print(data.iloc[41])
print(len(data.iloc[41].sentence))
print(len(data.iloc[41].word_labels))
# 到这里为止，都是构建训练样本，样本包含一个完整到sentence，以及对应到iob

MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []

    sentence = sentence.strip()
    for word, label in zip(sentence.split(), text_labels.split(",")):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)
    return tokenized_sentence, labels

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        sentence = self.data.sentence[index]
        word_labels = self.data.word_labels[index]
        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)
        tokenized_sentence =["[CLS]"] + tokenized_sentence + ["[SEP]"]
        labels.insert(0, "0")
        labels.insert(-1, "0")
        if len(tokenized_sentence) > self.max_len:
            tokenized_sentence = tokenized_sentence[:self.max_len]
            labels = labels[:self.max_len]
        else:
            tokenized_sentence = tokenized_sentence + ['[PAD]' for _ in range(self.max_len - len(tokenized_sentence))]
            labels = labels + ["0" for _ in range(self.max_len - len(labels))]

        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        label_ids = [label2id[label] for label in labels]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len