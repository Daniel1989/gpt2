# 训练加速
# 目前时间主要花在loss.item, loss.backward, optimizer.step上, 尤其是loss.backward
# https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/
# 1. 调整lr，有效
# 2. 开启多个worker，有效
# 3. 选择合适的batch_size，有效
# 4. 使用自动混合精度，只支持nvidia，未验证
# 5. 使用另外的优化器，无效
# 6. 设置 torch.backends.cudnn.benchmark = True， 无效
# 7. 直接在gpu生成tensor，而不是move，由于模型设置，无法生效。一般可以在dataset里，直接把所有数据都弄到gpu
# 8. 使用梯度/激活检查点.检查点的工作原理是用计算换取内存。检查点部分并不存储整个计算图的所有中间激活用于向后计算，而是不保存中间激活，而是在向后传递中重新计算它们。它可以应用于模型的任何部分。模型限制，无法生效


import os
import time

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import Dataset, DataLoader
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter

from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from transformers import Trainer, TrainingArguments
import torch.profiler
import csv
device = 'mps'
script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(script_path)

data = pd.read_csv(os.path.join(parent_directory, 'ner_datasetreference.csv'), encoding='unicode_escape')
# 本质上是一个分类
# 统计每个标签的个数，就是看要分到哪些类
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
#
entities_to_remove = ["B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"]
data = data[~data.Tag.isin(entities_to_remove)]
# # pandas has a very handy "forward fill" function to fill missing values based on the last upper non-nan value
data = data.fillna(method='ffill')
data['sentence'] = data[['Sentence #', 'Word', 'Tag']].groupby(['Sentence #'])['Word'].transform(
    lambda x: ' '.join(x))
# # let's also create a new column called "word_labels" which groups the tags by sentence
data['word_labels'] = data[['Sentence #', 'Word', 'Tag']].groupby(['Sentence #'])['Tag'].transform(
    lambda x: ','.join(x))

label2id = {k: v for v, k in enumerate(data.Tag.unique())}
id2label = {v: k for v, k in enumerate(data.Tag.unique())}
data = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
# # 到这里为止，都是构建训练样本，样本包含一个完整到sentence，以及对应到iob
#
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#
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
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        labels.insert(0, "O")
        labels.insert(-1, "O")
        if len(tokenized_sentence) > self.max_len:
            tokenized_sentence = tokenized_sentence[:self.max_len]
            labels = labels[:self.max_len]
        else:
            tokenized_sentence = tokenized_sentence + ['[PAD]' for _ in
                                                       range(self.max_len - len(tokenized_sentence))]
            labels = labels + ["O" for _ in range(self.max_len - len(labels))]

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

#
train_size = 0.8

# data = data.sample(n=1000, random_state=42)
train_dataset = data.sample(frac=train_size, random_state=200)
test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

#
training_set = dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = dataset(test_dataset, tokenizer, MAX_LEN)



def main():

    #
    # # print the first 30 tokens and corresponding labels
    # # for token, label in zip(tokenizer.convert_ids_to_tokens(training_set[0]["ids"][:30]), training_set[0]["targets"][:30]):
    # #     print('{0:10}  {1}'.format(token, id2label[label.item()]))
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 4,
                    'pin_memory': True
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 4,
                   'pin_memory': True
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)


    model = BertForTokenClassification.from_pretrained('bert-base-uncased',
                                                       num_labels=len(id2label),
                                                       id2label=id2label,
                                                       label2id=label2id)
    model.to(device)
    # ids = training_set[0]["ids"].unsqueeze(0)
    # mask = training_set[0]["mask"].unsqueeze(0)
    # targets = training_set[0]["targets"].unsqueeze(0)
    # ids = ids.to(device)
    # mask = mask.to(device)
    # targets = targets.to(device)
    # outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
    # initial_loss = outputs[0]
    # tr_logits = outputs[1]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(training_loader),
                                                    epochs=1)

    def train(epoch):
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        model.train()
        metric_csv = [
            ["num", "step", "total"]
        ]
        for idx, batch in enumerate(training_loader):
            metric = [idx + 1]
            step_start_time = time.time()  # Start timer for the step
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)
            # print(f'1 Epoch {epoch + 1}, Step {idx + 1} took {time.time() - step_start_time:.4f} seconds')
            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, tr_logits = outputs.loss, outputs.logits
            # print(f'2 Epoch {epoch + 1}, Step {idx + 1} took {time.time() - step_start_time:.4f} seconds')
            # tr_loss += loss.item() # 用于提取训练损失，主要是为了查看进度，可以不要
            # print(f'2.5 Epoch {epoch + 1}, Step {idx + 1} took {time.time() - step_start_time:.4f} seconds')

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # if idx % 100 == 0:
            #     loss_step = tr_loss / nb_tr_steps
            #     print(f"Training loss per 100 training steps: {loss_step}")
            # print(f'3 Epoch {epoch + 1}, Step {idx + 1} took {time.time() - step_start_time:.4f} seconds')

            # flatten the targets
            flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
            # 获取最大值的索引
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
            # print(f'4 Epoch {epoch + 1}, Step {idx + 1} took {time.time() - step_start_time:.4f} seconds')

            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tr_preds.extend(predictions)
            tr_labels.extend(targets)

            # 因为其实是每个token都是一次预测，所以通过这里计算精度
            # print(f'5 Epoch {epoch + 1}, Step {idx + 1} took {time.time() - step_start_time:.4f} seconds')

            tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy
            # print(f'6 Epoch {epoch + 1}, Step {idx + 1} took {time.time() - step_start_time:.4f} seconds')
            # gradient clipping
            # 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=MAX_GRAD_NORM
            )
            # print(f'7 Epoch {epoch + 1}, Step {idx + 1} took {time.time() - step_start_time:.4f} seconds')

            # backward pass
            optimizer.zero_grad()  # 将上次迭代的清零，时间忽略不计
            # print(f'7.1 Epoch {epoch + 1}, Step {idx + 1} took {time.time() - step_start_time:.4f} seconds')
            loss.backward()  # 最花时间，受模型复杂度和batch size影响
            # print(f'7.2 Epoch {epoch + 1}, Step {idx + 1} took {time.time() - step_start:.4f} seconds')

            step_start = time.time()
            optimizer.step()  # 比较花时间，受模型复杂度和优化算法影响
            scheduler.step()
            print(f'7.3 Epoch {epoch + 1}, Step {idx + 1} took {time.time() - step_start:.4f} seconds')
            metric.append(time.time() - step_start)
            step_end_time = time.time()  # End timer for the step
            print(f'Epoch {epoch + 1}, Step {idx + 1} took {step_end_time - step_start_time:.4f} seconds')
            metric.append(step_end_time - step_start_time)
            metric_csv.append([item for item in metric])
            # with open("basic"+str(idx)+".csv", mode='w', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerows(metric_csv)
            #     print("创建csv文件成功")

        print("total step:", len(training_loader))
        print("params", TRAIN_BATCH_SIZE, LEARNING_RATE, MAX_GRAD_NORM)
        with open("basic_batch_32_direct_gpu.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(metric_csv)
            print("创建csv文件成功")

        tr_accuracy = tr_accuracy / nb_tr_steps
        # print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

    for epoch in range(1):
        print(f"Training epoch: {epoch + 1}")
        train(epoch)

    # 如何构造Dataloder
    # 1. 依赖Dataset
    # 2. Dataset实现getitem，每个样本返回一个输入和label
    # 3. 训练时是对Dataset进行遍历
    # def valid(model, testing_loader):
    #     # put model in evaluation mode
    #     model.eval()
    #
    #     eval_loss, eval_accuracy = 0, 0
    #     nb_eval_examples, nb_eval_steps = 0, 0
    #     eval_preds, eval_labels = [], []
    #
    #     with torch.no_grad():
    #         for idx, batch in enumerate(testing_loader):
    #
    #             ids = batch['ids'].to(device, dtype=torch.long)
    #             mask = batch['mask'].to(device, dtype=torch.long)
    #             targets = batch['targets'].to(device, dtype=torch.long)
    #
    #             outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
    #             loss, eval_logits = outputs.loss, outputs.logits
    #
    #             eval_loss += loss.item()
    #
    #             nb_eval_steps += 1
    #             nb_eval_examples += targets.size(0)
    #
    #             if idx % 100 == 0:
    #                 loss_step = eval_loss / nb_eval_steps
    #                 print(f"Validation loss per 100 evaluation steps: {loss_step}")
    #
    #             # compute evaluation accuracy
    #             flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
    #             active_logits = eval_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
    #             flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)
    #             # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
    #             active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
    #             targets = torch.masked_select(flattened_targets, active_accuracy)
    #             predictions = torch.masked_select(flattened_predictions, active_accuracy)
    #
    #             eval_labels.extend(targets)
    #             eval_preds.extend(predictions)
    #
    #             tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
    #             eval_accuracy += tmp_eval_accuracy
    #
    #     # print(eval_labels)
    #     # print(eval_preds)
    #
    #     labels = [id2label[id.item()] for id in eval_labels]
    #     predictions = [id2label[id.item()] for id in eval_preds]
    #
    #     # print(labels)
    #     # print(predictions)
    #
    #     eval_loss = eval_loss / nb_eval_steps
    #     eval_accuracy = eval_accuracy / nb_eval_steps
    #     print(f"Validation Loss: {eval_loss}")
    #     print(f"Validation Accuracy: {eval_accuracy}")
    #
    #     return labels, predictions
    #
    # labels, predictions = valid(model, testing_loader)
    # from seqeval.metrics import classification_report
    #
    # print(classification_report([labels], [predictions]))

    ### 查找lr
    # class TokenClassifierLoss(torch.nn.Module):
    #     def __init__(self):
    #         super(TokenClassifierLoss, self).__init__()
    #
    #     def forward(self, model_output, labels):
    #         logits = model_output.logits
    #
    #         # 计算交叉熵损失
    #         loss_fn = torch.nn.CrossEntropyLoss()
    #         # 需要将 logits 和 labels 调整为合适的形状
    #         loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    #         return loss
    # criterion = TokenClassifierLoss()
    # lr_finder = LRFinder(model, optimizer, criterion, device="mps")
    # class CustomTrainIter(TrainDataLoaderIter):
    #     def inputs_labels_from_batch(self, batch_data):
    #         return batch_data["ids"], batch_data["targets"]
    # class CustomValIter(ValDataLoaderIter):
    #     def inputs_labels_from_batch(self, batch_data):
    #         return batch_data["ids"], batch_data["targets"]
    # lr_finder.range_test(CustomTrainIter(training_loader), val_loader=CustomValIter(testing_loader), end_lr=1, num_iter=100,
    #                      step_mode="exp")
    # lr_finder.plot(log_lr=False)
    # lr_finder.reset()


if __name__ == '__main__':
    main()
