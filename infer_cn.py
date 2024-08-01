import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
# torch.set_default_device("mps")  # <-------- MPS backend
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
from datasets import DatasetDict, load_dataset

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
#
text_generator = TextGenerationPipeline(model, tokenizer, device='mps')
print(text_generator("以下是保持健康的三个提示", max_length=100, do_sample=True))

# from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
# training_args = TrainingArguments("test-trainer", num_train_epochs=50, per_device_train_batch_size=10)
# context_length = 128
#
# def tokenize(element):
#     # print(element["output"])
#     outputs = tokenizer(
#         element["output"],
#         truncation=True,
#         max_length=context_length,
#         return_overflowing_tokens=True,
#         return_length=True,
#     )
#     input_batch = []
#     for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
#         if length == context_length:
#             input_batch.append(input_ids)
#     return {"input_ids": input_batch}
#
# ds_train = load_dataset("shibing624/alpaca-zh", split="train")
# # ds_train = ds_train[:1000]
# # ds_valid = load_dataset("shibing624/alpaca-zh", split="validation")
# print(len(ds_train))
# raw_datasets = DatasetDict(
#     {
#         "train": ds_train.select(range(1000)),  # .shuffle().select(range(50000)),
#         "valid": ds_train.select(range(1000)),  # .shuffle().select(range(500))
#     }
# )
# tokenized_datasets = raw_datasets.map(
#     tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
# )
#
# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
#
# trainer = Trainer(
#     model=model,
#     tokenizer=tokenizer,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["valid"],
# )
# trainer.train()
#
# model.save_pretrained("output")
# # 保存分词器（如果需要）
# tokenizer.save_pretrained("output")