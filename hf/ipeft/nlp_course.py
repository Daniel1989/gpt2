# from datasets import load_dataset
# from transformers import AutoTokenizer, DataCollatorWithPadding
# import evaluate
# import numpy as np
#
# raw_datasets = load_dataset("glue", "mrpc")
# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
#
# def tokenize_function(example):
#     return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
#
#
# tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#
# from transformers import TrainingArguments
#
# training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
# from transformers import AutoModelForSequenceClassification
#
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
#
# from transformers import Trainer
#
#
# def compute_metrics(eval_preds):
#     metric = evaluate.load("glue", "mrpc")
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
#
#
# trainer = Trainer(
#     model,
#     training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )
#
# trainer.train()
# model.save_pretrained("output_dir")
#
from transformers import pipeline

pipe = pipeline("text-classification", model="output_dir", device='mps')
print(pipe(
    "how are you",
    max_length=200))