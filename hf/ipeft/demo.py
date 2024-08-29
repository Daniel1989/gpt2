import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
import os
from transformers import pipeline, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, AutoTokenizer, \
    DataCollatorForSeq2Seq, pipelines
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

# dataset = load_dataset("json", data_files="demo.jsonl")
# dataset = dataset['train'].train_test_split(test_size=0.2)
# print(dataset)
# print(dataset['train'].features)
# print(len(dataset['train']))
# print(dataset['train'][10])
# tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-small")
#
#
# def tokenize_function(examples):
#     "mt0是一个encoder-decoder模型, 所以需要额外添加一个labels"
#     padding = "max_length"
#     max_length = 200
#     model_inputs = tokenizer("how are you", max_length=max_length, padding=padding, truncation=True)
#     labels = tokenizer("I'm 13 years old", max_length=max_length, padding=padding, truncation=True)
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
#
#
# tokenized_datasets = dataset.map(tokenize_function, batched=False)
# print(tokenized_datasets)
#
#
# model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-small", device_map="auto", torch_dtype=torch.float16) # 16位精度，对应2.98g
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()
# training_args = TrainingArguments(
#     output_dir="output/bigscience/mt0-large-lora",
#     learning_rate=1e-3,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     num_train_epochs=1,
#     weight_decay=0.01,
#     # evaluation_strategy="epoch",
#     # save_strategy="epoch",
#     # load_best_model_at_end=True,
# )
# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
#
#
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = logits.argmax(axis=-1)  # Convert logits to predicted labels
#     accuracy = accuracy_score(labels, predictions)
#     f1 = f1_score(labels, predictions, average='weighted')
#     return {
#         'accuracy': accuracy,
#         'f1': f1
#     }
#
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     # compute_metrics=compute_metrics,
# )
#
# trainer.train()
# # eval_results = trainer.evaluate()
# # print("Evaluation results:", eval_results)
# model.save_pretrained("output_dir")

# validate
# tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-small")
# model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-small", device_map="cpu")
# lora_model = PeftModel.from_pretrained(model, "./output_dir", device_map="cpu")
# # 由于pipeline识别不出了通过lorawrapper的模型，所以有告警。最好直接使用模型
# pipe = pipeline("text2text-generation", model=lora_model, tokenizer=tokenizer)
# print(pipe(
#     "how are you",
#     max_length=200))
# supported_tasks = pipelines.SUPPORTED_TASKS
#
# # Print all supported tasks
# for task in supported_tasks:
#     print(task)


# tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-small")
# # 加载模型
# model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-small", device_map="auto")
# # 加载lora权重
# model = PeftModel.from_pretrained(model, model_id="./output_dir")
# model.to('mps')
# model_inputs = tokenizer(["how are you"], return_tensors="pt")
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=200
# )
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)
