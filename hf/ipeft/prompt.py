import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model = 'bigscience/bloomz-560m'
model = AutoModelForCausalLM.from_pretrained(base_model).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(base_model)

# from datasets import load_dataset
# ds = load_dataset("ought/raft", "twitter_complaints")
# print(ds["train"][10])
# classes = [k.replace("_", " ") for k in ds["train"].features["Label"].names]
# print(classes)
# ds = ds.map(
#     lambda x: {"text_label": [classes[label] for label in x["Label"]]},
#     batched=True,
#     num_proc=1
# )
#
# print(ds["train"][10])
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id
# target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
# print(target_max_length)
#
# import torch
#
# max_length = 64
#
# def preprocess_function(examples, text_column="Tweet text", label_column="text_label"):
#     batch_size = len(examples[text_column])
#     inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
#     targets = [str(x) for x in examples[label_column]]
#     model_inputs = tokenizer(inputs)
#     labels = tokenizer(targets)
#     classes = [k.replace("_", " ") for k in ds["train"].features["Label"].names]
#     for i in range(batch_size):
#         sample_input_ids = model_inputs["input_ids"][i]
#         label_input_ids = labels["input_ids"][i]
#         model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
#             max_length - len(sample_input_ids)
#         ) + sample_input_ids
#         model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
#             "attention_mask"
#         ][i]
#         labels["input_ids"][i] = [-100] * (max_length - len(label_input_ids)) + label_input_ids
#         model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
#         model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
#         labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
#
# processed_ds = ds.map(
#     preprocess_function,
#     batched=True,
#     num_proc=1,
#     remove_columns=ds["train"].column_names,
#     load_from_cache_file=False,
#     desc="Running tokenizer on dataset",
# )
#
# from torch.utils.data import DataLoader
# from transformers import default_data_collator
#
# train_ds = processed_ds["train"]
# eval_ds = processed_ds["test"]
#
# batch_size = 16
#
# train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
# eval_dataloader = DataLoader(eval_ds, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
#
# from peft import PromptEncoderConfig, get_peft_model
#
# peft_config = PromptEncoderConfig(task_type="CAUSAL_LM", num_virtual_tokens=20, encoder_hidden_size=128)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()
#
# from transformers import get_linear_schedule_with_warmup
#
# lr = 3e-2
# num_epochs = 1
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# lr_scheduler = get_linear_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=(len(train_dataloader) * num_epochs),
# )
#
# from tqdm import tqdm
#
# device = "mps"
# model = model.to(device)
#
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     for step, batch in enumerate(tqdm(train_dataloader)):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         total_loss += loss.detach().float()
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#
#     model.eval()
#     eval_loss = 0
#     eval_preds = []
#     for step, batch in enumerate(tqdm(eval_dataloader)):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**batch)
#         loss = outputs.loss
#         eval_loss += loss.detach().float()
#         eval_preds.extend(
#             tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
#         )
#
#     eval_epoch_loss = eval_loss / len(eval_dataloader)
#     eval_ppl = torch.exp(eval_epoch_loss)
#     train_epoch_loss = total_loss / len(train_dataloader)
#     train_ppl = torch.exp(train_epoch_loss)
#     print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
#
# model.save_pretrained("output_prompt")

inputs = tokenizer(f'Tweet text : @NYTsupport i have complained a dozen times &amp; yet my papers are still thrown FAR from my door. Why is this so hard to resolve? Label : ', return_tensors="pt").to('cpu')
# with torch.no_grad():
#     inputs = {k: v.to('cpu') for k, v in inputs.items()}
#     outputs = model(input_ids=inputs["input_ids"])
#     print(outputs.logits.shape)
#     outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
#     print(outputs.shape)
#     print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))


model = PeftModel.from_pretrained(model, model_id="./output_prompt")
model.to('cpu')
with torch.no_grad():
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    outputs = model(input_ids=inputs["input_ids"])
    print(outputs.logits.shape)
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(outputs.shape)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

