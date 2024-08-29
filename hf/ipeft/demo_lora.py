from datasets import load_dataset
from peft import PeftModel

ds = load_dataset("food101")

labels = ds["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

def preprocess_train(example_batch):
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

train_ds = ds["train"]
val_ds = ds["validation"]

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

import torch

# def collate_fn(examples):
#     pixel_values = torch.stack([example["pixel_values"] for example in examples])
#     labels = torch.tensor([example["label"] for example in examples])
#     return {"pixel_values": pixel_values, "labels": labels}
#
#
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
#
# from peft import LoraConfig, get_peft_model
#
# config = LoraConfig(
#     r=16,
#     lora_alpha=16,
#     target_modules=["query", "value"],
#     lora_dropout=0.1,
#     bias="none",
#     modules_to_save=["classifier"],
# )
# model = get_peft_model(model, config)
# model.print_trainable_parameters()
#
# from transformers import TrainingArguments, Trainer
#
# account = "stevhliu"
# peft_model_id = f"{account}/google/vit-base-patch16-224-in21k-lora"
# batch_size = 32
#
# args = TrainingArguments(
#     output_dir='output_lora',
#     remove_unused_columns=False,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-3,
#     per_device_train_batch_size=batch_size,
#     gradient_accumulation_steps=4,
#     per_device_eval_batch_size=batch_size,
#     # fp16=True,
#     num_train_epochs=1,
#     # logging_steps=10,
#     load_best_model_at_end=True,
#     label_names=["labels"],
# )
#
# trainer = Trainer(
#     model,
#     args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     tokenizer=image_processor,
#     data_collator=collate_fn,
# )
# trainer.train()

from PIL import Image
import requests
image = Image.open("./hf/ipeft/beignets.jpeg")
encoding = image_processor(image.convert("RGB"), return_tensors="pt")
model = PeftModel.from_pretrained(model, model_id="./output_lora/checkpoint-592")
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])