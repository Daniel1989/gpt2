import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
# torch.set_default_device("mps")  # <-------- MPS backend

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# 加载 GPT-2 模型和分词器
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 将模型设置为评估模式并移动到 GPU（如果可用）
device = torch.device('mps')
model.to(device)
model.eval()

# 定义输入文本
input_text = "从前有座山"

# 使用分词器对输入文本进行编码
inputs = tokenizer.encode(input_text, return_tensors='pt')
inputs = inputs.to(device)
#
# # 生成文本
with torch.no_grad():
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)

# # 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)