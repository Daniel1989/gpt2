from transformers import TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer

# 加载保存的模型和分词器
model_dir = "output"
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 使用加载的模型进行推理
text_generator = TextGenerationPipeline(model, tokenizer, device='mps')
print(text_generator("以下是保持健康的三个提示", max_length=100, do_sample=True))
