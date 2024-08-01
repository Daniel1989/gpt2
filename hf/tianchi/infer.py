import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

model_path = './Qwen/Qwen2-1___5B-Instruct'
lora_path = "./output/Qwen2-1_5B-Instruct_novel_all/final"
device = torch.device('cpu')

max_new_tokens = 2048

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)


# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

model.to(device)
## 傲慢与偏见 data[0]
    # {
    #     "instruction": "你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。",
    #     "input": "一个有钱的单身汉必定想要娶妻，这是社会普遍认同的事实。班纳特太太兴奋地告诉丈夫，尼日斐花园被一位名叫彬格莱的富绅租下，她希望他能成为女儿们的潜在配偶，而班纳特先生则以幽默的方式回应她的期望。",
    #     "output": "凡是有钱的单身汉，总想娶位太太，这已经成了一条举世公认的真理。这样的单身汉，每逢新搬到一个地方，四邻八舍虽然完全不了解他的性情如何，见解如何，可是，既然这样的一条真理早已在人们心目中根深蒂固，因此人们总是把他看作自己某一个女儿理所应得的一笔财产。\n有一天班纳特太太对她的丈夫说：“我的好老爷，尼日斐花园终于租出去了，你听说过没有？”班纳特先生回答道，他没有听说过。\n“的确租出去了，”她说，“朗格太太刚刚上这儿来过，她把这件事的底细，一五一十地告诉了我。”班纳特先生没有理睬她。\n“你难道不想知道是谁租去的吗？”太太不耐烦地嚷起来了。\n“既是你要说给我听，我听听也无妨。”这句话足够鼓励她讲下去了。\n“哦！亲爱的，你得知道，郎格太太说，租尼日斐花园的是个阔少爷，他是英格兰北部的人；听说他星期一那天，乘着一辆驷马大轿车来看房子，看得非常中意，当场就和莫理斯先生谈妥了；他要在‘米迦勒节’以前搬进来，打算下个周未先叫几个佣人来住。”“这个人叫什么名字？”“彬格莱。”“有太太的呢，还是单身汉？”“噢！是个单身汉，亲爱的，确确实实是个单身汉！一个有钱的单身汉；每年有四五千磅的收入。真是女儿们的福气！”“这怎么说？关女儿女儿们什么事？”“我的好老爷，”太太回答道，“你怎么这样叫人讨厌！告诉你吧，我正在盘算，他要是挑中我们一个女儿做老婆，可多好！”“他住到这儿来，就是为了这个打算吗？”“打算！胡扯，这是哪儿的话！不过，他倒作兴看中我们的某一个女儿呢。他一搬来，你就得去拜访拜访他。”“我不用去。你带着女儿们去就得啦，要不你干脆打发她们自己去，那或许倒更好些，因为你跟女儿们比起来，她们哪一个都不能胜过你的美貌，你去了，彬格莱先生倒可能挑中你呢？”“我的好老爷，你太捧我啦。从前也的确有人赞赏过我的美貌，现在我可有敢说有什么出众的地方了。一个女人家有了五个成年的女儿，就不该对自己的美貌再转什么念头。”“这样看来，一个女人家对自己的美貌也转不了多少念头喽。"
    # },


prompt = "一个有钱的单身汉必定想要娶妻，这是社会普遍认同的事实。班纳特太太兴奋地告诉丈夫，尼日斐花园被一位名叫彬格莱的富绅租下，她希望他能成为女儿们的潜在配偶，而班纳特先生则以幽默的方式回应她的期望。"
messages = [
    {"role": "system", "content": "你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt")

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=max_new_tokens
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
# PYTORCH_ENABLE_MPS_FALLBACK = 1

new_model_directory = "./merged_model"
merged_model = model.merge_and_unload()
# 将权重保存为safetensors格式的权重, 且每个权重文件最大不超过2GB(2048MB)
merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)
