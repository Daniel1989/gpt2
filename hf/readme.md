## BERT
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
BERT 是一种具有绝对位置嵌入的模型，因此通常建议将输入填充到右侧而不是左侧。
BERT 使用掩码语言模型 (MLM) 和下一句预测 (NSP) 目标进行训练。一般来说，它在预测屏蔽标记和 NLU 方面非常有效，但对于文本生成来说并不是最佳选择。
它是文本分类、命名实体识别、提取问答等领域广泛使用的模型。
BERT 是一个针对 2 个任务进行预训练的神经网络：掩码语言建模和下一个句子预测

### 命名实体识别
1. IOB格式
B-PERS 是一个人的开头，I-PERS 表示单词“Obama”在人体内，“O”表示单词“was”在命名实体外部