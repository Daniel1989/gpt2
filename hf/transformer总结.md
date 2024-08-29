## 术语
架构: 这是模型的骨架 — 每个层的定义以及模型中发生的每个操作。

Checkpoints(参数): 这些是将在给架构中结构中加载的权重。

模型: 这是一个笼统的术语，没有“架构”或“参数”那么精确：它可以指两者。

例如，BERT是一个架构，而 bert-base-cased， 这是谷歌团队为BERT的第一个版本训练的一组权重参数，是一个参数。我们可以说“BERT模型”和”bert-base-cased模型.”

## 概述
语言模型 包含 transformer模型（GPT、BERT/DistilBERT、BART、T5, GPT-2, GPT-3等等，他们都是基于无监督的预训练模型）
其他模型，基于上述模型，进行finetune（监督训练得来的模型）。注意，不同的架构，会需要不同的tokenizer，即不同的输入

### GPT-like(decode-only)
称作自回归Transformer模型，或者causal 模型，即因果模型，只需要关注前面的单词。
1. 适合任务--适用于生成任务，如文本生成。
2. 训练方式--通常围绕预测句子中的下一个单词进行。
3. 类型--CTRL, GPT, GPT-2, Transformer XL

### BERT-like(encode-only)
称作自动编码Transformer模型，模型通常具有“双向”注意力。
1. 适合任务--适用于需要理解输入的任务，如句子分类、命名实体识别（以及更普遍的单词分类）和阅读理解后回答问题。
2. 训练方式--通常围绕着以某种方式破坏给定的句子（例如：通过随机遮盖其中的单词），并让模型寻找或重建给定的句子。
3. 类型--ALBERT BERT DistilBERT ELECTRA RoBERTa

### BART/T5-like(encoder-decode)
称作序列到序列的 Transformer模型, 在每个阶段，编码器的注意力层可以访问初始句子中的所有单词，而解码器的注意力层只能访问位于输入中将要预测单词前面的单词。
1. 适合任务--适用于生成任务，如文本生成。
2. 训练方式--训练可以使用训练编码器或解码器模型的方式来完成，但通常涉及更复杂的内容。例如，T5通过将文本的随机跨度（可以包含多个单词）替换为单个特殊单词来进行预训练，然后目标是预测该掩码单词替换的文本。
3. 类型--BART， mBART， Marian， T5

### 其他
1. ViT 视觉模型
2. CLIP 多模态模型
3. Whisper 自动语音识别模型

## 任务类型(影响到tokenizer)
1. 句子分类--识别垃圾邮件等等
2. 成分识别--提取实体
3. 生成文本--填充空白
4. 提取答案--回答问题
5. 文本到文本--翻译总结

### pipeline枚举
1. audio-classification
2. automatic-speech-recognition
3. text-to-audio
4. feature-extraction
5. text-classification
6. token-classification
7. question-answering
8. table-question-answering
9. visual-question-answering
10. document-question-answering
11. fill-mask
12. summarization
13. translation
14. text2text-generation
15. text-generation
16. zero-shot-classification
17. zero-shot-image-classification
18. zero-shot-audio-classification
19. image-classification
20. image-feature-extraction
21. image-segmentation
22. image-to-text
23. object-detection
24. zero-shot-object-detection
25. depth-estimation
26. video-classification
27. mask-generation
28. image-to-image

