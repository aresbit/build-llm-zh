# 从零开始构建大语言模型 (Build a Large Language Model From Scratch) · 中译版

> 原书: Build a Large Language Model (From Scratch)
> 作者: Sebastian Raschka
> 出版社: Manning Publications, 2024

## 关于本书

本书是 Raschka《Build a Large Language Model (From Scratch)》的完整中文译本，采用代码优先方法，从零实现一个 GPT 模型——涵盖文本数据处理、注意力机制、Transformer 架构、预训练、微调分类与指令遵循的全流程。

本译本在原作基础上，**对关键的数学推导进行了大幅扩写**，包括：
- 自注意力机制的完整矩阵代数推导（Q/K/V 权重、缩放点积、softmax）
- 多头注意力的维度分析与并行拆分证明
- Layer Normalization 的统计推导
- GELU 激活函数的数值近似
- 交叉熵损失与困惑度的信息论连接
- 温度缩放与 top-k 采样的形式化
- 梯度下降与 Adam 优化器的完整推导

## 目录

| 章节 | 标题 | 内容 |
|------|------|------|
| 00 | [前言](ch00_前言.md) | 作者序、关于本书、PyTorch入门 |
| 01 | [理解大语言模型](ch01_理解LLM.md) | LLM定义、Transformer架构、GPT详解 |
| 02 | [处理文本数据](ch02_文本数据.md) | Token化、BPE、词嵌入、位置编码 |
| 03 | [注意力机制](ch03_注意力.md) | 自注意力、因果注意力、多头注意力 |
| 04 | [实现GPT模型](ch04_GPT实现.md) | LayerNorm、GELU、残差连接、文本生成 |
| 05 | [预训练](ch05_预训练.md) | 损失函数、温度缩放、top-k、权重加载 |
| 06 | [微调分类](ch06_微调分类.md) | 分类头、监督微调、垃圾邮件检测 |
| 07 | [指令微调](ch07_指令微调.md) | 指令数据、监督微调、模型评估 |
| 08 | [附录](ch08_附录.md) | PyTorch、参考文献、习题解答、LoRA |

## 术语对照

详见 [术语对照表](glossary.md)。