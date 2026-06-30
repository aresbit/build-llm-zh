# 第1章: 理解大语言模型 (Understanding Large Language Models)

本章为全书奠定基础: 我们首先给出大语言模型 (LLM) 的定义并建立其数学形式化, 随后概述 LLM 的典型应用场景, 接着介绍构建与使用 LLM 的三大阶段, 深入剖析 Transformer 架构, 讨论大规模训练数据集的规模与构成, 聚焦 GPT 架构的设计细节, 并最后给出从零构建一个 LLM 的完整路线图。

## 1.1  什么是大语言模型 (LLM)

一个大语言模型 (Large Language Model, LLM) 是一个深度神经网络, 设计目标是理解、生成和响应人类文本。这些模型在海量文本数据上训练, 有时几乎涵盖了互联网上所有的公开文本。

LLM 中的 "大" (Large) 指两个维度: 模型参数量 (数十亿至数千亿) 和训练数据规模 (万亿级 token)。模型的参数是在训练过程中优化的可调权重, 其核心任务看似简单: **预测序列中的下一个词**。

### 1.1.1  形式化定义: 自回归语言模型

从数学上严格定义, 一个 LLM 是一个自回归语言模型, 它将一个 token 序列的概率分解为条件概率的乘积。给定一个 token 序列 $$\mathbf{x} = (x_1, x_2, \ldots, x_T)$$, 模型对该序列的联合概率为:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta)$$

其中 $$\theta$$ 表示模型的所有参数。这个公式的含义是: 序列中每个 token 的概率仅依赖于它之前的所有 token (自回归性质)。在实践中, 模型并不显式计算整个联合概率, 而是将下一 token 预测作为训练目标。对于参数 $$\theta$$, 训练目标是最小化负对数似然 (Negative Log-Likelihood, NLL):

$$\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)$$

其中 $$x_{<t} = (x_1, \ldots, x_{t-1})$$ 表示第 $$t$$ 个位置之前的所有 token。最小化 $$\mathcal{L}(\theta)$$ 等价于最大化模型给训练数据分配的概率——这正是机器学习和深度学习中标准的经验风险最小化原则。

下一词预测任务是自监督学习的一种形式: 不需要人工标注, 因为 "标签" 就是输入序列本身的下一个词。这种自监督特性使得 LLM 能够从海量未标注文本中学习, 而无需昂贵的人工标注。

下一词预测之所以能产生如此强大的模型, 原因在于它迫使模型学习语言的内在结构: 语法、语义、常识、推理模式, 甚至某种形式的 "世界知识", 这一切都编码在从输入上下文中推断下一词的统计规律中。

从更宏观的视角看, LLM 的成功可归因于两个因素: Transformer 架构和前所未有的数据规模。传统自然语言处理 (NLP) 模型通常为特定任务设计——如文本分类或机器翻译——并且依赖于人工设计的特征 (规则、关键词频率、句法树等)。LLM 则颠覆了这一范式: 同一个模型 (仅通过改变输入提示) 可以执行多种任务, 且不再需要人工特征工程。深度神经网络自动从原始文本中学习层次化的特征表示——从低层的词法模式到高层的语义抽象。

LLM 中的 "深度" 不仅指层数多 (数十至数百层), 更指模型习得的表示具有层级结构。浅层捕捉局部的词法和句法特征, 中间层编码语义角色和实体关系, 深层则可能编码更高层次的常识推理和抽象概念。这种逐层抽象的能力正是深度学习的核心优势。

## 1.2  LLM 的应用

凭借解析和理解非结构化文本的强大能力, LLM 在众多领域有广泛应用: 机器翻译、文本生成、情感分析、文本摘要、内容创作 (小说、文章、代码) 等。LLM 还可以驱动机器人聊天助手和虚拟助手, 如 OpenAI 的 ChatGPT 和 Google 的 Gemini, 它们能回答用户查询并增强传统搜索引擎。

在医学和法律等专业领域, LLM 可用于从海量文档中检索知识、摘要长篇文章、回答技术问题。简言之, LLM 几乎可以自动化任何涉及文本解析和生成的任务。

本书将聚焦于从底层理解 LLM 的工作机制——从零编码一个能生成文本的 LLM, 并逐步学习如何使其执行问答、摘要、翻译等任务。

## 1.3  构建与使用 LLM 的阶段

构建 LLM 的一般过程包括 **预训练** (pretraining) 和 **微调** (fine-tuning) 两大阶段。"预" (pre) 指模型首先在大规模多样化数据集上进行初始训练, 以获得对语言的广泛理解。得到的预训练模型被称为 **基座模型** (foundation model), 随后可通过微调在更窄、更特定任务的数据集上进一步优化。

最流行的两类微调范式是 **指令微调** (instruction fine-tuning) 和 **分类微调** (classification fine-tuning)。指令微调的标注数据由指令-回答对构成, 分类微调的标注数据由文本及其类别标签构成 (如 "垃圾邮件" / "非垃圾邮件")。

预训练阶段使用自监督学习——模型从输入数据本身生成标签, 无需人工标注。微调阶段则使用有监督学习, 需要标注数据。这种两阶段训练策略的有效性源于迁移学习: 预训练阶段学到的通用语言知识作为强先验, 使得微调阶段只需少量任务特定数据即可实现高性能。

## 1.4  Transformer 架构

大多数现代 LLM 基于 **Transformer** 架构, 这是 Vaswani 等人在 2017 年论文 "Attention Is All You Need" 中提出的深度神经网络架构。

### 1.4.1  编码器-解码器结构

原始 Transformer 由两个子模块组成: **编码器** (encoder) 和 **解码器** (decoder)。编码器处理输入文本并将其编码为捕获上下文信息的一系列数值表示 (向量); 解码器接收这些编码后的向量并生成输出文本。在翻译任务中, 编码器将源语言文本编码为向量, 解码器将这些向量解码为目标语言文本。

编码器和解码器均由多个层堆叠而成, 通过 **自注意力机制** (self-attention mechanism) 连接。自注意力使得模型在处理序列中某个位置的 token 时, 能够加权关注序列中所有其他位置——这正是 LLM 捕获长距离依赖和上下文关系的关键。

在自注意力出现之前, 序列建模的主流方法是循环神经网络 (RNN) 和长短期记忆网络 (LSTM)。这些架构按时间步串行处理输入, 导致两个致命缺陷: (1) 长序列中早期信息被逐渐稀释 (梯度消失), (2) 无法并行计算 (每个时间步依赖前一步的输出)。Transformer 通过自注意力一举解决了这两个问题: 它允许序列中任意两个位置直接交互 (路径长度从 RNN 的 O(n) 降低到 O(1)), 并且所有位置的计算可以完全并行化。

### 1.4.2  Transformer 层的计算图: 扩展推导

一个标准的 Transformer 层 (无论是编码器层还是解码器层) 的计算图可以用以下公式完整描述。记输入为 $$\mathbf{x}$$:

**步骤 1: 多头自注意力 (Multi-Head Self-Attention)。**

对于输入表示 $$\mathbf{X} \in \mathbb{R}^{n \times d}$$ ($$n$$ 个 token, 每个 $$d$$ 维), 首先通过线性投影得到查询 (Query) $$\mathbf{Q}$$、键 (Key) $$\mathbf{K}$$、值 (Value) $$\mathbf{V}$$:

$$\mathbf{Q} = \mathbf{X} \mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}^K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}^V$$

其中 $$\mathbf{W}^Q, \mathbf{W}^K \in \mathbb{R}^{d \times d_k}$$, $$\mathbf{W}^V \in \mathbb{R}^{d \times d_v}$$ 是可学习的权重矩阵。

缩放点积注意力 (Scaled Dot-Product Attention) 定义为:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

除以 $$\sqrt{d_k}$$ 的作用是防止点积过大导致 softmax 梯度消失。在多头注意力中, 我们并行计算 $$h$$ 个注意力 "头":

$$\text{head}_i = \text{Attention}(\mathbf{X} \mathbf{W}_i^Q, \mathbf{X} \mathbf{W}_i^K, \mathbf{X} \mathbf{W}_i^V)$$

$$\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O$$

其中 $$\mathbf{W}^O \in \mathbb{R}^{h d_v \times d}$$ 是输出投影矩阵。

**步骤 2: 残差连接与层归一化 (Add & Norm)。**

$$\mathbf{X}_{\text{attn}} = \text{LayerNorm}\big(\mathbf{X} + \text{MultiHead}(\mathbf{X})\big)$$

LayerNorm 对每个 token 的特征维度进行归一化: 对于向量 $$\mathbf{z} \in \mathbb{R}^d$$,

$$\text{LayerNorm}(\mathbf{z}) = \gamma \odot \frac{\mathbf{z} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中 $$\mu = \frac{1}{d}\sum_{i=1}^{d} z_i$$, $$\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (z_i - \mu)^2$$, $$\gamma, \beta \in \mathbb{R}^d$$ 是可学习的缩放和偏移参数, $$\epsilon$$ 是小常数。

**步骤 3: 前馈网络 (Feed-Forward Network, FFN)。**

$$\mathbf{X}_{\text{ffn}} = \text{LayerNorm}\big(\mathbf{X}_{\text{attn}} + \text{FFN}(\mathbf{X}_{\text{attn}})\big)$$

其中 FFN 通常为两层全连接网络:

$$\text{FFN}(\mathbf{z}) = \text{GELU}(\mathbf{z} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$

**完整的一层 Transformer 计算图。** 将上述步骤串联, 一个完整的 Transformer 层形式为:

$$\mathbf{X}_{\text{out}} = \text{FFN}\big(\text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X}))\big)$$

带残差的精确形式是:

$$\mathbf{X}_{\text{out}} = \mathbf{X}_{\text{attn}} + \text{FFN}\big(\text{LayerNorm}(\mathbf{X}_{\text{attn}})\big)$$

其中 $$\mathbf{X}_{\text{attn}} = \mathbf{X} + \text{MultiHead}\big(\text{LayerNorm}(\mathbf{X})\big)$$ (此处的 LayerNorm 位置采用 Pre-LN 风格, 即 GPT 系列实际使用的方案)。

这个残差结构的优雅之处在于: 当 FFN 或 Attention 子层学到无效的变换时, $$F(\mathbf{x}) \approx 0$$, 梯度仍可通过残差路径无衰减地传播, 这极大地缓解了深层网络的训练困难。

## 1.5  大规模数据集

GPT 和 BERT 类模型的大型训练数据集代表了多样化和全面的文本语料, 包含数十亿词, 涵盖广泛的主题和语言。以 GPT-3 为例, 其预训练数据集如表 1.1 所示。

| 数据集名称 | 描述 | Token 数量 | 训练数据占比 |
|-----------|------|-----------|------------|
| CommonCrawl (过滤后) | 网络爬取数据 | 4100 亿 | 60% |
| WebText2 | 网络爬取数据 | 190 亿 | 22% |
| Books1 | 互联网图书语料 | 120 亿 | 8% |
| Books2 | 互联网图书语料 | 550 亿 | 8% |
| Wikipedia | 高质量文本 | 30 亿 | 3% |

表 1.1 中的 token 是模型读取的文本单位, 大致等同于文本中的单词与标点符号数量之和。GPT-3 总计有 4990 亿 token 的数据集, 但论文指出其仅在所有 epoch 中共训练了 3000 亿 token。

预训练 LLM 需要巨大的计算资源。GPT-3 的预训练成本估计为 460 万美元 (云计算积分)。幸运的是, 许多预训练 LLM 以开源模型的形式提供, 可直接使用或进行微调。

### 1.5.1  扩展推导: 缩放定律 (Scaling Laws)

Kaplan 等人 (2020) 的经典缩放定律描述了模型性能 (以交叉熵损失 $$L$$ 度量) 与模型参数量 $$N$$ 和训练数据量 $$D$$ 之间的关系。损失函数具有以下函数形式:

$$L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}$$

其中:
- $$N_c$$ 和 $$D_c$$ 是归一化常数,
- $$\alpha_N \approx 0.076$$ 和 $$\alpha_D \approx 0.095$$ 是 Kaplan 等人经验拟合的指数,
- $$L(N, D)$$ 是模型的测试损失。

该公式揭示了损失随模型规模和数据规模分别以幂律衰减的规律。当 $$N \to \infty$$ 或 $$D \to \infty$$ 时, $$L(N, D) \to 0$$ (严格来说趋向某个不可约损失 $$L_{\text{irred}}$$), 但在实践中遵循幂律而非指数衰减——这意味着要获得线性程度的损失下降, 需要指数级地增加计算量。

### 1.5.2  扩展推导: 训练计算量与 Chinchilla 缩放定律

总训练计算量 (以 FLOPs 计) 可以用模型参数量 $$N$$ 和训练 token 数 $$D$$ 近似计算。对于 Transformer 架构, 一次前向传播和一次反向传播的计算成本分别为:

- 前向传播 FLOPs: $$\approx 2ND$$
- 反向传播 FLOPs: $$\approx 4ND$$

因此, 每次完整的前向 + 反向传播总共约需 $$6ND$$ 次 FLOPs。总训练计算量为:

$$C \approx 6ND$$

Kaplan 的结论是: 在固定计算预算 $$C$$ 下, 应优先增大模型规模 $$N$$ 而非训练数据量 $$D$$。但 Hoffmann 等人 (2022) 的 **Chinchilla 缩放定律** 修正了这一结论: 在计算最优配置下, 模型参数量 $$N$$ 和训练 token 数 $$D$$ 应 **等比例** 增长:

$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}$$

这意味着对一个 $$N$$ 参数的模型, 最优训练 token 数约为 $$D \approx 20N$$。换句话说, 许多早期大模型实际上是 **训练不足** (undertrained) 的——它们有足够大的参数量, 但训练数据不够多。Chinchilla 定律深刻地影响了后续 LLM (如 LLaMA 系列) 的训练策略, 使其在较小的模型上使用更多数据进行训练以提升效率。

将缩放定律与计算预算理解为一个整体决策框架: 给定 $$C$$ 美元的计算预算, 开发者需要在模型大小和数据量之间做出最优权衡。Kaplan 定律建议 "大模型, 一般数据", 而 Chinchilla 定律建议 "中等模型, 海量数据"。实际工程实践已经向 Chinchilla 方向收敛——例如 LLaMA-2 70B 在 2 万亿 token 上训练, 其 $$D/N$$ 比率远高于 GPT-3。

## 1.6  深入 GPT 架构

GPT (Generative Pretrained Transformer) 最初由 OpenAI 的 Radford 等人在论文 "Improving Language Understanding by Generative Pre-Training" 中提出。GPT-3 是该模型的放大版, 拥有更多参数并训练于更大数据集。ChatGPT 则通过指令微调 GPT-3 得到。

### 1.6.1  仅解码器架构

与 1.4 节介绍的原始 Transformer 不同, GPT 架构只使用 **解码器部分**, 不包含编码器。它是:

- **单向 (左到右) 处理**: 每个 token 只能关注其左侧的 token,
- **自回归生成**: 将先前的输出作为下一步的输入, 逐词生成。

GPT-3 拥有 96 个 Transformer 层和 1750 亿参数——远远大于原始 Transformer (仅 6 层编码器 + 6 层解码器)。尽管规模庞大, 其架构原理与原始 Transformer 的解码器基本相同。更近期的架构 (如 Meta 的 LLaMA) 仍基于相同的底层概念, 仅作了微小修改。

### 1.6.2  扩展推导: GPT 与 BERT 的架构差异——因果掩码 vs 双向注意力

GPT 和 BERT 的核心架构差异体现在自注意力机制中注意力矩阵的掩码方式上。

**GPT (因果注意力 / Causal Attention)。** GPT 使用因果掩码 (causal mask), 确保位置 $$i$$ 只能关注位置 $$j \leq i$$。注意力权重矩阵 $$\mathbf{A} \in \mathbb{R}^{n \times n}$$ 定义为:

$$\mathbf{A} = \text{softmax}\!\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)$$

其中因果掩码矩阵 $$\mathbf{M}$$ 定义为:

$$M_{ij} = \begin{cases}
0 & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}$$

经过 softmax 后, $$\exp(-\infty) = 0$$, 因此位置 $$i$$ 对位置 $$j > i$$ (未来 token) 的注意力权重为零。这保证了自回归性质——模型在预测当前位置时无法 "偷看" 未来的词。

**BERT (双向注意力)。** BERT 不使用任何掩码 (或等价地, $$\mathbf{M} = \mathbf{0}$$), 每个 token 可以关注序列中所有其他 token (包括其前面和后面的 token)。注意力矩阵为:

$$\mathbf{A}_{\text{BERT}} = \text{softmax}\!\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right)$$

这种双向注意力使得 BERT 能从前后的完整上下文中捕获信息, 因而特别适合 **理解类任务** (如文本分类、命名实体识别、问答), 但不适合自回归文本生成。

总结两种范式的形式化差异:

| 特性 | GPT (因果注意力) | BERT (双向注意力) |
|------|----------------|------------------|
| 注意力掩码 | $$M_{ij} = -\infty \text{ if } j > i$$ | $$M_{ij} = 0 \;\; \forall i, j$$ |
| 上下文方向 | 单向 (左到右) | 双向 (左右同时) |
| 训练目标 | 下一词预测 (自回归) | 掩码词预测 (完形填空) |
| 适用任务 | 文本生成、翻译、对话 | 文本分类、序列标注、提取式问答 |
| 架构组成 | 仅 Transformer 解码器 | 仅 Transformer 编码器 |

### 1.6.3  涌现行为

GPT 模型表现出 **涌现行为** (emergent behavior): 它们能够执行训练时未显式设计的目标任务。例如, GPT 主要为下一词预测而训练, 却能执行翻译、拼写纠正、分类等任务——这些能力并非被显式教授, 而是作为模型接触海量多语言数据后自然涌现的结果。这充分展示了大规模生成式语言模型的能力和潜力: 我们无需为每个任务单独设计和训练一个模型。

涌现行为的一个关键特征是: 它在模型规模超过某个临界阈值后才会显著出现。较小的模型 (如 GPT-2 Small, 1.17 亿参数) 在多个任务上表现平庸, 但 GPT-3 (1750 亿参数) 在相同任务上表现出令人惊讶的少样本和零样本泛化能力。这种在参数规模增长过程中突然出现的能力跃迁——而非随规模线性增长——正是 "涌现" 的核心含义, 也表明在某个规模阈值之上, 模型开始在训练数据中捕捉到更抽象的、跨任务的统计规律。

## 1.7  从零构建大语言模型的路线图

本书以 GPT 的核心思想为蓝图, 分三个阶段从零编码一个 LLM:

**阶段 1: 构建 LLM 架构和数据准备 (第 2-4 章)。** 学习数据预处理步骤并实现每个 LLM 核心的注意力机制, 最终构建出能生成文本的完整 GPT 类模型。

**阶段 2: 预训练基座模型 (第 5 章)。** 学习如何在小规模数据集上预训练 GPT 类 LLM——包括训练循环、损失函数设计、文本生成策略和模型评估方法。同时提供加载公开可用模型权重的代码, 使读者能够跳过大模型的昂贵预训练阶段。

**阶段 3: 微调 (第 6-7 章)。** 将预训练 LLM 微调为遵循指令的助手或文本分类器——这是现实世界应用和研究中最常见的任务。

预训练一个 GPT 类模型从零开始需要数千至数百万美元的计算成本, 因此阶段 2 的重点是使用小型数据集进行教育性预训练。通过加载公开可用的模型权重, 读者仍然可以获得在实际规模上进行微调的实践经验。

## 本章小结

- LLM 是一个 **自回归语言模型**, 形式化为条件概率链: $$P(\mathbf{x}) = \prod_{t=1}^{T} P(x_t \mid x_{<t}; \theta)$$。其训练目标是最小化负对数似然, 使用自监督学习从海量未标注文本中学习。
- 现代 LLM 的训练分为两大步骤: 首先在大型未标注文本语料上 **预训练** (通过下一词预测任务自监督), 然后在较小的标注数据集上 **微调** (指令微调或分类微调)。
- LLM 基于 Transformer 架构。Transformer 的核心是 **自注意力机制**, 它使模型在生成每个输出词时能选择性关注整个输入序列。完整的 Transformer 层计算图包括多头注意力、残差连接、层归一化和前馈网络: $$\mathbf{X}_{\text{out}} = \mathbf{X} + \text{FFN}(\text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X})))$$。
- GPT 只使用 Transformer 的 **解码器部分**, 通过 **因果掩码** 实现自回归生成, 保证位置 $$i$$ 不能关注位置 $$j > i$$。BERT 使用 **双向注意力**, 适合理解类任务但不适合生成。
- **缩放定律** (Kaplan 定律) 描述了损失随模型规模和数据规模的幂律衰减: $$L(N, D) = (N_c/N)^{\alpha_N} + (D_c/D)^{\alpha_D}$$。**Chinchilla 定律** 修正了最优训练配置: $$N_{\text{opt}} \propto C^{0.5}, D_{\text{opt}} \propto C^{0.5}$$, 其中 $$C \approx 6ND$$。
- 尽管 GPT 仅为下一词预测而训练, 这些模型展现出 **涌现属性**, 如分类、翻译和摘要能力——使一个模型无需重训练即可执行多种任务。
- 预训练后的基座模型可针对特定下游任务更高效地微调, 基于自定义数据集的微调 LLM 在特定任务上可以超越通用 LLM。

---

<!-- chapter-nav -->
<div style="display:flex; justify-content:space-between; align-items:center; padding:1em 0;">
  <div><a href="ch00_前言.md">← 前言</a></div>
  <div><a href="index.md">↑ 目录</a></div>
  <div><a href="ch02_文本数据.md">第2章 处理文本数据 →</a></div>
</div>
