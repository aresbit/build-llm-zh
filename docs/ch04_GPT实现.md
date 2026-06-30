# 第4章: 实现GPT模型 (Implementing a GPT Model)

本章从零开始实现一个完整的 GPT 模型，涵盖从占位符骨架到最终文本生成的全过程。我们将逐步编码 Layer Normalization、GELU 激活函数、前馈网络、残差连接和 Transformer 块，最终将它们组装为可工作的 GPT 架构，并编写文本生成循环。

## 4.1 编码 LLM 架构

### 4.1.1 配置字典

GPT-2 小型的参数如下:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,       # BPE 词表大小
    "context_length": 1024,    # 最大上下文长度
    "emb_dim": 768,            # 嵌入维度
    "n_heads": 12,             # 注意力头数
    "n_layers": 12,            # Transformer 块层数
    "drop_rate": 0.1,          # Dropout 比率
    "qkv_bias": False          # Q/K/V 线性层是否使用偏置
}
```

### 4.1.2 占位符架构 DummyGPTModel

为建立全局视野，我们先实现一个使用占位符组件的 GPT 骨架:

```python
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

数据流: Token ID $$\to$$ Token 嵌入 + 位置嵌入 $$\to$$ Dropout $$\to$$ 12 层 Transformer 块 $$\to$$ 最终 LayerNorm $$\to$$ 线性输出头 $$\to$$ Logits。对于两个文本输入 (各有 4 个 token)，输出形状为 `[2, 4, 50257]`。

## 4.2 Layer Normalization 激活归一化

### 4.2.1 动机

深层网络训练面临梯度消失和梯度爆炸问题。设第 $$l$$ 层的激活值为 $$h^{(l)}$$，若每层将信号放大或缩小一个因子 $$\lambda$$，则经过 $$L$$ 层后梯度为:

$$\frac{\partial \mathcal{L}}{\partial h^{(1)}} = \frac{\partial \mathcal{L}}{\partial h^{(L)}} \prod_{l=1}^{L-1} \frac{\partial h^{(l+1)}}{\partial h^{(l)}} \approx \frac{\partial \mathcal{L}}{\partial h^{(L)}} \cdot \lambda^{L-1}$$

当 $$\vert \lambda\vert  < 1$$ 时梯度消失，$$\vert \lambda\vert  > 1$$ 时梯度爆炸。层归一化通过将每层输出标准化为零均值、单位方差来解决此问题。

### 4.2.2 标准推导

设输入向量为 $$x = (x_1, x_2, \ldots, x_d) \in \mathbb{R}^d$$。计算:

$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$$

$$\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$$

归一化:

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

其中 $$\varepsilon$$ (例如 $$10^{-5}$$) 防止除以零。然后引入可学习的缩放和平移参数 $$\gamma, \beta \in \mathbb{R}^d$$:

$$\text{LN}(x)_i = \gamma_i \cdot \hat{x}_i + \beta_i$$

完整的 LayerNorm 运算:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta$$

### 4.2.3 与 Batch Normalization 的对比

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化轴 | 跨 batch 维度 (每个特征独立) | 跨特征维度 (每个样本独立) |
| 统计量计算 | $$\mu_c = \frac{1}{N}\sum_{n=1}^{N} x_{n,c}$$ | $$\mu_n = \frac{1}{d}\sum_{i=1}^{d} x_{n,i}$$ |
| 对 batch size 的依赖 | 强依赖; 小 batch 时不稳定 | 完全独立 |
| 适用场景 | CNN (计算机视觉) | Transformer / RNN (NLP) |

BatchNorm 对每个特征通道计算跨 batch 的统计量:

$$\mu_c = \frac{1}{N}\sum_{n=1}^{N} x_{n,c}, \quad \sigma_c^2 = \frac{1}{N}\sum_{n=1}^{N}(x_{n,c} - \mu_c)^2$$

LayerNorm 对每个样本的所有特征计算统计量，使其天然适合变长序列、小 batch 和分布式训练——这对于大语言模型至关重要。

### 4.2.4 代码实现

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

**参数说明**: `unbiased=False` 表示方差计算中除数使用 $$n$$ 而非 $$n-1$$ (贝塞尔校正)。对于嵌入维度为 768 的 LLM，此差异可忽略，但该选择确保与 OpenAI GPT-2 预训练权重的兼容性 (GPT-2 使用 TensorFlow 默认行为)。

## 4.3 GELU 激活与前馈网络

### 4.3.1 GELU 的数学定义

GELU (Gaussian Error Linear Unit) 定义为:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

其中 $$\Phi(x)$$ 是标准高斯分布的累积分布函数 (CDF):

$$\Phi(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right] = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-t^2/2} dt$$

直观理解: GELU 将输入 $$x$$ 乘以它大于随机高斯变量的概率 $$P(X \leq x)$$，从而对负值施加随机性正则化。

### 4.3.2 tanh 近似推导

由于精确计算 $$\Phi(x)$$ 开销较大，实践中使用基于 tanh 的近似 (通过曲线拟合得出):

$$\text{GELU}(x) \approx 0.5x \left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right)\right]$$

**推导思路**: 对 $$\Phi(x)$$ 进行一阶近似，注意到 $$\Phi(x) \approx \frac{1}{2} + \frac{1}{2}\tanh(\sqrt{2/\pi} \cdot x)$$，在此基础上加入三次项 $$0.044715 x^3$$ 以提高拟合精度。系数 $$0.044715$$ 是通过最小化近似值与精确值之间的误差得到的数值优化结果。

### 4.3.3 与 ReLU 的对比

| 特性 | ReLU | GELU |
|------|------|------|
| 定义 | $$\max(0, x)$$ | $$x \cdot \Phi(x)$$ |
| 光滑性 | 在 $$x=0$$ 处不可导 (尖角) | 处处光滑 |
| 负值行为 | 严格为零 | 允许小幅度非零输出 |
| 梯度 | 正值区域为常数 1 | 在全部定义域内连续变化 |

GELU 的光滑性共提供了更好的优化特性——参数可通过更细腻的梯度的调整来学习。负值区域的小非零输出意味着接收到负输入的神经元仍可持续参与学习，只是贡献减弱。

### 4.3.4 前馈网络

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```

数据流: $$x \in \mathbb{R}^{d}$$ $$\to$$ Linear $$(d \to 4d)$$ $$\to$$ GELU $$\to$$ Linear $$(4d \to d)$$ $$\to$$ 输出。

这是一种「扩展-压缩」设计: 首先将嵌入维度扩展 4 倍 (由 768 到 3072)，然后通过非线性激活，最后压缩回原始维度。此瓶颈结构使模型能够在高维空间中探索更丰富的表示，同时保持输入/输出维度一致，从而便于多层堆叠。

## 4.4 残差连接 (Shortcut Connections)

### 4.4.1 动机: 梯度消失

深层网络的标准前向传播为:

$$y = F(x)$$

其中 $$F$$ 为若干层组成的函数。在残差连接中:

$$y = x + F(x)$$

反向传播时:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(I + \frac{\partial F}{\partial x}\right)$$

关键洞察: 恒等项 $$I$$ 确保了即使 $$\frac{\partial F}{\partial x}$$ 趋于零 (梯度消失)，$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial y}$$ 仍能保持非零梯度向后传播。

### 4.4.2 梯度流分析

对于 $$L$$ 层无残差的深层网络，从输出到输入的梯度乘积可能导致梯度数量级呈指数衰减:

$$\frac{\partial \mathcal{L}}{\partial x_1} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{l=1}^{L-1} W_l$$

若各层权重矩阵 $$W_l$$ 的谱半径均小于 1，梯度将迅速消失。

加入残差连接后，对任意深度的层:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left[I + \sum_{k=l}^{L-1} \frac{\partial F_k}{\partial x_l} + (\text{高阶项})\right]$$

求和项保证了梯度可以通过恒等路径跨越任意数量的层，从而有效缓解了深层网络中的梯度消失问题。

### 4.4.3 形式化证明 (残差连接的梯度保持性质)

设 $$y = x + F(x)$$，记 $$J_F = \frac{\partial F}{\partial x}$$ 为 $$F$$ 的雅可比矩阵。则反向传播的梯度流为:

$$\nabla_x \mathcal{L} = \nabla_y \mathcal{L} \cdot (I + J_F) = \nabla_y \mathcal{L} + \nabla_y \mathcal{L} \cdot J_F$$

其中第一项 $$\nabla_y \mathcal{L}$$ 直接将损失梯度无损地传回输入，第二项通过 $$F$$ 调整。这保证了一个最小梯度下限: $$\|\nabla_x \mathcal{L}\| \geq \|\nabla_y \mathcal{L}\| - \|\nabla_y \mathcal{L} \cdot J_F\|$$。在最坏情况下 ($$J_F = 0$$)，仍有 $$\nabla_x \mathcal{L} = \nabla_y \mathcal{L}$$，梯度完全不会被衰减。

实验验证: 在五层网络中，无残差连接时第一层平均梯度为 $$0.0002$$ (近乎消失)，有残差连接时为 $$0.22$$ (三个数量级的提升)。

## 4.5 Transformer 块

### 4.5.1 完整计算推导

Transformer 块是 GPT 架构的核心构建单元，结合了掩码多头注意力 (第 3 章) 及其他的所有组件。一个 Transformer 块的完整计算如下:

**步骤 1 — 注意力分支**:

$$x_1 = \text{LayerNorm}(x)$$

$$x_2 = \text{MaskedMultiHeadAttention}(x_1)$$

$$x_3 = \text{Dropout}(x_2)$$

$$x_{\text{mid}} = x + x_3$$

步骤 1 的紧凑形式:

$$x_{\text{mid}} = x + \text{Dropout}(\text{MHA}(\text{LN}(x)))$$

**步骤 2 — 前馈分支**:

$$x_4 = \text{LayerNorm}(x_{\text{mid}})$$

$$x_5 = \text{FeedForward}(x_4)$$

$$x_6 = \text{Dropout}(x_5)$$

$$x_{\text{out}} = x_{\text{mid}} + x_6$$

步骤 2 的紧凑形式:

$$x_{\text{out}} = x_{\text{mid}} + \text{Dropout}(\text{FFN}(\text{LN}(x_{\text{mid}})))$$

**完整合并**:

$$\text{TransformerBlock}(x) = x + \text{Dropout}\Big(\text{MHA}\big(\text{LN}(x)\big)\Big) + \text{Dropout}\Big(\text{FFN}\big(\text{LN}\big(x + \text{Dropout}(\text{MHA}(\text{LN}(x)))\big)\big)\Big)$$

注意此处使用的是 **Pre-LayerNorm** 方案: LayerNorm 在注意力和前馈子层 **之前** 应用，而非在之后 (Post-LayerNorm)。实验表明 Pre-LN 提供更稳定的训练动态。

### 4.5.2 代码实现

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
```

Transformer 块的输入和输出维度完全相同 (`[batch, seq_len, emb_dim]`)，这一恒等形状保留设计是其核心特征——输出是融入了全局语境信息的「上下文向量」。

## 4.6 组装 GPT 模型

### 4.6.1 完整架构

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

### 4.6.2 参数计数推导

对于 GPT-2 小型的配置 (`V` 为词表大小，`d` 为嵌入维度，`N` 为层数，`d_ff = 4d`):

**嵌入层**:

- Token 嵌入: $$V \times d = 50257 \times 768$$
- 位置嵌入: $$\text{ctx} \times d = 1024 \times 768$$

**每个 Transformer 块**:

- LayerNorm (两个): $$2 \times 2d = 4d$$ (每个 LN 有 $$\gamma$$ 和 $$\beta$$)
- 多头注意力: $$4 \times d \times d = 4d^2$$ (Q, K, V 投影 + 输出投影; 无偏置时)
  - 推导: Q 投影 $$d \times d$$, K 投影 $$d \times d$$, V 投影 $$d \times d$$, 输出投影 $$d \times d$$, 合计 $$4d^2$$
- 前馈网络: $$d \times 4d + 4d \times d = 8d^2$$ (两个全连接层; 不含偏置)

**最终层**:

- 最终 LayerNorm: $$2d$$
- 输出头 (无偏置): $$d \times V$$

**汇总公式** (不含偏置，无权重绑定):

$$\text{total\_params} = d(V + \text{ctx}) + N(4d + 4d^2 + 8d^2) + 2d + dV$$

代入值: $$768(50257+1024) + 12(4 \cdot 768 + 12 \cdot 768^2) + 2 \cdot 768 + 768 \cdot 50257 \approx 163,009,536$$

### 4.6.3 权重绑定 (Weight Tying)

在原始 GPT-2 中，输出层权重与 token 嵌入层权重共享，即:

$$W_{\text{out}} = W_{\text{tok\_emb}}^T$$

扣除 $$V \times d = 50,257 \times 768 = 38,597,376$$ 个参数后:

$$163,009,536 - 38,597,376 = 124,412,160 \approx 124\text{M}$$

这解释了为何该模型被称为「124M」版本。权重绑定减少了内存占用，但现代 LLM 通常使用独立的嵌入层和输出层以获得更好的训练效果。

### 4.6.4 模型规格族

| 模型 | 参数 | `emb_dim` | `n_layers` | `n_heads` |
|------|------|-----------|------------|-----------|
| GPT-2 Small | 124M | 768 | 12 | 12 |
| GPT-2 Medium | 345M | 1024 | 24 | 16 |
| GPT-2 Large | 762M | 1280 | 36 | 20 |
| GPT-2 XL | 1542M | 1600 | 48 | 25 |

以上所有变体均使用同一个 `GPTModel` 类，仅需修改配置字典即可切换。

## 4.7 文本生成

### 4.7.1 自回归生成原理

GPT 是一个自回归模型，一次生成一个 token。给定前缀 $$x_{<t} = (x_1, \ldots, x_{t-1})$$，模型输出条件分布:

$$P_\theta(x_t \mid x_{<t}) = \text{softmax}(f_\theta(x_{<t})_t)$$

其中 $$f_\theta(x_{<t})_t \in \mathbb{R}^V$$ 是最后一个位置的 logit 向量。

### 4.7.2 贪婪解码

最简单的生成策略是每一步选择概率最高的 token (贪婪解码):

$$x_t = \underset{v \in \mathcal{V}}{\arg\max} \, P_\theta(v \mid x_{<t})$$

由于 softmax 函数是单调递增的，直接取 logit 的最大值等价:

$$x_t = \underset{v \in \mathcal{V}}{\arg\max} \, f_\theta(x_{<t})_t[v]$$

### 4.7.3 温度缩放

为控制生成的随机性，引入温度参数 $$T > 0$$:

$$P_\theta^{(T)}(v \mid x_{<t}) = \text{softmax}\left(\frac{f_\theta(x_{<t})_t}{T}\right)[v] = \frac{\exp(z_v / T)}{\sum_{w} \exp(z_w / T)}$$

温度效应的极限行为:
- $$T \to 0^+$$: 分布集中在最大值上，等价于贪婪解码
- $$T = 1$$: 原始概率分布
- $$T \to \infty$$: 均匀分布，完全随机生成

### 4.7.4 Top-k 采样

Top-k 采样先截取概率最高的 $$k$$ 个 token，再从中采样:

$$\mathcal{V}^{(k)} = \{v \in \mathcal{V} : \text{rank}(P_\theta(v \mid x_{<t})) \leq k\}$$

重归一化:

$$P_\theta^{(k)}(v \mid x_{<t}) = \begin{cases} \frac{P_\theta(v \mid x_{<t})}{\sum_{w \in \mathcal{V}^{(k)}} P_\theta(w \mid x_{<t})} & v \in \mathcal{V}^{(k)} \\ 0 & \text{否则} \end{cases}$$

然后从 $$P_\theta^{(k)}$$ 中采样。结合温度缩放和 Top-k:

$$P_\theta^{(T,k)}(v \mid x_{<t}) \propto \begin{cases} \exp(z_v / T) & v \in \mathcal{V}^{(k)} \\ 0 & \text{否则} \end{cases}$$

### 4.7.5 生成循环

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                           # 取最后一个位置
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

循环过程: 每一步以当前上下文为输入计算 logits $$\to$$ 取最后一位置的输出向量 $$\to$$ softmax 转换为概率 $$\to$$ argmax 选择 token ID $$\to$$ 将其追加到输入序列。重复此过程直到生成指定数量的新 token。

### 4.7.6 未训练模型的行为

未训练的模型 (权重为随机初始化) 输出的 token 分布接近均匀，生成的文本毫无意义——这恰恰说明了模型训练的必要性。第 5 章将介绍完整的预训练流程。

## 本章小结

- **Layer Normalization** 对每个样本的特征维度独立归一化至零均值、单位方差，通过可学习的 $$\gamma$$ 和 $$\beta$$ 恢复表达能力，天然适合变长序列和分布式训练
- **GELU** 是 ReLU 的光滑化变体，定义为 $$x \cdot \Phi(x)$$，实践使用 tanh 近似以加速计算
- **FeedForward** 采用扩展-压缩架构 ($$d \to 4d \to d$$)，在中间高维空间中学习丰富的特征表示
- **残差连接** 引入恒等梯度路径 ($$\nabla_x\mathcal{L} = \nabla_y\mathcal{L} \cdot (I + J_F)$$)，从根本上缓解了深层网络的梯度消失问题
- **Transformer 块** 以 Pre-LN 方式整合多头注意力、前馈网络、残差连接和 LayerNorm，是整个 GPT 架构的核心构件
- **GPT 模型** 堆叠 $$N$$ 个 Transformer 块，经过最终 LayerNorm 和线性输出头映射到词表空间
- **参数计数** 揭示了不同组件 (嵌入层、注意力、前馈、输出头) 的参数分布，权重绑定是 124M 命名的来源
- **文本生成** 是自回归过程: 每一步基于已有前缀预测下一 token，通过温度缩放和 Top-k 采样平衡确定性与多样性

---

<!-- chapter-nav -->
<div style="display:flex; justify-content:space-between; align-items:center; padding:1em 0;">
  <div><a href="ch03_注意力.md">← 第3章 注意力机制</a></div>
  <div><a href="index.md">↑ 目录</a></div>
  <div><a href="ch05_预训练.md">第5章 预训练 →</a></div>
</div>
