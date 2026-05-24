"""
microgpt_torch —— microgpt 的 PyTorch 版本
==========================================

与原始版本完全相同的架构和超参数，但使用 PyTorch：
  - torch.autograd 替代手工 Value 类
  - nn.Module / nn.Linear / nn.Embedding 替代手工参数矩阵
  - torch.optim.Adam 替代手工 Adam
  - 整序列并行前向传播（相比原始逐 token 循环更高效）
  - 推理时仍用逐 token 自回归生成
"""

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ============================================================================
# Dataset
# ============================================================================
if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt',
        'input.txt'
    )

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ============================================================================
# Tokenizer
# ============================================================================
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ============================================================================
# Hyperparameters
# ============================================================================
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

# ============================================================================
# Model
# ============================================================================

class RMSNorm(nn.Module):
    """
    RMSNorm（均方根归一化）—— 稳定训练的关键组件。

    公式：y = x / sqrt(mean(x²) + ε)

    直觉：把向量的"长度"缩放到接近 1，防止深层网络中数值爆炸或消失。
    和 LayerNorm 的区别：没有减均值（不 center），也没有可学习的缩放参数。
    """
    def forward(self, x):
        # x shape: (T, n_embd)  序列 × 嵌入维度
        ms = torch.mean(x ** 2, dim=-1, keepdim=True)
        # ms shape: (T, 1)  每个 token 的均方值
        # keepdim=True 保持维度，方便广播运算：(T,1) / (T,1) → (T,n_embd)
        return x / torch.sqrt(ms + 1e-5)


class CausalSelfAttention(nn.Module):
    """
    因果自注意力（Causal Self-Attention）—— token 之间的"对话"。

    三个角色：
      Q（Query，查询）："我在找什么样的信息？"
      K（Key，键）：    "我拥有什么样的信息？"
      V（Value，值）：   "如果你选中我，我能给你什么？"

    过程：
      1. Q 和每个 K 做点积 → 相似度分数
      2. 因果掩码：挡住未来 token（只许看过去，不许看未来）
      3. Softmax → 注意力权重（概率分布）
      4. 用权重对 V 加权求和 → 输出

    多头（Multi-Head）：4 个头各自做独立的注意力，关注不同类型的模式。
    """
    def __init__(self):
        super().__init__()
        # 四组线性变换，每个都是 nn.Linear，内部自动维护权重矩阵
        # 输入：n_embd=16，输出：n_embd=16，无偏置
        # 内部矩阵 shape：weight (16, 16)  →  y = x @ W^T
        self.wq = nn.Linear(n_embd, n_embd, bias=False)   # Query
        self.wk = nn.Linear(n_embd, n_embd, bias=False)   # Key
        self.wv = nn.Linear(n_embd, n_embd, bias=False)   # Value
        self.wo = nn.Linear(n_embd, n_embd, bias=False)   # 输出投影

    def forward(self, x):
        """
        x shape: (T, n_embd)  序列长度16，每个token是16维向量
        返回:     (T, n_embd)  经过注意力交换信息后的新表示
        """
        T = x.size(0)  # T ≤ block_size = 16

        # ---- 第1步：生成 Q, K, V ----
        # x: (T, 16) → Linear(16, 16) → q/k/v: (T, 16)
        # 每个 Linear 等价于 y = x @ W^T，W shape (16, 16)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # ---- 第2步：分头（split heads） ----
        # 目标：从 (T, 16) 变成 (n_head, T, head_dim) = (4, T, 4)
        # 这样每个头独立看到 T×4 的子空间
        #
        # q.view(T, n_head, head_dim):  (T, 16) → (T, 4, 4)
        #   把16维拆成4组×4维，相当于把向量切成4段
        # .transpose(0, 1):  (T, 4, 4) → (4, T, 4)
        #   交换 batch 和 head 维度，让头维度在最前面方便并行
        q = q.view(T, n_head, head_dim).transpose(0, 1)   # (4, T, 4)
        k = k.view(T, n_head, head_dim).transpose(0, 1)   # (4, T, 4)
        v = v.view(T, n_head, head_dim).transpose(0, 1)   # (4, T, 4)

        # ---- 第3步：计算注意力分数 ----
        # q @ k.transpose(-2, -1)  →  每个头内：Q 与所有 K 的点积
        #   q shape: (4, T, 4), k shape: (4, T, 4)
        #   k.transpose(-2, -1): (4, T, 4) → (4, 4, T)
        #   矩阵乘法: (4, T, 4) @ (4, 4, T) → (4, T, T)
        #   结果 att[h, i, j] = 头h中 token_i 的 Q 与 token_j 的 K 的点积
        #   解释：att[h, i, j] = token_i 对 token_j 的"关注度分数"
        #
        # * (head_dim ** -0.5):  缩放因子 = 1/√4 = 0.5
        # 为什么要缩放？点积的大小随维度增长。不缩放的话，softmax 后分布会
        # 过于尖锐（接近 one-hot），梯度接近 0，训练不动。
        # 除以 √d 让方差回到 1，softmax 分布适度平滑。
        att = q @ k.transpose(-2, -1) * (head_dim ** -0.5)
        # att shape: (4, T, T)

        # ---- 第4步：因果掩码（Causal Mask） ----
        # 训练时每个 token 只能看它自己和之前的 token，不能看未来的 token。
        # 为什么？因为推理时就是从左到右生成的，训练必须和推理一致。
        #
        # torch.triu(..., diagonal=1):  生成上三角矩阵（不包括对角线）
        #   T=4 时：[[0, -inf, -inf, -inf],
        #            [0,    0, -inf, -inf],
        #            [0,    0,    0, -inf],
        #            [0,    0,    0,    0]]
        # 加上 -inf 后，softmax 会把对应位置变成 0（e^{-inf} ≈ 0）
        # 这样 token_i 就看不到 token_j（j>i）的信息了
        mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1
        )
        att = att + mask  # 广播加到所有头上：(4, T, T) + (T, T)

        # ---- 第5步：Softmax 归一化 ----
        # 对每个 token 的所有历史关注度做 softmax，变成概率分布
        # dim=-1 表示对最后一维（T 个 key）做 softmax
        # 结果：att[h, i, :] 是 token_i 对所有历史 token 的注意力权重，和为1
        att = F.softmax(att, dim=-1)  # (4, T, T)

        # ---- 第6步：加权求和 ----
        # att @ v: (4, T, T) @ (4, T, 4) → (4, T, 4)
        # 每个 token 的最终表示 = 所有历史 token 的 V 的加权平均
        # 权重就是上面算出的注意力分数
        y = att @ v  # (4, T, 4)

        # ---- 第7步：合并头 ----
        # y.transpose(0, 1): (4, T, 4) → (T, 4, 4)
        # .contiguous(): 确保内存连续（transpose 可能不连续）
        # .view(T, n_embd): (T, 4, 4) → (T, 16)
        #   把4个头×4维拼回16维向量
        y = y.transpose(0, 1).contiguous().view(T, n_embd)  # (T, 16)

        # ---- 第8步：输出投影 ----
        # 混合4个头的信息。每个头可能关注不同的模式，需要整合。
        # Linear(16, 16):  (T, 16) → (T, 16)
        return self.wo(y)


class MLP(nn.Module):
    """
    MLP（多层感知机）—— 每个 token 独立的"思考"步骤。

    注意力负责 token 之间交换信息，MLP 则让每个 token 在自己位置上
    消化吸收刚得到的信息。

    结构：16维 → 64维（扩展4倍，获得更大表达空间）
          → ReLU（非线性激活，让网络能学复杂模式）
          → 16维（压缩回来）
    """
    def __init__(self):
        super().__init__()
        # fc1: wide_n = 4*n_embd = 64, 输出维度是输入维度的4倍
        # 矩阵 shape: (64, 16)  →  把16维投影到64维空间
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        # fc2: 从64维压缩回16维
        # 矩阵 shape: (16, 64)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        # x: (T, 16)
        # fc1: (T, 16) @ (64, 16)^T = (T, 16) @ (16, 64) = (T, 64)
        #     PyTorch 内部实现为 x @ W^T，W shape (64, 16)
        x = self.fc1(x)       # (T, 16) → (T, 64)
        x = F.relu(x)         # 非线性：负值截断为0，正值不变
        # fc2: (T, 64) @ (16, 64)^T = (T, 64) @ (64, 16) = (T, 16)
        x = self.fc2(x)       # (T, 64) → (T, 16)
        return x


class Block(nn.Module):
    """
    Transformer 块：一层完整的"注意力 + MLP"。

    数据流：
      x → [RMSNorm → 注意力] → 残差连接(+) → [RMSNorm → MLP] → 残差连接(+) → 输出

      ★ 残差连接（Residual Connection）★
        output = layer(norm(input)) + input
        两个好处：
          1. 梯度直通：反向传播时梯度可以跳过注意力/MLP层，防止梯度消失
          2. 学的是"增量"：每层只需要学"在原始信息上做什么修改"
    """
    def __init__(self):
        super().__init__()
        self.attn_norm = RMSNorm()      # 注意力前的归一化
        self.attn = CausalSelfAttention()  # 多头因果自注意力
        self.mlp_norm = RMSNorm()       # MLP 前的归一化
        self.mlp = MLP()                # 前馈网络

    def forward(self, x):
        # 注意力子层 + 残差连接
        x = x + self.attn(self.attn_norm(x))
        # MLP 子层 + 残差连接
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    """
    GPT 模型：Token 嵌入 → 位置编码 → [Transformer块]×n_layer → 输出投影

    架构遵循 GPT-2（简化版）：
      - 用 RMSNorm 代替 LayerNorm
      - 用 ReLU 代替 GeLU
      - 去掉所有偏置
    """
    def __init__(self):
        super().__init__()
        # wte：Token 嵌入表 (vocab_size, n_embd) = (27, 16)
        # 每个 token（a-z + BOS）映射到一个 16 维向量
        # 相当于一个"身份证"，把离散的编号变成有意义的连续向量
        self.wte = nn.Embedding(vocab_size, n_embd)

        # wpe：位置嵌入表 (block_size, n_embd) = (16, 16)
        # 每个位置（0-15）映射到一个 16 维向量
        # 告诉模型"这个字母在名字中的第几个位置"
        self.wpe = nn.Embedding(block_size, n_embd)

        self.embed_norm = RMSNorm()           # 嵌入后的归一化
        self.layers = nn.ModuleList(           # Transformer 层栈
            [Block() for _ in range(n_layer)]  # 1 层
        )
        # lm_head：输出投影 (n_embd, vocab_size) = (16, 27)
        # 把模型内部的 16 维向量映射回 27 个分数（logits）
        # 分数越高 → 模型越认为这个 token 应该出现在下一个位置
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        """
        idx: (T,)  一维整数张量，token ID 序列
        返回: (T, vocab_size)  每个位置的预测 logits
        """
        T = idx.size(0)  # 序列实际长度（可能 ≤ block_size）

        # ---- 第1步：Token 嵌入 ----
        # idx: (T,) → 查表 wte → tok_emb: (T, 16)
        # 每个 token ID 被替换为对应的 16 维向量
        tok_emb = self.wte(idx)

        # ---- 第2步：位置嵌入 ----
        # torch.arange(T): 生成 [0, 1, 2, ..., T-1]
        # 查表 wpe → pos_emb: (T, 16)
        # 每个位置被替换为对应的 16 维向量
        pos_emb = self.wpe(torch.arange(T, device=idx.device))

        # ---- 第3步：合并嵌入并归一化 ----
        # tok_emb + pos_emb: (T, 16) + (T, 16) = (T, 16)
        # 直觉：一个 token 的最终表示 = "它是什么字母" + "它在第几个位置"
        x = self.embed_norm(tok_emb + pos_emb)

        # ---- 第4步：经过所有 Transformer 层 ----
        for layer in self.layers:
            x = layer(x)  # x shape 始终为 (T, 16)

        # ---- 第5步：输出投影 ----
        # lm_head: (T, 16) @ (27, 16)^T = (T, 16) @ (16, 27) = (T, 27)
        # 每个位置输出 27 个分数，对应词汇表中 27 个 token
        return self.lm_head(x)


model = GPT()
# 计算总参数量
# wte: 27*16=432, wpe: 16*16=256, embed_norm: 0（无参数）
# layers[0].attn: 4个Linear × 16×16 = 1024
# layers[0].mlp: 64×16 + 16×64 = 2048
# lm_head: 27×16 = 432
# 合计: 432 + 256 + 1024 + 2048 + 432 = 4192
print(f"num params: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# Training
# ============================================================================
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,          # 学习率
    betas=(0.85, 0.99), # Adam 的动量参数（beta1, beta2）
    eps=1e-8           # 防止除0
)
num_steps = 1000

for step in range(num_steps):
    # ---- 第1步：准备训练样本 ----
    # 选一个名字，两边加上 BOS（Beginning of Sequence）标记
    # 例如 "emma" → [26, 4, 12, 12, 0, 26]
    #                BOS  e   m   m   a  BOS
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    tokens = tokens[:block_size + 1]  # 截断到最长 block_size+1

    # x = 输入序列（去掉最后一个 token）
    # y = 目标序列（去掉第一个 token），即每个位置的下一个 token
    # 例如 tokens=[26,4,12,12,0,26] →
    #   x=[26,4,12,12,0],  y=[4,12,12,0,26]
    # 模型看到 x[i] 要预测 y[i]
    x = torch.tensor(tokens[:-1])  # (n,)  输入
    y = torch.tensor(tokens[1:])   # (n,)  正确输出

    # ---- 第2步：前向传播 ----
    # model(x): (n,) → (n, vocab_size) = (n, 27)
    # 输出每个位置对所有 27 个 token 的预测分数
    logits = model(x)

    # ---- 第3步：计算损失 ----
    # F.cross_entropy(logits, y):
    #   对 logits 做 softmax → 取目标 token 的概率 → -log(概率) → 平均
    #   等价于原始代码中逐个 token 算 -log(p) 再平均
    #   如果模型很有把握（p≈1）：loss≈0；如果模型猜错（p≈0）：loss 很大
    loss = F.cross_entropy(logits, y)

    # ---- 第4步：反向传播 ----
    optimizer.zero_grad()   # 清空旧梯度（否则会累加）
    loss.backward()         # 自动算所有参数的梯度（PyTorch 自动微分）

    # ---- 第5步：学习率线性衰减 ----
    # 训练后期减小步长，让模型精细调整而不是大幅跳动
    lr_t = 0.01 * (1 - step / num_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_t

    # ---- 第6步：Adam 更新参数 ----
    optimizer.step()

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}", end='\r')

# ============================================================================
# Generation
# ============================================================================
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")

@torch.no_grad()
def generate(temperature=0.5):
    """
    自回归生成：逐 token 预测，每次把新 token 拼回输入序列。

    过程：
      1. 从 BOS 开始
      2. 模型预测下一个 token 的概率分布
      3. 按概率采样（不是贪心选最高概率的）
      4. 把结果拼回序列，重复
      5. 遇到 BOS 或达到最大长度时停止

    温度（temperature）控制"创造力"：
      t→0：贪心解码，每次都选概率最高的（确定性，较"安全"）
      t=1：原始分布
      t>1：分布更均匀，更随机（可能产生奇怪但有趣的结果）
    """
    model.eval()
    tokens = [BOS]  # 从开始标记出发
    for _ in range(block_size):
        # 取最近 block_size 个 token（模型只看这么多上下文）
        idx = torch.tensor(tokens[-block_size:])  # (context_len,)

        # 前向传播，取最后一个位置的 logits（预测下一个 token）
        # model(idx): (context_len,) → (context_len, vocab_size)
        # [-1]: 取最后一个位置 → (vocab_size,)
        # / temperature: 温度缩放，控制分布尖锐程度
        logits = model(idx)[-1] / temperature

        # Softmax 转成概率分布
        probs = F.softmax(logits, dim=-1)  # (vocab_size,)

        # 按概率分布采样（不是简单取 argmax！）
        # 这样模型有"创造力"，不会每次都输出一样的名字
        next_token = torch.multinomial(probs, 1).item()

        # 如果模型预测了 BOS，说明它认为名字应该结束了
        if next_token == BOS:
            break

        tokens.append(next_token)

    # 去掉开头的 BOS，把数字转换成字符
    return ''.join(uchars[t] for t in tokens[1:])

for i in range(20):
    print(f"sample {i+1:2d}: {generate(temperature)}")
