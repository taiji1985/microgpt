以下内容翻译自Andrej Karpathy 的博客

# microgpt

**Andrej Karpathy** | 2026年2月12日

> 原文链接：[http://karpathy.github.io/2026/02/12/microgpt/](http://karpathy.github.io/2026/02/12/microgpt/)

---

这是我新的艺术项目 [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) 的简要指南——一个仅有200行纯Python代码、零依赖的文件，可以训练和推理一个GPT模型。这个文件包含了所需的完整算法内容：文档数据集、分词器、自动微分引擎、类GPT-2的神经网络架构、Adam优化器、训练循环和推理循环。除此之外的一切都只是为了效率。我已经无法再进一步简化了。这个脚本是多个项目（micrograd、makemore、nanogpt等）的最终结晶，也是我十年来将LLM简化到最本质的执念，我觉得它很美 🥹。它甚至完美地分成了3列：

![microgpt代码全景图](https://karpathy.github.io/assets/microgpt.jpg)

**在哪里可以找到它：**

- GitHub Gist上有完整源代码：[microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- 也可以在这个网页上查看：[https://karpathy.ai/microgpt.html](https://karpathy.ai/microgpt.html)
- 还可以作为 [Google Colab 笔记本](https://colab.research.google.com/) 使用

以下是我为有兴趣的读者逐步讲解代码的指南。

---

## 数据集（Dataset）

大语言模型的燃料是文本数据流，可以选择性地分成一组文档。在生产级应用中，每个文档会是一个互联网网页，但对于 microgpt，我们使用一个更简单的例子——32,000个名字，每行一个：

```python
# 让这里有一个输入数据集 `docs`：list[str] 的文档列表（例如一个名字数据集）
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]  # list[str] 文档列表
random.shuffle(docs)
print(f"num docs: {len(docs)}")
```

数据集看起来是这样的。每个名字就是一个文档：

```
emma
olivia
ava
isabella
sophia
charlotte
mia
amelia
harper
...（约32,000个名字）
```

模型的目标是学习数据中的模式，然后生成共享相同统计模式的新文档。作为预览，在脚本运行结束时，我们的模型将会生成（"幻觉"出！）新的、听起来合理的名字。提前剧透一下，我们会得到：

```
sample  1: kamon
sample  2: ann
sample  3: karai
sample  4: jaire
sample  5: vialan
sample  6: karia
sample  7: yeran
sample  8: anna
sample  9: areli
sample 10: kaina
sample 11: konna
sample 12: keylen
sample 13: liole
sample 14: alerin
sample 15: earan
sample 16: lenne
sample 17: kana
sample 18: lara
sample 19: alela
sample 20: anton
```

看起来不算什么，但从像ChatGPT这样的模型的角度来看，你和它的对话只不过是一种形式特殊的"文档"。当你用提示词（prompt）初始化文档时，模型的回复从它的视角来看只是一种统计上的文档补全。

---

## 分词器（Tokenizer）

在底层，神经网络处理的是数字而非字符，因此我们需要一种方法将文本转换为整数token id的序列，然后再转回来。生产级分词器如 tiktoken（GPT-4使用的）为了效率会操作字符块，但最简单的分词器只是为数据集中每个唯一字符分配一个整数：

```python
# 让这里有一个分词器，将字符串翻译为离散符号，再翻译回来
uchars = sorted(set(''.join(docs)))  # 数据集中的唯一字符成为 token id 0..n-1
BOS = len(uchars)  # 特殊的序列开始（BOS）token 的 id
vocab_size = len(uchars) + 1  # 唯一 token 的总数，+1 是给 BOS 的
print(f"vocab size: {vocab_size}")
```

在上面的代码中，我们收集数据集中所有唯一字符（即所有小写字母 a-z），排序后每个字母通过其索引获得一个 id。注意，整数值本身没有任何意义；每个 token 只是一个独立的离散符号。它们不是0、1、2，用不同的emoji来代替也一样。

此外，我们创建了一个额外的特殊token叫BOS（Beginning of Sequence，序列开始），它充当分隔符：告诉模型"一个新文档从这里开始/结束"。后面在训练时，每个文档的两侧都会用BOS包裹：`[BOS, e, m, m, a, BOS]`。模型学会BOS意味着开始一个新名字，另一个BOS意味着结束它。

因此，我们最终的词汇表大小为27（26个可能的小写字母 a-z，加上1个BOS token）。

---

## 自动微分（Autograd）

训练神经网络需要梯度：对于模型中的每个参数，我们需要知道"如果我把这个数字稍微增大一点，损失是上升还是下降，变化了多少？"。计算图有很多输入（模型参数和输入token），但最终汇聚到一个标量输出：损失（我们将在下面准确定义损失是什么）。反向传播从那个单一输出开始，沿着计算图反向工作，计算损失相对于每个输入的梯度。它依赖于微积分中的链式法则。

在生产中，PyTorch等库会自动处理这些。这里，我们在一个叫做 `Value` 的类中从头实现它：

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data           # 前向传播中计算的标量值
        self.grad = 0              # 损失对该节点的导数，在反向传播中计算
        self._children = children  # 计算图中该节点的子节点
        self._local_grads = local_grads  # 该节点对其子节点的局部导数

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad
```

我知道这是数学和算法上最密集的部分，我有一个 [2.5小时的视频](https://www.youtube.com/watch?v=VMj-3S1tku0) 专门讲解它：micrograd视频。

简单来说，一个 `Value` 包装了一个标量数字（`.data`），并追踪它是如何被计算出来的。把每个操作想象成一块小乐高积木：它接收一些输入，产生一个输出（前向传播），并且知道它的输出相对于每个输入会如何变化（局部梯度）。这就是自动微分从每个积木块中所需的全部信息。其余的一切只是链式法则，把这些积木串在一起。

每当你用 `Value` 对象做数学运算（加法、乘法等），结果是一个新的 `Value`，它记住了它的输入（`_children`）以及该操作的局部导数（`_local_grads`）。例如，`__mul__` 记录了 ∂(a·b)/∂a = b 和 ∂(a·b)/∂b = a。

完整的乐高积木集合：

| 运算 | 前向 | 局部梯度 |
|------|------|----------|
| a + b | a + b | ∂/∂a = 1, ∂/∂b = 1 |
| a * b | a · b | ∂/∂a = b, ∂/∂b = a |
| a ** n | aⁿ | ∂/∂a = n · aⁿ⁻¹ |
| log(a) | ln(a) | ∂/∂a = 1/a |
| exp(a) | eᵃ | ∂/∂a = eᵃ |
| relu(a) | max(0, a) | ∂/∂a = **1**_{a>0} |

`backward()` 方法按照反向拓扑排序遍历计算图（从损失开始，到参数结束），在每一步应用链式法则。如果损失是 L，一个节点 v 有一个子节点 c，局部梯度为 ∂v/∂c，那么：

**∂L/∂c += ∂v/∂c · ∂L/∂v**

如果你对微积分不太熟悉，这看起来可能有点吓人，但这实际上就是以直觉的方式将两个数字相乘。一种理解方式是："如果一辆汽车的速度是自行车的两倍，而自行车的速度是步行者的四倍，那么汽车的速度就是步行者的 2 × 4 = 8 倍。"链式法则就是同样的道理：沿着路径乘以变化率。

我们通过在损失节点处设置 `self.grad = 1` 来启动，因为 ∂L/∂L = 1：损失相对于自身的变化率显然是1。从那里开始，链式法则只需沿每条路径将局部梯度相乘回传到参数。

注意 `+=`（累加，而非赋值）。当一个值在计算图中多处被使用（即图产生分支）时，梯度沿每个分支独立回传，必须求和。这是多元链式法则的结果：如果 c 通过多条路径对 L 有贡献，总导数是每条路径贡献的总和。

`backward()` 完成后，图中的每个 `Value` 都有一个 `.grad`，包含 ∂L/∂v，告诉我们如果微调该值，最终损失会如何变化。

这里有一个具体的例子。注意 a 被使用了两次（图产生分支），所以它的梯度是两条路径的总和：

```python
a = Value(2.0)
b = Value(3.0)
c = a * b     # c = 6.0
L = c + a     # L = 8.0
L.backward()
print(a.grad)  # 4.0 (dL/da = b + 1 = 3 + 1，通过两条路径)
print(b.grad)  # 2.0 (dL/db = a = 2)
```

这和 PyTorch 的 `.backward()` 给出的结果完全一致：

```python
import torch
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = a * b
L = c + a
L.backward()
print(a.grad)  # tensor(4.)
print(b.grad)  # tensor(2.)
```

这和 PyTorch 的 `loss.backward()` 运行的是同一个算法，只不过是在标量而非张量（标量数组）上——算法完全相同，规模显著更小更简单，但当然效率低得多。

让我们详细说明上面 `.backward()` 给出的结果。自动微分计算出，如果 L = a*b + a，且 a=2, b=3，那么 a.grad = 4.0 告诉我们 a 对 L 的局部影响。如果你微调输入 a，L 会往哪个方向变化？这里，L 对 a 的导数是4.0，意味着如果我们将 a 增加一个微小量（比如0.001），L 将增加大约4倍（0.004）。类似地，b.grad = 2.0 意味着对 b 的同样微调会使 L 增加大约2倍（0.002）。

换句话说，这些梯度告诉我们每个单独输入对最终输出（损失）的影响方向（正或负取决于符号）和陡度（幅度）。这然后允许我们迭代地微调神经网络的参数以降低损失，从而改善其预测。

---

## 参数（Parameters）

参数是模型的知识。它们是一大堆浮点数（用 `Value` 包装以支持自动微分），初始为随机值，在训练过程中被迭代优化。每个参数的确切角色在我们定义模型架构后会更有意义，但现在我们只需要初始化它们：

```python
n_embd = 16       # 嵌入维度
n_head = 4        # 注意力头数
n_layer = 1       # 层数
block_size = 16   # 最大序列长度
head_dim = n_embd // n_head  # 每个头的维度

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")
```

每个参数被初始化为从高斯分布中采样的小随机数。`state_dict` 将它们组织成命名矩阵（借用PyTorch的术语）：嵌入表、注意力权重、MLP权重和最终输出投影。我们还将所有参数展平成一个列表 `params`，以便优化器稍后遍历它们。在我们的微型模型中，这共有4,192个参数。GPT-2有16亿个，现代LLM有数千亿个。

---

## 架构（Architecture）

模型架构是一个无状态函数：它接收一个 token、一个位置、参数以及之前位置缓存的 key/value，返回 logits（分数），表示模型认为序列中下一个最可能出现的 token。我们遵循GPT-2并做了小幅简化：RMSNorm替代LayerNorm，没有偏置，ReLU替代GeLU。

首先，三个小的辅助函数：

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

`linear` 是矩阵-向量乘法。它接收一个向量 x 和一个权重矩阵 w，对 w 的每一行计算一个点积。这是神经网络的基本构建块：一个学习到的线性变换。

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

`softmax` 将一个原始分数向量（logits）——范围可以从 -∞ 到 +∞——转换为概率分布：所有值都在 [0, 1] 之间且和为1。我们先减去最大值以保证数值稳定（这在数学上不改变结果，但防止 exp 溢出）。

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

`rmsnorm`（均方根归一化）重新缩放一个向量，使其值具有单位均方根。这使得激活值在网络中流动时不会增长或缩小，从而稳定训练。它是原始GPT-2中使用的 LayerNorm 的简化版本。

现在是模型本身：

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]     # token 嵌入
    pos_emb = state_dict['wpe'][pos_id]       # 位置嵌入
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # token 和位置的联合嵌入
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) 多头注意力块
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim))
                          / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                       for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # 2) MLP块
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits
```

这个函数处理一个 token（id 为 `token_id`），在时间上的特定位置（`pos_id`），以及由之前迭代中 key 和 value 的激活值总结的上下文，即 KV Cache。以下是逐步发生的事情：

**嵌入（Embeddings）**。神经网络不能直接处理像5这样的原始 token id。它只能处理向量（数字列表）。所以我们为每个可能的 token 关联一个学习到的向量，并将其作为 token 的神经签名输入。token id 和 position id 各自在相应的嵌入表（wte 和 wpe）中查找一行。这两个向量相加，给模型一个同时编码了 token 是什么以及它在序列中位置的表示。现代LLM通常跳过位置嵌入，引入其他基于相对位置的方案，例如 RoPE。

**注意力块（Attention block）**。当前 token 被投影为三个向量：查询（Q）、键（K）和值（V）。直觉上，查询说"我在找什么？"，键说"我包含什么？"，值说"如果被选中，我提供什么？"。例如，在名字"emma"中，当模型在第二个"m"处试图预测下一个字符时，它可能学到一个类似"最近出现了什么元音？"的查询。较早的"e"会有一个与此查询匹配良好的键，因此它获得高注意力权重，它的值（关于是元音的信息）就流入当前位置。

键和值被追加到 KV cache 中，以便之前的位置可用。每个注意力头计算其查询和所有缓存键之间的点积（除以 √d_head 进行缩放），应用 softmax 得到注意力权重，然后对缓存值取加权和。所有头的输出被拼接后通过 attn_wo 投影。

值得强调的是，注意力块是位置 t 的 token "查看"过去 0..t-1 位置 token 的**唯一且精确的位置**。注意力是一种 token 通信机制。

**MLP块**。MLP是多层感知机（multilayer perceptron）的缩写，是一个两层前馈网络：先投影到4倍嵌入维度，应用 ReLU，再投影回来。这是模型在每个位置进行大部分"思考"的地方。与注意力不同，这个计算完全局限于时间 t。Transformer 交替使用通信（注意力）和计算（MLP）。

**残差连接（Residual connections）**。注意力和MLP块都将其输出加回其输入（`x = [a + b for ...]`）。这让梯度可以直接流过网络，使更深的模型可以训练。

**输出**。最终的隐藏状态通过 lm_head 投影到词汇表大小，产生词汇表中每个 token 的一个 logit。在我们的例子中，这只是27个数字。更高的 logit = 模型认为对应的 token 更可能是下一个。

你可能注意到我们在训练过程中也使用了 KV cache，这并不常见。人们通常将 KV cache 与推理联系在一起。但 KV cache 在概念上一直存在，即使在训练中也是如此。在生产实现中，它只是隐藏在高度向量化的注意力计算中，该计算同时处理序列中的所有位置。由于 microgpt 一次处理一个 token（没有批次维度，没有并行时间步），我们显式构建 KV cache。与典型推理设置中 KV cache 持有分离张量不同，这里缓存的 key 和 value 是计算图中活跃的 Value 节点，所以我们实际上通过它们进行反向传播。

---

## 训练循环（Training Loop）

现在我们把所有东西串联起来。训练循环重复执行：(1) 选择一个文档，(2) 将模型在其 token 上前向运行，(3) 计算损失，(4) 反向传播得到梯度，(5) 更新参数。

```python
# 让这里有 Adam，神圣的优化器及其缓冲区
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)  # 一阶矩缓冲区
v = [0.0] * len(params)  # 二阶矩缓冲区

# 按顺序重复
num_steps = 1000  # 训练步数

for step in range(num_steps):
    # 取单个文档，分词，两侧用BOS特殊token包裹
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # 将 token 序列通过模型前向传播，一路构建计算图直到损失
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)  # 文档序列上的最终平均损失。愿你的损失很低。

    # 反向传播损失，计算所有模型参数的梯度
    loss.backward()

    # Adam 优化器更新：基于对应梯度更新模型参数
    lr_t = learning_rate * (1 - step / num_steps)  # 线性学习率衰减
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")
```

让我们逐一讲解每个部分：

**分词**。每个训练步选取一个文档，两侧用BOS包裹：名字"emma"变成 `[BOS, e, m, m, a, BOS]`。模型的任务是根据前面的 token 预测下一个 token。

**前向传播和损失**。我们将 token 一个接一个地送入模型，同时构建 KV cache。在每个位置，模型输出27个 logits，通过 softmax 转换为概率。每个位置的损失是正确下一个 token 的负对数概率：-log p(target)。这叫做**交叉熵损失**。直觉上，损失衡量了误预测的程度：模型对实际出现的下一个 token 有多惊讶。如果模型将概率1.0赋给正确的 token，它完全不惊讶，损失为0。如果它赋予接近0的概率，模型非常惊讶，损失趋向 +∞。我们对文档中各位置的损失取平均得到一个标量损失。

**反向传播**。一次 `loss.backward()` 调用就能通过整个计算图运行反向传播，从损失一直回到 softmax、模型和每个参数。之后，每个参数的 `.grad` 告诉我们如何改变它来降低损失。

**Adam 优化器**。我们可以直接做 `p.data -= lr * p.grad`（梯度下降），但 Adam 更智能。它为每个参数维护两个运行平均值：m 跟踪最近梯度的均值（动量，像滚动的球），v 跟踪最近平方梯度的均值（每个参数自适应学习率）。m_hat 和 v_hat 是偏差修正，考虑到 m 和 v 初始化为零需要预热。学习率在训练过程中线性衰减。更新后，我们将 `.grad` 重置为0以备下一步。

经过1,000步训练，损失从约3.3（在27个token中随机猜测：-log(1/27) ≈ 3.3）下降到约2.37。越低越好，最低可能是0（完美预测），所以仍有改进空间，但模型显然在学习名字的统计模式。

---

## 推理（Inference）

训练完成后，我们可以从模型中采样新名字。参数被冻结，我们只需在循环中运行前向传播，将每个生成的 token 反馈作为下一个输入：

```python
temperature = 0.5  # 在 (0, 1] 之间，控制生成文本的"创造力"，从低到高
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

每个样本从BOS token开始，告诉模型"开始一个新名字"。模型产生27个 logits，我们转换为概率，然后按这些概率随机采样一个 token。该 token 被反馈作为下一个输入，重复直到模型产生BOS（意味着"我完成了"）或达到最大序列长度。

**温度（temperature）** 参数控制随机性。在 softmax 之前，我们将 logits 除以温度。温度为1.0时直接从模型学到的分布中采样。较低的温度（如这里的0.5）使分布更尖锐，让模型更保守，更可能选择其首选项。接近0的温度会总是选择最可能的 token（贪心解码）。较高的温度使分布更平坦，产生更多样但可能不太连贯的输出。

---

## 运行它

你只需要Python（不需要pip install，没有依赖）：

```bash
python train.py
```

脚本在我的MacBook上大约运行1分钟。你会看到每一步打印的损失：

```
train.py
num docs: 32033
vocab size: 27
num params: 4192
step    1 / 1000 | loss 3.3660
step    2 / 1000 | loss 3.4243
step    3 / 1000 | loss 3.1778
step    4 / 1000 | loss 3.0664
step    5 / 1000 | loss 3.2209
step    6 / 1000 | loss 2.9452
step    7 / 1000 | loss 3.2894
step    8 / 1000 | loss 3.3245
step    9 / 1000 | loss 2.8990
step   10 / 1000 | loss 3.2229
step   11 / 1000 | loss 2.7964
step   12 / 1000 | loss 2.9345
step   13 / 1000 | loss 3.0544
...
```

观察它从约3.3（随机）下降到约2.37。这个数字越低，说明网络对序列中下一个 token 的预测已经越准确。训练结束时，训练 token 序列的统计模式知识被蒸馏到模型参数中。固定这些参数，我们现在可以生成新的、幻觉出的名字。

作为替代方案，你可以直接在 [Google Colab 笔记本](https://colab.research.google.com/) 上运行它，并向 Gemini 提问。试着玩一下这个脚本！你可以尝试不同的数据集。或者你可以训练更长时间（增加 num_steps）或增大模型来获得越来越好的结果。

---

## 进阶路径（Progression）

要查看代码逐步构建的过程（像洋葱一样一层层剥开），建议的进阶路径如下：

| 文件 | 新增内容 |
|------|----------|
| train0.py | 二元组（Bigram）计数表——无神经网络，无梯度 |
| train1.py | MLP + 手动梯度（数值和解析）+ SGD |
| train2.py | 自动微分（Value类）——替代手动梯度 |
| train3.py | 位置嵌入 + 单头注意力 + rmsnorm + 残差连接 |
| train4.py | 多头注意力 + 层循环——完整GPT架构 |
| train5.py | Adam优化器——这就是 train.py |

我创建了一个叫 [build_microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) 的 Gist，在其修订历史中你可以看到所有这些版本以及每一步之间的差异。我认为这可能是逐步了解代码库的一种有用方式，你一次添加一个组件。

---

## 真实世界（Real Stuff）

microgpt 包含了训练和运行GPT的完整算法精髓。但从这到像ChatGPT这样的生产级LLM，有一长串需要改变的东西。这些都不会改变核心算法和整体布局，但它们是使其在规模上真正工作的关键。按同样的章节顺序：

**数据**。与32K短名字不同，生产模型训练于数万亿 token 的互联网文本：网页、书籍、代码等。数据经过去重、质量过滤，并在不同领域之间仔细混合。

**分词器**。与单个字符不同，生产模型使用子词分词器如BPE（字节对编码），它学习将频繁共同出现的字符序列合并为单个 token。常见单词如"the"变成一个 token，稀有单词被拆分成片段。这给出约100K token 的词汇表，效率更高，因为模型每个位置看到更多内容。

**自动微分**。microgpt 在纯Python中操作标量 Value 对象。生产系统使用张量（大型多维数字数组），在GPU/TPU上运行，每秒执行数十亿次浮点运算。PyTorch等库处理张量上的自动微分，FlashAttention等CUDA内核融合多个操作以提速。数学是相同的，只是对应于并行处理的许多标量。

**架构**。microgpt有4,192个参数。GPT-4级别的模型有数千亿个。总体来说，它是一个非常相似的Transformer神经网络，只是更宽（嵌入维度10,000+）和更深（100+层）。现代LLM还引入了更多类型的乐高块并改变它们的顺序：例如 RoPE（旋转位置嵌入）替代学习的位置嵌入，GQA（分组查询注意力）减少 KV cache 大小，门控线性激活替代 ReLU，专家混合（MoE）层等。但注意力（通信）和 MLP（计算）在残差流上交替的核心结构保持良好。

**训练**。与每步一个文档不同，生产训练使用大批次（每步数百万 token）、梯度累积、混合精度（float16/bfloat16）和仔细的超参数调优。训练一个前沿模型需要数千个GPU运行数月。

**优化**。microgpt使用简单的线性学习率衰减的Adam，仅此而已。在规模上，优化本身成为一门学科。模型以降低精度（bfloat16甚至fp8）在大型GPU集群上训练以提高效率，这引入了自己的数值挑战。优化器设置（学习率、权重衰减、beta参数、预热计划、衰减计划）必须精确调优，正确值取决于模型大小、批次大小和数据集组成。缩放法则（如Chinchilla）指导如何在模型大小和训练 token 数之间分配固定的计算预算。在规模上任何这些细节出错都可能浪费数百万美元的计算，因此团队在投入完整训练之前会运行大量较小规模的实验来预测正确设置。

**后训练**。从训练中产生的基础模型（称为"预训练"模型）是一个文档补全器，不是聊天机器人。将其变成ChatGPT分两个阶段。第一，SFT（监督微调）：你只需将文档替换为精心策划的对话并继续训练。算法上没有任何变化。第二，RL（强化学习）：模型生成回复，回复被评分（由人类、另一个"裁判"模型或算法），模型从该反馈中学习。从根本上说，模型仍然在文档上训练，但这些文档现在由模型自身产生的 token 组成。

**推理**。为数百万用户提供模型服务需要自己的工程栈：请求批处理、KV cache管理和分页（vLLM等）、推测解码加速、量化（以int8/int4代替float16运行）减少内存，以及将模型分布到多个GPU。从根本上说，我们仍然在预测序列中的下一个 token，只是花了大量工程来使其更快。

所有这些都是重要的工程和研究贡献，但如果你理解了 microgpt，你就理解了算法的精髓。

---

## 常见问题（FAQ）

**模型"理解"了什么吗？**

这是一个哲学问题，但从机制上看：没有魔法发生。模型是一个大型数学函数，将输入 token 映射到下一个 token 的概率分布。在训练过程中，参数被调整以使正确的下一个 token 概率更高。这是否构成"理解"由你来判断，但机制完全包含在上面的200行代码中。

**为什么它有效？**

模型有数千个可调参数，优化器每步微调它们以使损失下降。经过许多步骤，参数稳定到捕获数据统计规律性的值。对于名字来说，这意味着：名字通常以辅音开头，"qu"倾向于一起出现，名字很少有三个连续辅音等。模型不学习显式规则，它学习一个恰好反映这些规则的概率分布。

**这和ChatGPT有什么关系？**

ChatGPT是同样的核心循环（预测下一个 token、采样、重复）的大规模放大版，加上后训练使其具有对话能力。当你和它聊天时，系统提示词、你的消息和它的回复都只是序列中的 token。模型在一个 token 接一个 token 地补全文档，就像 microgpt 补全一个名字一样。

**"幻觉"是怎么回事？**

模型通过从概率分布中采样来生成 token。它没有真理的概念，它只知道什么序列在训练数据的统计意义上是合理的。microgpt"幻觉"出一个像"karia"这样的名字，和ChatGPT自信地说出一个错误事实是同样的现象。两者都是听起来合理但碰巧不是真实的补全。

**为什么这么慢？**

microgpt在纯Python中一次处理一个标量。一个训练步需要几秒钟。在GPU上执行相同的数学运算可以并行处理数百万个标量，速度快几个数量级。

**我能让它生成更好的名字吗？**

可以。训练更长时间（增加 num_steps），增大模型（n_embd、n_layer、n_head），或使用更大的数据集。这些是在规模上同样重要的旋钮。

**如果我更换数据集会怎样？**

模型会学习数据中的任何模式。换成城市名、宝可梦名、英语单词或短诗的文件，模型就会学习生成那些。其余代码不需要改变。

---

## 社区评论与讨论总结

microgpt 发布后在技术社区引起了广泛关注和热烈讨论。以下是来自 Hacker News、Twitter/X 和 GitHub 的主要评论总结：

### 核心反馈

**高度赞誉教育价值**：社区普遍认为 microgpt 是理解LLM的最佳教育资源之一。有评论指出，许多使用LLM两年的开发者在阅读这200行代码后，才真正理解了"黑盒"内部到底发生了什么。正如一位评论者所说，在 MicroGPT、nanoGPT 和 Zero to Hero 系列之间，Karpathy 为机器学习教育所做的贡献可能超过了大多数大学课程。

**社区移植热潮**：发布后两周内，开发者们将 microgpt 移植到了 Rust、C++、Go 和 Zig 等多种语言。这说明了代码的清晰度和教育意义使得不同语言背景的开发者都能理解并重新实现它。

### Hacker News 讨论要点

**关于简化与实用**：一些讨论者注意到，microgpt 出色地展示了 GPT 的核心思想其实相当简单。正如一位评论者所言，要做有用的事情需要大量数据，然后一切开始变得越来越复杂。

**关于去除自动微分的优化**：有用户分享了一个有趣的发现——如果去掉自动微分并编写显式的反向传播，训练时间从40秒降到了5秒。

**关于可视化**：受 microgpt 启发，有开发者创建了浏览器内可视化工具，让用户可以实时观察网络中的激活传播，并点击各个组件获得解释。社区认为这比 bbycroft.net/llm 的LLM可视化更容易理解，因为可以实际运行训练循环。

**关于字符级 vs token 级分词**：部分评论者建议文章应更明确地指出 microgpt 使用字符级分词而非 token 级分词的区别和权衡。

**关于 AI 民主化**：多位评论者强调这个项目让AI世界变得更有趣、更民主化。一位博士研究者认为这是"AI透明性的基础性时刻"，展示了"智能"不需要依赖于复杂的技术栈。

### Twitter/X 讨论

Karpathy 的原推文获得了大量转发和讨论。许多知名AI从业者赞赏了该项目将GPT完整算法浓缩到一个屏幕可显示的代码量中的优雅性。人们特别欣赏的是，这个项目证明了你可以在一次阅读中真正理解LLM的工作原理，而不是把它们当作黑盒。

---

> 本文由 [Andrej Karpathy](https://karpathy.ai/) 撰写，翻译整理自原始博客文章。社区评论总结来源于 [Hacker News](https://news.ycombinator.com/item?id=47026186)、[Hacker News 原帖](https://news.ycombinator.com/item?id=47000263) 以及 [Twitter/X](https://x.com/karpathy/status/2021694437152157847) 上的讨论。
