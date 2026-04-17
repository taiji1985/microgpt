"""
microgpt —— 用200行纯Python从零实现GPT的训练和推理
=================================================

这是 Andrej Karpathy 的 microgpt 项目的详细注释版本。
原始代码地址：https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

【适合谁看？】
- 对 AI / 大语言模型（LLM）好奇的编程初学者
- 想搞清楚 ChatGPT 底层到底在干什么的人
- 有一些 Python 基础但没有机器学习背景的朋友

【一句话总结】
这个文件做了什么：读入一堆英文名字 → 训练一个迷你 GPT 模型 → 让模型"编造"出新的名字。
ChatGPT 做的是完全一样的事——只不过它的"名字"换成了整个互联网的文本，模型大了一千万倍。

【核心思路（5步）】
1. 数据准备：把文本变成数字序列
2. 自动微分：让计算机自动算出"每个参数该往哪个方向调"
3. 模型定义：搭建一个 Transformer 神经网络
4. 训练循环：反复喂数据、算误差、调参数
5. 推理生成：用训练好的模型生成新文本

@karpathy 原作 | 中文详细注释版
"""

# ============================================================================
# 第0部分：导入标准库（注意：没有任何第三方依赖！不需要 pip install 任何东西）
# ============================================================================
import os       # 用于检查文件是否存在（os.path.exists）
import math     # 用于数学运算（math.log 对数, math.exp 指数）
import random   # 用于生成随机数（初始化参数、采样等）

# 设置随机种子，保证每次运行结果一致（方便调试和复现）
# 如果去掉这行，每次运行生成的名字会不一样
random.seed(42)


# ============================================================================
# 第1部分：数据集（Dataset）
# ============================================================================
# 【目标】准备训练数据——32,000个英文名字
#
# 想象一下：你要教一个完全不懂英语的外星人"什么样的字母组合看起来像人名"。
# 你的做法就是给它看几万个真实名字，让它自己找规律。
# 这里的 GPT 模型就是那个"外星人"。
# ============================================================================

# 如果本地没有数据文件，就从网上下载
if not os.path.exists('input.txt'):
    import urllib.request  # Python 内置的网络下载工具
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
    # 下载完成后，input.txt 里的内容长这样：
    # emma
    # olivia
    # ava
    # isabella
    # ... （共约32,000个名字，每行一个）

# 读取文件，每行一个名字，去掉空白字符，存成列表
# 结果示例：docs = ["emma", "olivia", "ava", "isabella", ...]
docs = [line.strip() for line in open('input.txt') if line.strip()]

# 随机打乱顺序（让训练时每次看到的名字顺序不同，有助于学习）
random.shuffle(docs)
print(f"num docs: {len(docs)}")  # 打印：num docs: 32033


# ============================================================================
# 第2部分：分词器（Tokenizer）
# ============================================================================
# 【目标】把文字转换成数字，因为神经网络只能处理数字
#
# 类比：每个字母相当于一个"代号"
#   a → 0, b → 1, c → 2, ..., z → 25
#   BOS（特殊标记）→ 26
#
# 为什么需要 BOS？
#   BOS = Beginning of Sequence（序列开始标记）
#   它就像一个"开始/结束信号"。训练时，每个名字两边都加上 BOS：
#   "emma" → [BOS, e, m, m, a, BOS]
#   这样模型就知道：看到 BOS 就意味着"一个新名字要开始了"或"名字结束了"
# ============================================================================

# sorted(set(...)) 收集所有出现过的字符并排序
# 对于名字数据集，结果就是 ['a', 'b', 'c', ..., 'z']
uchars = sorted(set(''.join(docs)))

# BOS 的 token id 设为字符总数（这里是 26）
BOS = len(uchars)

# 词汇表大小 = 26个字母 + 1个BOS = 27
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")  # 打印：vocab size: 27


# ============================================================================
# 第3部分：自动微分引擎（Autograd）
# ============================================================================
# 【这是整个代码中最核心、最难理解的部分，但也是最优雅的部分】
#
# ★ 问题：我们怎么知道该如何调整模型的参数？
#
# 举个生活例子：
#   假设你在调收音机的旋钮想收到一个电台。你稍微往右拧了一点，信号变好了。
#   那你就知道：应该继续往右拧。
#   如果信号变差了，你就往左拧。
#   "信号变好还是变差"以及"变化了多少"——这就是"梯度"。
#
# 自动微分做的事情：
#   1. 记录所有计算过程（构建"计算图"）
#   2. 从最终结果（损失）往回推，自动算出每个参数的梯度
#   3. 梯度告诉我们：这个参数该增大还是减小，以及幅度多大
#
# 这就是 PyTorch 的 loss.backward() 在做的事情，只不过这里我们自己从头实现。
# ============================================================================

class Value:
    """
    Value 类：包装一个数字，让它具备自动求梯度的能力。

    你可以把 Value 想象成一个"智能数字"：
    - 它知道自己的值是多少（data）
    - 它知道自己是怎么被计算出来的（_children, _local_grads）
    - 训练时，它能自动算出"如果我变大一点点，最终损失会怎么变"（grad）

    生活类比：
      普通数字就像一张照片——只有最终结果。
      Value 就像一段录像——记录了整个计算过程，可以倒放（反向传播）。
    """

    # __slots__ 是 Python 的内存优化技巧
    # 告诉 Python："这个类只有这4个属性，不需要为其他属性预留空间"
    # 因为我们会创建成千上万个 Value 对象，这能节省不少内存
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        # ↑ 这个节点的实际数值（前向传播时计算得到）
        # 例如：如果 c = a + b，且 a.data=3, b.data=4，则 c.data=7

        self.grad = 0
        # ↑ 梯度：损失函数对这个节点的导数 ∂Loss/∂self
        # 初始为0，在反向传播（backward）时被计算
        # 它的含义是："如果把这个值增大一丢丢，损失会变化多少"
        # grad > 0 → 增大此值会增大损失 → 应该减小它
        # grad < 0 → 增大此值会减小损失 → 应该增大它

        self._children = children
        # ↑ 这个节点的"父母"（产生它的输入节点）
        # 例如：c = a + b，则 c._children = (a, b)
        # 这形成了一个计算图（有向无环图 DAG）

        self._local_grads = local_grads
        # ↑ 局部梯度：这个运算对每个输入的偏导数
        # 例如：c = a + b
        #   ∂c/∂a = 1, ∂c/∂b = 1 → local_grads = (1, 1)
        # 例如：c = a * b（假设 a=3, b=4）
        #   ∂c/∂a = b = 4, ∂c/∂b = a = 3 → local_grads = (4, 3)

    # ========================
    # 6种基本运算（"乐高积木"）
    # ========================
    # 整个 GPT 不管多复杂，都是由这6种基本运算组合而成的。
    # 每种运算做两件事：
    #   1. 计算结果（前向传播）
    #   2. 记录局部梯度（为反向传播做准备）

    def __add__(self, other):
        """
        加法：c = a + b

        前向：c.data = a.data + b.data
        局部梯度：∂c/∂a = 1, ∂c/∂b = 1
        直觉：a 或 b 增加1，c 也增加1（一比一传递）
        """
        other = other if isinstance(other, Value) else Value(other)
        # ↑ 如果 other 是普通数字（如 a + 3），先包装成 Value
        return Value(self.data + other.data, (self, other), (1, 1))
        #                ↑ 计算结果            ↑ 子节点      ↑ 局部梯度都是1

    def __mul__(self, other):
        """
        乘法：c = a * b

        前向：c.data = a.data * b.data
        局部梯度：∂c/∂a = b, ∂c/∂b = a
        直觉：a * b 对 a 的敏感度是 b 的大小（反过来也一样）
              比如 3 * 4 = 12，如果 a 从3变成4，c 变成 16，增加了4（= b 的值）
        """
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
        #                                                     ↑ ∂c/∂a=b  ↑ ∂c/∂b=a

    def __pow__(self, other):
        """
        幂运算：c = a^n （other 是一个普通数字，不是 Value）

        前向：c.data = a.data ^ n
        局部梯度：∂c/∂a = n * a^(n-1)  （幂函数求导法则）
        例子：a^3 的导数是 3*a^2
        """
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        """
        自然对数：c = ln(a)

        前向：c.data = ln(a.data)
        局部梯度：∂c/∂a = 1/a
        用途：计算交叉熵损失时需要 -log(概率)
        """
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        """
        指数函数：c = e^a

        前向：c.data = e^(a.data)
        局部梯度：∂c/∂a = e^a （指数函数的导数还是自己！）
        用途：softmax 中需要对 logits 取 exp
        """
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        """
        ReLU（Rectified Linear Unit，修正线性单元）：c = max(0, a)

        这是神经网络中最常用的"激活函数"之一。
        作用：如果输入是正数，原样输出；如果是负数，输出0。
        就像一个"只让正数通过"的阀门。

        前向：c.data = max(0, a.data)
        局部梯度：a > 0 时为1，a ≤ 0 时为0
        直觉：正数区域梯度畅通无阻，负数区域梯度被"关闭"
        """
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # ========================
    # 辅助运算（由上面6种基本运算组合得到）
    # ========================
    # 这些方法让 Value 对象可以像普通数字一样使用 +, -, *, / 运算符

    def __neg__(self):        return self * -1           # -a = a * (-1)
    def __radd__(self, other): return self + other       # 3 + a → a + 3
    def __sub__(self, other):  return self + (-other)    # a - b = a + (-b)
    def __rsub__(self, other): return other + (-self)    # 3 - a → 3 + (-a)
    def __rmul__(self, other): return self * other       # 3 * a → a * 3
    def __truediv__(self, other): return self * other**-1   # a / b = a * b^(-1)
    def __rtruediv__(self, other): return other * self**-1  # 3 / a = 3 * a^(-1)

    # ========================
    # 反向传播（Backward Pass）—— 自动求梯度的核心
    # ========================
    def backward(self):
        """
        反向传播：从当前节点（通常是损失函数）开始，自动计算所有节点的梯度。

        【算法流程】
        1. 构建拓扑排序（确保处理某个节点时，所有依赖它的下游节点已处理完）
        2. 从损失节点开始，设 grad = 1（∂L/∂L = 1）
        3. 按逆拓扑序遍历每个节点，用链式法则传递梯度

        【链式法则直觉】
        假设有连锁反应：a → b → c → Loss
        - Loss 对 c 的敏感度是 ∂L/∂c（已知）
        - c 对 b 的敏感度是 ∂c/∂b（局部梯度，前向时已记录）
        - 那么 Loss 对 b 的敏感度 = ∂L/∂c × ∂c/∂b（两个敏感度相乘）

        就像多米诺骨牌：推倒第一张牌的力量，会沿着链条传递下去。
        """

        # 第1步：拓扑排序
        # 把计算图中的所有节点排成一个线性序列，使得每个节点排在它的所有子节点之后
        # 这样反向遍历时，处理到某个节点时，它的"下游"（离损失更近的方向）都已算完了
        topo = []
        visited = set()  # 记录已访问的节点，避免重复

        def build_topo(v):
            """深度优先搜索，后序遍历，构建拓扑排序"""
            if v not in visited:
                visited.add(v)
                for child in v._children:  # 先递归处理所有子节点
                    build_topo(child)
                topo.append(v)  # 子节点都处理完了，再把自己加入
        build_topo(self)

        # 第2步：起点——损失对自身的梯度是1
        # 因为 ∂L/∂L = 1（任何东西对自身的变化率是1）
        self.grad = 1

        # 第3步：反向遍历，传递梯度
        for v in reversed(topo):  # 从损失节点开始，往输入方向走
            for child, local_grad in zip(v._children, v._local_grads):
                # 链式法则核心公式：
                #   ∂L/∂child += ∂v/∂child × ∂L/∂v
                #   即：子节点的梯度 += 局部梯度 × 当前节点的梯度
                #
                # 为什么是 += 而不是 = ？
                # 因为一个节点可能被多个下游节点使用（图分叉了）
                # 比如 a 同时参与了 c = a*b 和 d = a+b
                # 那么 a 的梯度 = 通过 c 传来的 + 通过 d 传来的
                child.grad += local_grad * v.grad


# ============================================================================
# 第4部分：模型参数初始化
# ============================================================================
# 【目标】创建模型的所有可学习参数，初始化为小随机数
#
# 类比：这些参数就像收音机上的几千个旋钮，初始时随机拨了一下。
# 训练过程就是不断微调这些旋钮，直到收音机能放出好听的音乐。
#
# 为什么不初始化为0？
#   如果所有参数都是0，那所有神经元的输出都一样，梯度也一样，
#   它们就永远无法分化出不同的功能——就像一个合唱团所有人唱同一个音。
#   小随机数打破了这种"对称性"。
# ============================================================================

# --- 超参数（Hyperparameters）---
# 这些是我们手动设定的"设计图纸"参数，控制模型的大小和形状
n_layer = 1         # Transformer 的层数（深度）。GPT-3 有96层，我们只用1层
n_embd = 16         # 嵌入维度（宽度）。GPT-3 是 12288，我们只用16
block_size = 16     # 最长能处理的序列长度。最长的名字是15个字符，16够用了
n_head = 4          # 注意力头的数量。多个头可以关注不同类型的模式
head_dim = n_embd // n_head  # 每个头的维度 = 16 / 4 = 4

# 创建参数矩阵的工具函数
# 每个参数是一个 Value 对象，初始值从 N(0, 0.08²) 高斯分布中采样
# nout × nin 的矩阵 = nout 行、nin 列
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)]  # 一行有 nin 个参数
    for _ in range(nout)                                 # 共 nout 行
]

# --- 参数字典（state_dict）---
# 借用 PyTorch 的命名习惯，按名字存储所有参数矩阵
state_dict = {
    'wte': matrix(vocab_size, n_embd),    # Token嵌入表：27×16
    # ↑ 每个 token（字母或BOS）对应一个16维向量
    # 你可以理解为：给26个字母+BOS 各分配一个"身份证"，身份证上有16个数字
    # 这些数字一开始是随机的，训练后会变得有意义（相似的字母距离更近）

    'wpe': matrix(block_size, n_embd),    # 位置嵌入表：16×16
    # ↑ 每个位置（0到15）对应一个16维向量
    # 告诉模型"这个字母在名字中的第几个位置"
    # 位置很重要！名字开头和结尾的字母分布完全不同

    'lm_head': matrix(vocab_size, n_embd) # 输出投影：27×16
    # ↑ 把模型内部的16维向量转换回27个分数（logits）
    # 每个分数对应一个 token，分数越高 → 模型越觉得这个 token 应该出现
}

# 每一层 Transformer 的参数
for i in range(n_layer):
    # --- 注意力（Attention）的参数 ---
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # Query 权重：16×16
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # Key 权重：16×16
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # Value 权重：16×16
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # 输出投影：16×16
    # ↑ Q/K/V 是注意力机制的三个核心角色（后面会详细解释）

    # --- MLP（多层感知机）的参数 ---
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # 第一层：64×16（扩展4倍）
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # 第二层：16×64（压缩回来）
    # ↑ MLP 先把16维扩展到64维（给模型更大的"思考空间"），再压缩回16维

# 把所有参数展平成一个大列表，方便优化器统一遍历
# 想象把所有旋钮编了号，优化器按编号一个一个调
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")  # 打印：num params: 4192
# 我们的模型有 4,192 个参数。GPT-2 有 16 亿个，GPT-4 有数千亿个。
# 算法完全一样，只是规模天差地别。


# ============================================================================
# 第5部分：模型架构（GPT Model）
# ============================================================================
# 【目标】定义 GPT 模型的计算过程
#
# 架构遵循 GPT-2，做了一些简化：
#   - LayerNorm → RMSNorm（更简单的归一化）
#   - GeLU → ReLU（更简单的激活函数）
#   - 去掉了所有偏置（bias）
#
# 数据流：
#   输入 token → 嵌入 → [归一化 → 注意力 → 残差] → [归一化 → MLP → 残差] → 输出 logits
#
# 直觉：
#   注意力（Attention）= 不同位置的 token 之间"互相交流信息"
#   MLP = 每个 token 自己"思考消化"刚得到的信息
#   两者交替进行，就像一个讨论会：先讨论（注意力），再各自思考（MLP），再讨论...
# ============================================================================

def linear(x, w):
    """
    线性变换：y = W × x（矩阵乘向量）

    参数：
        x: 输入向量，长度为 nin 的列表 [Value, Value, ...]
        w: 权重矩阵，nout × nin 的二维列表

    返回：
        输出向量，长度为 nout 的列表

    例子：
        如果 x = [1, 2, 3]，w = [[1,0,0], [0,1,0]]
        结果 = [1*1+0*2+0*3, 0*1+1*2+0*3] = [1, 2]

    这是神经网络最基本的操作。每一行做一个点积（dot product）。
    """
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
    #       ↑ 对 w 的每一行 wo，计算 wo·x（点积）


def softmax(logits):
    """
    Softmax 函数：把任意数字变成概率分布

    输入：一组"分数"（logits），可以是任意实数，比如 [2.0, 1.0, 0.1]
    输出：概率分布，所有值在0-1之间且求和为1，比如 [0.66, 0.24, 0.10]

    公式：P(i) = exp(z_i) / Σ exp(z_j)

    为什么要减去 max_val？（log-sum-exp trick）
      假设 logits 里有个很大的数比如 1000，
      exp(1000) 会直接变成无穷大（数值溢出）！
      但 exp(1000 - 1000) = exp(0) = 1，完全没问题。
      减去最大值不改变 softmax 的结果（因为分子分母同时乘除相同的数）。
    """
    max_val = max(val.data for val in logits)         # 找到最大值
    exps = [(val - max_val).exp() for val in logits]  # 减去最大值后取 exp
    total = sum(exps)                                  # 求和
    return [e / total for e in exps]                   # 归一化为概率


def rmsnorm(x):
    """
    RMSNorm（Root Mean Square Normalization，均方根归一化）

    作用：把向量的"大小"归一化到大约为1。

    为什么需要它？
      想象你在传话游戏中传一个数字。每传一次可能放大或缩小一点。
      传100次后，数字可能变得巨大或微小到接近0。
      归一化就像每传一次后重新校准大小，防止数字失控。

    在神经网络中，数据经过很多层变换，如果不归一化，
    激活值可能会"爆炸"（变得极大）或"消失"（变得极小），导致训练失败。

    公式：x̂_i = x_i / √(mean(x²) + ε)
    其中 ε = 1e-5 是个很小的数，防止除以0。
    """
    ms = sum(xi * xi for xi in x) / len(x)  # 计算均方值（mean square）
    scale = (ms + 1e-5) ** -0.5              # 1/√(ms + ε)
    return [xi * scale for xi in x]          # 每个元素除以 RMS


def gpt(token_id, pos_id, keys, values):
    """
    GPT 模型：给定一个 token 和它的位置，预测下一个 token 的概率分布。

    参数：
        token_id: 当前输入 token 的编号（0-26）
        pos_id:   当前 token 在序列中的位置（0-15）
        keys:     KV缓存中的 Key（之前位置的"钥匙"）
        values:   KV缓存中的 Value（之前位置的"信息"）

    返回：
        logits: 27个分数，每个对应词汇表中的一个 token
                分数越高 → 模型越认为该 token 应该出现在下一个位置

    【完整数据流】

    1. 嵌入（Embedding）
       "我是字母 e，我在第2个位置" → 变成一个16维数字向量

    2. 注意力（Attention）—— token 之间的"对话"
       每个 token 问自己："之前的 token 中，哪些和我相关？"
       然后从相关的 token 那里获取信息

    3. MLP —— 每个 token 自己"思考"
       消化刚从其他 token 获取的信息

    4. 输出
       把最终的16维向量转换为27个分数
    """

    # ---- 第1步：嵌入（Embedding）----
    # 把 token_id 和 pos_id 分别查表，得到两个16维向量，然后相加
    tok_emb = state_dict['wte'][token_id]  # Token 嵌入：查找"这个字母的身份证"
    pos_emb = state_dict['wpe'][pos_id]    # 位置嵌入：查找"这个位置的特征"
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 两者相加 → 16维向量
    # 现在 x 同时包含了"我是什么字母"和"我在第几个位置"的信息

    x = rmsnorm(x)  # 归一化，稳定数值

    # ---- 第2步 & 第3步：Transformer 层 ----
    for li in range(n_layer):  # 遍历每一层（我们只有1层）

        # ========================================
        # 2A) 多头注意力（Multi-Head Attention）
        # ========================================
        # 【核心直觉】
        # 注意力机制让当前 token "看到"之前所有 token 的信息。
        #
        # 三个角色（QKV）：
        #   Q (Query, 查询)：  "我在找什么样的信息？"
        #   K (Key, 键/钥匙)："我拥有什么样的信息？"
        #   V (Value, 值)：   "如果你选中我，我能给你什么？"
        #
        # 过程：
        #   1. 当前 token 生成一个 Query："我在找以元音开头的信息"
        #   2. 每个历史 token 都有一个 Key："我包含辅音相关信息"
        #   3. Query 和每个 Key 做点积 → 得到"相关度分数"
        #   4. Softmax 归一化分数 → 注意力权重（加权比例）
        #   5. 用权重对所有历史 token 的 Value 加权求和 → 获取信息
        #
        # 多头（Multi-Head）：
        #   4个头各自独立做注意力，关注不同类型的模式。
        #   比如头1关注"前一个字母是什么"，头2关注"名字开头是什么"。
        #   最后把4个头的结果拼起来。

        x_residual = x  # 保存输入，用于残差连接
        x = rmsnorm(x)  # 归一化

        # 生成 Q, K, V（三次线性变换）
        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # Query：16维
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # Key：16维
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # Value：16维

        # 把当前位置的 K 和 V 加入缓存
        # 这样下一个 token 处理时，可以"看到"当前 token 的信息
        keys[li].append(k)
        values[li].append(v)

        x_attn = []  # 存储所有注意力头的输出

        for h in range(n_head):  # 遍历每个注意力头
            # 每个头只看16维中自己负责的那4维（head_dim = 4）
            hs = h * head_dim  # 起始索引
            q_h = q[hs:hs+head_dim]                            # 当前 token 的 Query 片段
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]      # 所有历史 token 的 Key 片段
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]    # 所有历史 token 的 Value 片段

            # 计算注意力分数：Q 和每个 K 的点积，除以 √(head_dim) 缩放
            # 为什么要除以 √(head_dim)？
            #   点积的结果大小和维度成正比。如果不缩放，维度大时点积会很大，
            #   softmax 后分布会极端尖锐（接近 one-hot），梯度接近0，训练不动。
            #   除以 √d 让方差回到1，softmax 分布适度平滑。
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]

            # Softmax → 注意力权重（加起来等于1的概率分布）
            attn_weights = softmax(attn_logits)
            # 例如：attn_weights = [0.1, 0.3, 0.6]
            # 意味着当前 token 对位置0关注10%，位置1关注30%，位置2关注60%

            # 用注意力权重对 V 加权求和 → 该头的输出
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            # 直觉：从历史 token 中"提取"信息，关注度高的贡献更大

            x_attn.extend(head_out)  # 把这个头的4维输出追加到总输出

        # 所有头的输出拼接后（4头×4维=16维），做一次线性变换混合
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])

        # ★ 残差连接（Residual Connection）★
        # x = attention_output + original_input
        # 为什么？两个好处：
        #   1. 梯度直通：反向传播时，梯度可以直接跳过注意力层回到输入
        #      （加法的梯度是1，不会衰减），防止"梯度消失"
        #   2. 学的是"增量"：注意力层只需要学"该在原始信息上加什么"，而不是从零开始
        x = [a + b for a, b in zip(x, x_residual)]

        # ========================================
        # 2B) MLP（多层感知机）
        # ========================================
        # 注意力负责 token 之间"交流"，MLP 负责每个 token 独立"思考"。
        #
        # 结构：16维 → 64维（扩展，获得更大的表达空间）
        #       → ReLU（非线性激活，让网络能学复杂模式）
        #       → 16维（压缩回来）
        #
        # 为什么需要非线性（ReLU）？
        #   如果只有线性变换（矩阵乘法），不管叠多少层，效果都等于一个矩阵。
        #   加入 ReLU 后，网络就能学习复杂的非线性模式。

        x_residual = x   # 再次保存用于残差连接
        x = rmsnorm(x)   # 归一化
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 16维 → 64维
        x = [xi.relu() for xi in x]                       # ReLU 激活
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # 64维 → 16维
        x = [a + b for a, b in zip(x, x_residual)]       # 残差连接

    # ---- 第4步：输出层 ----
    # 把16维的隐藏状态投影到27维（词汇表大小）
    # 每个维度对应一个 token 的"分数"（logit）
    logits = linear(x, state_dict['lm_head'])
    return logits
    # logits 示例：[2.1, -0.5, 1.3, ..., 0.8]（27个数字）
    # 数字越大 → 模型越觉得对应的字母应该是下一个


# ============================================================================
# 第6部分：训练循环（Training Loop）
# ============================================================================
# 【目标】通过反复看数据来调整参数，让模型学会名字的统计规律
#
# 每一步训练做4件事：
#   1. 选一个名字，转成数字序列
#   2. 让模型逐个预测下一个字母（前向传播）
#   3. 计算预测有多差（损失），然后反向传播算梯度
#   4. 用 Adam 优化器微调所有参数
#
# 类比：
#   这就像背单词：看一个单词 → 尝试拼写 → 对答案 → 调整记忆 → 重复
# ============================================================================

# --- Adam 优化器 ---
# Adam 是目前最流行的优化算法之一（几乎所有LLM都用它）
# 比普通梯度下降更智能，因为它有两个"记忆"：
#   m（动量/momentum）：梯度的移动平均 → 平滑方向，减少震荡
#   v（自适应学习率）：梯度平方的移动平均 → 让每个参数有自己合适的步长
#     梯度一直很大的参数 → 步子小一点（已经在快速变化了）
#     梯度一直很小的参数 → 步子大一点（需要加速）

learning_rate = 0.01   # 学习率：每步调整参数的"步幅"
beta1 = 0.85           # 动量的衰减系数（通常0.9左右）
beta2 = 0.99           # 二阶矩的衰减系数（通常0.999左右）
eps_adam = 1e-8         # 防止除以0的小数

m = [0.0] * len(params)  # 一阶矩缓冲区（梯度的移动平均），初始为0
v = [0.0] * len(params)  # 二阶矩缓冲区（梯度平方的移动平均），初始为0

# --- 开始训练 ---
num_steps = 1000  # 总共训练1000步（可以增大来获得更好的效果）

for step in range(num_steps):

    # ---- 步骤1：准备数据 ----
    # 选一个名字，加上 BOS 标记
    doc = docs[step % len(docs)]  # 循环使用数据集中的名字
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    # 例如 "emma" → [26, 4, 12, 12, 0, 26]
    #                BOS  e   m   m   a  BOS
    n = min(block_size, len(tokens) - 1)
    # n = 需要预测的位置数（= token数 - 1，因为最后一个没有"下一个"需要预测）

    # ---- 步骤2：前向传播（Forward Pass）----
    # 逐个 token 送入模型，让它预测下一个 token
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    # ↑ 清空 KV 缓存（每个新名字从头开始）

    losses = []  # 记录每个位置的损失

    for pos_id in range(n):
        token_id = tokens[pos_id]       # 当前输入 token
        target_id = tokens[pos_id + 1]  # 正确答案：下一个 token

        # 模型预测
        logits = gpt(token_id, pos_id, keys, values)  # 得到27个分数
        probs = softmax(logits)                         # 转成概率

        # 计算损失：-log(正确答案的概率)
        loss_t = -probs[target_id].log()
        # 为什么是 -log？
        #   如果模型很确定（概率=0.9）：-log(0.9) = 0.105（损失小 ✓）
        #   如果模型很不确定（概率=0.01）：-log(0.01) = 4.6（损失大 ✗）
        #   如果模型完美预测（概率=1.0）：-log(1.0) = 0（损失为0 ★）
        # 所以：概率越高 → 损失越低 → 我们的目标就是让损失尽可能低

        losses.append(loss_t)

    # 平均损失 = 所有位置损失的平均
    loss = (1 / n) * sum(losses)

    # ---- 步骤3：反向传播（Backward Pass）----
    # 一行代码，从损失出发，自动算出所有4192个参数的梯度
    loss.backward()
    # 执行完后，每个参数的 .grad 都被填上了值
    # 告诉我们："要降低损失，这个参数应该增大还是减小，幅度多大"

    # ---- 步骤4：Adam 优化器更新参数 ----
    lr_t = learning_rate * (1 - step / num_steps)  # 学习率线性衰减
    # ↑ 训练后期减小步幅，让模型"精细调整"而不是大幅跳动

    for i, p in enumerate(params):
        # 更新一阶矩（梯度的指数移动平均 → 平滑方向）
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        # ↑ 85% 保留旧的方向 + 15% 融入新的梯度

        # 更新二阶矩（梯度平方的指数移动平均 → 衡量波动性）
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        # ↑ 99% 保留旧的波动估计 + 1% 融入新的

        # 偏差修正（Bias correction）
        # m 和 v 初始为0，前几步的估计值偏小，需要放大
        # 随着 step 增大，修正系数趋近于1（不再需要修正）
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))

        # ★ 核心更新公式 ★
        # 参数 -= 学习率 × 梯度方向 / √(波动性)
        # 梯度方向（m_hat）决定往哪走
        # 波动性（√v_hat）决定步子多大（波动大→小步，波动小→大步）
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)

        # 梯度清零，为下一步做准备
        p.grad = 0
        # ↑ 不清零的话，下一步的梯度会累加到旧梯度上，结果就错了

    # 打印训练进度
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')
    # 初始 loss ≈ 3.3（随机猜测 27 选 1：-log(1/27) ≈ 3.3）
    # 训练后 loss ≈ 2.37（模型学到了一些规律，但还不完美）
    # loss 越低，说明模型预测得越准


# ============================================================================
# 第7部分：推理 / 生成（Inference / Generation）
# ============================================================================
# 【目标】用训练好的模型生成新名字
#
# 过程（自回归生成）：
#   1. 输入 BOS（"开始一个新名字"）
#   2. 模型输出27个概率 → 按概率随机选一个字母
#   3. 把选中的字母作为下一步的输入
#   4. 重复，直到模型输出 BOS（"名字结束"）或达到最大长度
#
# 这和 ChatGPT 的工作方式完全一样！
# 只不过 ChatGPT 的 token 是词/词块，生成的是句子而非名字。
# ============================================================================

# 温度（Temperature）：控制生成的"创造力"
# temperature = 0.5  → 比较保守，倾向选概率高的字母（生成的名字更"正常"）
# temperature = 1.0  → 原始分布，多样性适中
# temperature = 2.0  → 很随机，会产生奇怪的名字
# temperature → 0    → 每次都选概率最高的那个（贪心解码，完全没有随机性）
#
# 原理：在 softmax 之前，把 logits 除以 temperature
# 小温度 → logits 的差距被放大 → softmax 更尖锐 → 更确定
# 大温度 → logits 的差距被缩小 → softmax 更平坦 → 更随机
temperature = 0.5

print("\n--- inference (new, hallucinated names) ---")

for sample_idx in range(20):
    # 每个新名字都从空白开始
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS  # 以 BOS 开始
    sample = []     # 收集生成的字母

    for pos_id in range(block_size):  # 最多生成 block_size 个字符
        # 前向传播：让模型预测下一个字母的概率
        logits = gpt(token_id, pos_id, keys, values)

        # 应用温度缩放后做 softmax
        probs = softmax([l / temperature for l in logits])

        # 按概率分布随机采样一个 token
        token_id = random.choices(
            range(vocab_size),                       # 候选：0-26
            weights=[p.data for p in probs]          # 权重：每个候选的概率
        )[0]

        # 如果采样到 BOS，说明模型认为名字应该结束了
        if token_id == BOS:
            break

        # 否则，把对应的字母加入结果
        sample.append(uchars[token_id])

   
    # 大多数生成的名字不在原始数据集中——它们是模型"编造"的！
    # 但它们听起来像真名字，因为模型学到了英文名字的统计规律。
    # 这就是所谓的"幻觉"（hallucination），和 ChatGPT 编造事实是同一个现象。


# ============================================================================
# 总结
# ============================================================================
#
# 恭喜你看完了！你刚刚理解了 ChatGPT 的核心算法。
#
# 回顾一下这 200 行代码做了什么：
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  数据集          32,000个英文名字                                 │
# │      ↓                                                           │
# │  分词器          字母 → 数字（a=0, b=1, ..., z=25, BOS=26）      │
# │      ↓                                                           │
# │  自动微分        Value 类，自动算梯度（反向传播）                  │
# │      ↓                                                           │
# │  模型            Transformer：嵌入 → 注意力 → MLP → 输出          │
# │      ↓                                                           │
# │  训练            1000步：前向 → 算损失 → 反向 → Adam更新           │
# │      ↓                                                           │
# │  推理            用训练好的模型生成新名字                          │
# └──────────────────────────────────────────────────────────────────┘
#
# 从 microgpt 到 ChatGPT，算法完全一样，区别只在于：
#   - 数据量：32K 名字 → 数万亿 token 的互联网文本
#   - 模型大小：4,192 参数 → 数千亿参数
#   - 训练资源：你的笔记本1分钟 → 数千GPU跑几个月
#   - 后训练：无 → SFT（监督微调）+ RLHF（人类反馈强化学习）
#
# 正如 Karpathy 所说：
# "This file is the complete algorithm. Everything else is just efficiency."
# "这个文件就是完整的算法。其他一切都只是为了效率。"
