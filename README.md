# GRPO:Zero

GRPO training with minimal dependencies (and low GPU memory usage!). We implement almost everything from scratch and only depend on `tokenizers` for tokenization and `pytorch` for training. 
- No `transformers` and `vLLM` dependencies! 
- The default config is set to run on a single A40 GPU (48GB VRAM) for a few hours to get good results. (An A40 costs `$0.44` per hour if you rent it from RunPod.)
- We also support training with a 24GB VRAM GPU (e.g., an RTX 4090 GPU) by offloading the optimizer to CPU. Fortunately, this only adds a small overhead to the training because we only update the policy network a few hundred times during the entire training process.
- We support several improvements over the original GRPO algorithm from the [DAPO project](https://arxiv.org/abs/2503.14476), including:
    - **Token-level policy gradient loss**: every token is equally weighted in the policy gradient loss.
    - **Removing KL Divergence**: the KL divergence is not used in the policy gradient loss. This reduces GPU memory usage as we no longer need the reference policy network.
    - **Overlong episode filtering**: skips unfinished episodes that exceed context length limits. This stabilizes training. Though we disabled it by default to observe model learning under limited context length. Set `skip_unfinished_episodes` to `true` to enable it.

## Algorithm 

Group Relative Policy Optimization (GRPO) is an algorithm proposed by Deepseek for training large language models with reinforcement learning. The idea is simple: for each question, we randomly sample multiple answers. The advantage of an answer is then defined as the normalized reward. This gets rid of the value estimation network. In particular, we implement the following algorithm:

1. For each training step, randomly sample $N$ questions $q_1, q_2, \cdots, q_N$.
2. For each question $q_i$, sample $M$ answers $a_{i,1}, a_{i,2}, \cdots, a_{i,M}$.
3. Compute the reward $r_{i,j}$ for each answer $a_{i,j}$.
4. Compute the mean and std of the rewards for each question $q_i$.

$$
\begin{aligned}
\mu_i &\leftarrow \text{mean}(r_{i,1}, r_{i,2}, \cdots, r_{i,M}) \\
\sigma_i &\leftarrow \text{std}(r_{i,1}, r_{i,2}, \cdots, r_{i,M})
\end{aligned}
$$

5. For each token $t$ in the answer $a_{i,j}$, compute the advantage as

$$A_{i,j}[t] \leftarrow \frac{r_{i,j} - \mu_i}{\sigma_i}$$

6. Compute policy gradient using PPO surrogate objective. For simplicity, we will only do one policy update per iteration, in which the gradient of the PPO objective is equivalent to following vanilla policy gradient estimation (per token).

$$
\nabla_\theta \log \pi_\theta(a_{i,j}[t]) \cdot A_{i,j}[t]
$$

7. Update the policy network $\pi(\theta)$ using the gradient. Go back to step 1.

## CountDown Task

We are going to train the Qwen2.5 models on the [CountDown task](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4). Given a list of 3 or 4 numbers and a target number, the model needs to generate a mathematical expression using simple arithmetic operations (+, -, *, /) that evaluates to the target number. For example:

```
Question: Given 1 2 3 4 and a target number 11. Show an expression that evaluates to 11.
Answer: 1 + (2 * 3) + 4
```

## Reward Function

To solve the CountDown task, we will use the GRPO algorithm to train the model to generate the chain of thought reasoning before generating the final expression. Specifically, the model is trained to follow the format:

```
<think>Model step by step reasoning</think>
<answer>Final answer</answer>
```

The reward is the sum of two components:

1. **Format Reward**: The model earns a reward of `0.1` when it correctly follows the specified format with thinking and answer tags, and `0` otherwise.
2. **Answer Reward**: The model receives a reward of `1` if its final answer uses each provided number exactly once and correctly evaluates to the target value, otherwise it receives `0`.


## Training

We use the `Qwen2.5-3B-Instruct` model for training. To train the model, run the following commands:

```bash
# initialize the environment
pip install uv
uv sync

# install git-lfs
apt update; apt install git-lfs -y; git lfs install

# download the dataset
git clone https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4

# download the pretrained model
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
# train the model
uv run train.py
# train the model with a 24GB VRAM GPU (e.g., an RTX 4090 GPU)
uv run train.py --config config_24GB.yaml
```
## Acknowledgements

This project builds upon the work of several outstanding projects:

- [DeepSeekMath](https://arxiv.org/abs/2402.03300) for pioneering the GRPO algorithm.
- [DAPO](https://arxiv.org/abs/2503.14476) for their enhancements to the original GRPO algorithm.
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) for their implementation of GRPO and creation of the [CountDown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) dataset.
- [nano-aha-moment](https://github.com/McGill-NLP/nano-aha-moment/tree/main) for their clear implementation and tutorial on the GRPO algorithm.
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) for developing the high-quality pretrained model used in this project.

## DPO
语⾔模型是⾃回归的：逐个token⽣成⽂本，以之前的所有token为条件。对于响应$y=(y_1, y_2,...,y_T)$ ，概率分解为
$$\pi_\theta(y|x) = \prod_{t=1}^{T} \pi_\theta(y_t | x, y_1, \ldots, y_{t-1}) = \prod_{t=1}^{T}
\pi_\theta(y_t | x, y_{<t}) $$

取对数有：
$$\boxed{\log \pi_\theta(y|x) = \sum_{t=1}^{T} \log \pi_\theta(y_t | x, y_{<t})} \tag{vii.i}
$$

即完整响应的对数概率是每个位置对数概率的总和

代码实现时, 对数概率求和，但只对响应token求和(不包括提⽰token)
$$\log \pi_\theta(y|x) = \sum_{t \in \text{响应位置}} \log \pi_\theta(y_t | x, y_{<t}) $$



## GRPO 伪代码


```python

# ========== 1. Rollout Generation Phase ==========
prompt = "Question: Which is bigger? 9.11 or 9.9?"

# Generate multiple completions through parallel sampling
completions = rollout_function(
    model=current_policy_model,
    prompt=prompt,
    num_generations=8,  # Hyperparameter: number of samples per prompt
    temperature=1.0     # Hyperparameter: sampling diversity
)
"""
completions = [
    (completion 1) "The larger number is 9.11...",
    (completion 2) "9.9 is bigger than...",
    ...
    (completion 8) "After calculation, 9.11..."
]
"""

# ========== 2. Reward Calculation Phase ==========
# Evaluate generated completions using reward model
rewards = reward_function(
    completions=completions,
    ground_truth="9.11"  # Expected correct answer
)
"""
rewards = [
    (reward 1) 1.0,  # Correct answer
    (reward 2) 0.0,  # Incorrect
    ...
    (reward 8) 1.0   # Correct
]
"""

# Normalize rewards to advantages
rewards_mean = mean(rewards)  # μ = 0.5
rewards_std = std(rewards)    # σ = 0.25
advantages = (rewards - rewards_mean) / (rewards_std + 1e-8)  # Standardization
"""
advantages = [
    (advantage 1)  2.0,  # (1.0 - 0.5)/0.25
    (advantage 2) -2.0,
    ...
    (advantage 8)  2.0
]
"""

# ========== 3. Policy Optimization Phase ==========
# Get token-level log probabilities from different models, 注意是 log_probs
current_logps = get_per_token_logps(current_policy_model, prompt, completions)  # π_θ
old_logps = get_per_token_logps(old_policy_model, prompt, completions)          # π_θ_old
ref_logps = get_per_token_logps(reference_model, prompt, completions)           # π_ref

# PPO Clipped Objective
is_ratio = exp(current_logps - old_logps)  # Importance sampling ratio: e^(π_θ - π_θ_old)
clipped_ratio = clip(is_ratio, 1-ε, 1+ε)   # ε=0.2 typically

# Policy gradient term (dual form)
policy_loss = -mean(
    minimum(is_ratio * advantages,       # Unclipped objective
           clipped_ratio * advantages)  # Clipped objective
)

# KL Divergence Penalty (K3 estimator)
# KL(π_θ||π_ref) ≈ e^(logπ_ref - logπ_θ) - (logπ_ref - logπ_θ) - 1
# GRPO（Group Relative Policy Optimization）使用 K3 estimator 而不是直接计算 KL 散度，主要是出于 数值稳定性、计算效率和梯度性质 的考虑
kl_penalty = beta * mean(
    exp(ref_logps - current_logps) -
    (ref_logps - current_logps) - 1
)

# Total Loss = Policy Loss + KL Penalty
total_loss = policy_loss + kl_penalty

# ========== 4. Update Rule ==========
# Apply gradient descent to minimize total_loss
optimizer.zero_grad()
total_loss.backward()
optimizer.step()

```


GRPO（Group Relative Policy Optimization）使用 **K3 estimator** 而不是直接计算 KL 散度，主要是出于 **数值稳定性、计算效率和梯度性质** 的考虑。让我详细解释：

## **1. KL 散度的两种计算方式对比**

### **直接 KL 散度计算**
```python
# 直接计算 KL(π_θ || π_ref)
kl_direct = current_logps - ref_logps  # log(π_θ/π_ref)
kl_loss = mean(kl_direct)  # 这不是正确的 KL！需要 exp 和期望
# 正确的 KL: KL = E_{π_θ}[log(π_θ/π_ref)] = mean(exp(current_logps) * (current_logps - ref_logps))
```

### **K3 estimator（实际使用）**
```python
# K3 estimator: e^(Δ) - Δ - 1，其中 Δ = logπ_ref - logπ_θ
kl_penalty = mean(
    exp(ref_logps - current_logps) -  # e^(logπ_ref - logπ_θ)
    (ref_logps - current_logps) -     # (logπ_ref - logπ_θ)
    1                                 # -1
)
```

## **2. 为什么选择 K3 estimator？**

### **原因1：数值稳定性（最重要）**
```python
import torch

# 假设概率差异很大
ref_logps = torch.tensor([-10.0])   # π_ref 很小
current_logps = torch.tensor([-1.0])  # π_θ 相对较大

# 直接计算会数值不稳定
ratio = exp(current_logps - ref_logps)  # exp(9) ≈ 8103，可能溢出
kl_direct = ratio * (current_logps - ref_logps)  # 8103 * 9 ≈ 72927

# K3 estimator 更稳定
delta = ref_logps - current_logps  # -9
kl_k3 = exp(delta) - delta - 1     # exp(-9) + 9 - 1 ≈ 0.000123 + 8 ≈ 8.000123
```

### **原因2：梯度性质更好**
```python
# K3 estimator 的梯度
# L = e^(Δ) - Δ - 1，其中 Δ = logπ_ref - logπ_θ
# dL/dθ = (e^(Δ) - 1) * (-dΔ/dθ)
#       = (1 - e^(Δ)) * d(logπ_θ)/dθ

# 当 π_θ ≈ π_ref 时（训练后期）：
# Δ ≈ 0，e^(Δ) ≈ 1，dL/dθ ≈ 0 * d(logπ_θ)/dθ ≈ 0
# 梯度自然变小，训练更稳定
```

### **原因3：与重要性采样的一致性**
K3 estimator 实际上来自 **重要性采样** 的 Taylor 展开：

```python
# KL(π_θ||π_ref) = E_{π_θ}[log(π_θ/π_ref)]
# 用重要性采样改写：E_{π_ref}[ (π_θ/π_ref) * log(π_θ/π_ref) ]
# 令 r = π_θ/π_ref = exp(current_logps - ref_logps)
# KL = E_{π_ref}[r * log(r)]

# 对 f(r) = r * log(r) 在 r=1 处进行二阶 Taylor 展开：
# f(r) ≈ f(1) + f'(1)*(r-1) + 1/2*f''(1)*(r-1)²
#      = 0 + 1*(r-1) + 1/2*(1)*(r-1)²
#      = (r-1) + 1/2(r-1)²

# 令 δ = log(r) = current_logps - ref_logps
# 当 r ≈ 1 时，δ ≈ r-1，所以 (r-1)² ≈ δ²
# 因此 KL ≈ δ + 1/2 δ²

# 而 K3 estimator 是：
# e^(-δ) - (-δ) - 1 = e^(-δ) + δ - 1
# 对 e^(-δ) 在 δ=0 处 Taylor 展开：1 - δ + δ²/2 - ...
# 所以 e^(-δ) + δ - 1 ≈ (1 - δ + δ²/2) + δ - 1 = δ²/2
# 这正是 KL 的二阶近似！
```

## **3. 实际实现对比**

### **方法A：朴素 KL（问题多）**
```python
def naive_kl_loss(current_logps, ref_logps):
    ratio = torch.exp(current_logps - ref_logps)  # 可能溢出
    kl = ratio * (current_logps - ref_logps)      # 可能数值不稳定
    return kl.mean()
```

### **方法B：K3 estimator（GRPO 使用）**
```python
def k3_kl_loss(current_logps, ref_logps):
    delta = ref_logps - current_logps  # log(π_ref/π_θ)
    kl = torch.exp(delta) - delta - 1.0
    return kl.mean()
```

### **方法C：稳定版直接计算**
```python
def stable_kl_loss(current_logps, ref_logps):
    # 使用 log-sum-exp 技巧
    log_ratio = current_logps - ref_logps
    # 限制 log_ratio 的范围防止溢出
    log_ratio_clamped = torch.clamp(log_ratio, min=-10, max=10)
    kl = torch.exp(log_ratio_clamped) * log_ratio
    return kl.mean()
```

## **4. K3 estimator 的数学推导**

K3 estimator 来自 **三次交叉熵（Cross Entropy of order 3）**：

```
定义 f-divergence：D_f(P||Q) = E_Q[f(P/Q)]

取 f(t) = t*log(t) - (t-1) 得到 KL 散度
但计算困难，使用 surrogate function：

g(δ) = e^(-δ) + δ - 1，其中 δ = log(Q/P)

性质：
1. g(0) = 0
2. g'(0) = 0
3. g''(0) = 1
4. g(δ) ≥ 0，等号当且仅当 δ=0

所以 g(δ) 是 δ²/2 的上界，且更平滑。
```

## **5. 在 PPO/GRPO 中的实际优势**

### **优势1：与 clip 机制协同**
```python
# PPO 的 clip 机制限制了 ratio = π_θ/π_old 在 [1-ε, 1+ε]
# 相应地，log(π_ref/π_θ) 也被限制在一定范围
# K3 estimator 在这个范围内近似二次函数，梯度平滑

# 如果 π_θ 偏离 π_ref 太远：
# - 直接 KL 会惩罚很大，可能梯度爆炸
# - K3 estimator 增长较慢，更稳定
```

### **优势2：适合 group-based 优化**
GRPO 的核心是 **group-wise comparison**：
```python
# 每组有多个样本，计算相对优势
# KL 惩罚需要与优势估计尺度匹配

# 直接 KL 的尺度可能变化很大
# K3 estimator 的输出更可控，大致在 [0, δ_max²/2] 范围内
```

### **优势3：计算效率**
```python
# K3 estimator 只需要一次 exp 运算
kl = exp(delta) - delta - 1

# 稳定版直接计算需要更多操作
ratio = exp(log_ratio)  # 可能溢出，需要 clamp
kl = ratio * log_ratio  # 可能 NaN
```

## **6. 实验验证**

根据原始论文和后续研究：

1. **PPO 论文**：发现直接 KL 惩罚导致训练不稳定
2. **TRPO 论文**：使用约束而不是惩罚，因为 KL 惩罚难调
3. **GRPO 论文**：K3 estimator 在 group-based 设置中表现更好

## **7. 完整代码示例对比**

```python
import torch
import torch.nn.functional as F

def compare_kl_methods():
    batch_size = 4
    vocab_size = 1000
    
    # 模拟 logits 和 log probs
    current_logits = torch.randn(batch_size, vocab_size)
    ref_logits = torch.randn(batch_size, vocab_size)
    
    # 计算 log probs
    current_logps = F.log_softmax(current_logits, dim=-1)
    ref_logps = F.log_softmax(ref_logits, dim=-1)
    
    # 方法1：直接 KL（不稳定）
    ratio = torch.exp(current_logps - ref_logps)
    kl_direct = ratio * (current_logps - ref_logps)
    
    # 方法2：K3 estimator
    delta = ref_logps - current_logps
    kl_k3 = torch.exp(delta) - delta - 1.0
    
    # 方法3：PyTorch 内置（稳定实现）
    kl_pytorch = F.kl_div(
        ref_logps, current_logps, 
        reduction='none', log_target=True
    )
    
    print(f"直接KL均值: {kl_direct.mean().item():.4f}")
    print(f"K3 estimator均值: {kl_k3.mean().item():.4f}")
    print(f"PyTorch KL均值: {kl_pytorch.mean().item():.4f}")
    
    # 检查数值稳定性
    print(f"\n直接KL是否有inf: {torch.isinf(kl_direct).any()}")
    print(f"K3是否有inf: {torch.isinf(kl_k3).any()}")
```

## **8. 总结**

**GRPO 使用 K3 estimator 而不是直接计算 KL 的主要原因**：

1. **数值稳定性**：避免 `exp(current_logps - ref_logps)` 可能导致的溢出
2. **梯度平滑性**：K3 estimator 在 π_θ ≈ π_ref 时梯度趋近于零
3. **计算一致性**：与重要性采样和 PPO clip 机制更匹配
4. **理论保证**：是 KL 散度的上界，保证策略不会偏离参考模型太多
5. **实践效果**：在实际 RLHF 训练中更稳定，超参数更易调节

**简单记忆**：K3 estimator 就像给 KL 散度加了一个 **"安全气囊"**，防止训练过程中出现数值爆炸，同时保持理论上的约束效果。