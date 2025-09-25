import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import torch
import torch.nn.functional as F
from torch import nn

"""
为何要重新实现一遍呢？不用transformers的呢？

因为作者的目标就是： No transformers and vLLM dependencies!
即不依赖于transformers和vLLM的依赖，所以需要重新手动实现

很好的学习代码
"""
@dataclass
class Qwen2Config:
    attention_dropout: float = 0.0 # 不启用dropout
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 32768
    max_window_layers: int = 70
    model_type: str = "qwen2" # qwen2.5也可以跑
    num_attention_heads: int = 16 # 16个head并行
    num_hidden_layers: int = 36 # 36个transformer layer
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    sliding_window: int = 32768
    tie_word_embeddings: bool = True # 是否输入与输出共享embedding, 即lm head与embedding共享参数
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936
    use_gradient_checkpointing: bool = True # 是否启用梯度检查, 以减小activation的占用内存,  但会增加计算量


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # NOTE:并没有去中心化, 此处假定mean(x)为0
        # x: (batch_size, seq_len, hidden_dim)
        # x/std(x) = x / sqrt(mean(x^2)) + eps
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = self._norm(x).type_as(x)
        # y = alpha * x, 注意此处没有偏移beta
        x = self.weight * x.to(input_dtype)
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x: (batch_size, seq_len, n_heads, head_dim)
    half_head_dim = x.shape[-1]//2
    x1 = x[..., : half_head_dim]
    x2 = x[..., half_head_dim:]
    return torch.cat((-x2, x1), dim=-1)

# 对q,k分别进行旋转
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    # q: (batch_size, seq_len, n_heads, head_dim)
    # k: (batch_size, seq_len, n_kv_heads, head_dim)
    # cos: [batch=1, seq_len, head_dim]
    # unsequeeze => [batch=1, seq_len, n_heads=1, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    """
    对两两元素组成的向量：(qi,q(i+ dim//2)) 进行旋转x度, x为旋转角度:
    (qi, q(i+ dim//2))* [[ cos(x), sin(x) 
                        [ -sin(x), cos(x)  ]]
    = (qi * cos(x) - q(i+ dim//2) * sin(x), qi * sin(x) + q(i+ dim//2) * cos(x))
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


"""
torch.nn.functional.scaled_dot_product_attention()
scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
is_causal=False, scale=None, enable_gqa=False) -> Tensor:

Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed, and applying dropout if a probability greater than 0.0 is specified. The optional scale argument can only be specified as a keyword argument.

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

There are currently three supported implementations of scaled dot product attention:

------------------------------------
FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
Memory-Efficient Attention
A PyTorch implementation defined in C++ matching the above formulation
------------------------------------

The function may call optimized kernels for improved performance when using the CUDA backend. For all other backends, the PyTorch implementation will be used.

All implementations are enabled by default. 
Scaled dot product attention attempts to automatically select the most optimal implementation based on the inputs. 
In order to provide more fine-grained control over what implementation is used,
the following functions are provided for enabling and disabling implementations. 
The context manager is the preferred mechanism:
"""
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    # query: (batch_size, n_heads, seq_len, head_dim)
    # key/value: (batch_size, n_kv_heads, seq_len, head_dim)
    query_len, key_len = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    # attn_bias: (query_len, key_len)
    attn_bias = torch.zeros(query_len, key_len, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(query_len, key_len, dtype=torch.bool).tril(diagonal=0) # 下三角全1的矩阵
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf")) # 0的地方全设为-inf
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # 0的地方全设为-inf
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # 在GQA中，key/value需要复制n_rep次
        # query: (batch_size, n_heads, seq_len, head_dim) 
        # key/value: (batch_size, n_kv_heads, seq_len, head_dim)
        # => 
        # key/value: (batch_size, n_heads, seq_len, head_dim)
        """
        # repeat_interleave 与 repeat 的区别
        tensor = torch.tensor([[1, 2], [3, 4]])

        print("原始张量:")
        print(tensor)

        key在维度0重复2次, 注意是元素级的重复, 而不是块级重复
        print(tensor.repeat_interleave(repeats=2, dim=0))  # 元素级重复
        # [[1, 2],
        #  [1, 2], 
        #  [3, 4],
        #  [3, 4]]

        print(tensor.repeat(2, 1))  # 块级重复, 第0维度重复2次, 第1维度重复1次
        # [[1, 2],
        #  [3, 4],
        #  [1, 2],
        #  [3, 4]]
        """
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    # query: (batch_size, n_heads, seq_len, head_dim) 
    # key/value: (batch_size, n_heads, seq_len, head_dim)
    # attn_weight: (batch_size, n_heads, seq_len, seq_len)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    # attn_weight: (batch_size, n_heads, seq_len, seq_len)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True) # dropout_p=0.0时不启用dropout
    # value: (batch_size, n_heads, seq_len, head_dim)
    attn_value = attn_weight @ value
    # attn_value: (batch_size, n_heads, seq_len, head_dim) 
    return attn_value

class Attention(nn.Module):
    def __init__(self, args: Qwen2Config):
        super().__init__()
        self.n_kv_heads = (args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads)
        self.n_heads = args.num_attention_heads
        # self.n_rep = args.num_attention_heads // args.num_key_value_heads, kv需要复制的次数
        # 16/2=8, 8个head共享1个kv, 即group query attention(GQA)
        # 若为MHA, 则n_rep=1, 若为GQA, 则n_rep=8
        self.n_rep = self.n_heads // self.n_kv_heads
        # 每个head的维度， number of attention heads, 也即多少个head一起并行推理
        self.head_dim = args.hidden_size // args.num_attention_heads  # 2048/16=128

        # in=2048, out=16*128=2048
        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * self.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=False,
        )
        self.args = args

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Initialize key and value cache for inference."""
        # device:一般在gpu上分配内存, NOTE:这里kv cache是在attention层中进行的, 而且是固定大小分配，并非动态分配
        # cache shape: (batch_size, seq_len, num_heads, head_dim)
        cache_shape = (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
        cache_k = torch.zeros(cache_shape, dtype=dtype, device=device)
        cache_v = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.register_buffer("cache_k", cache_k, persistent=False)
        self.register_buffer("cache_v", cache_v, persistent=False)

    def del_kv_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[Union[int, torch.Tensor]] = None,
    ):
        batch_size, seq_len, hidden_dim = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        cos, sin = pos_embed
        # xq: (batch_size, seq_len, n_heads, head_dim)
        # xk: (batch_size, seq_len, n_kv_heads, head_dim)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, unsqueeze_dim=2)
        # 推理时start_pos不为None，训练时为None
        if start_pos is not None:
            # inference mode
            # NOTE:在正常的推理过程中，seq_len一般为1
            end_pos = start_pos + seq_len
            # 先将xk, xv 复制到cache中, 然后使用kv cache中的全部xk, xv进行attention
            self.cache_k[:batch_size, start_pos:end_pos, :, :] = xk
            self.cache_v[:batch_size, start_pos:end_pos, :, :] = xv
            #output = torch.nn.functional.scaled_dot_product_attention(  # 启动pytorch的scaled_dot_product_attention, 有gpu的话，会调用flash attention
            # output: (batch_size, n_heads, seq_len, head_dim) 
            # => output: (batch_size, seq_len, n_heads, head_dim)
            output = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=self.cache_k[:batch_size, :end_pos].transpose(1, 2),
                value=self.cache_v[:batch_size, :end_pos].transpose(1, 2),
                # 如果只有一个token,没必要mask
                is_causal=True if seq_len > 1 else False,
                enable_gqa=True,
            ).transpose(1, 2)
        else:
            # training mode
            #output = torch.nn.functional.scaled_dot_product_attention(
            output = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                is_causal=True,
                enable_gqa=True, # grouped-query attention
            ).transpose(1, 2)
        # output: (batch_size, seq_len, n_heads, head_dim)
        # => output: (batch_size, seq_len, hidden_dim=n_heads * head_dim)
        output = output.reshape(batch_size, seq_len, -1)
        # output: (batch_size, seq_len, hidden_dim)
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int, # 2048
        intermediate_size: int, # 2048* 5.375 = 11008  
    ):
        super().__init__()
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_dim)
        # silu = x*sigmoid(x), Sigmoid Linear Unit (SiLU)
        # y = down_proj(silu(gate_proj(x)) * up_proj(x))
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: Qwen2Config):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.dim = args.hidden_size
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.self_attn = Attention(args)
        self.mlp = FeedForward(dim=args.hidden_size, intermediate_size=args.intermediate_size,)
        self.layer_id = layer_id
        # 注意：名字不要修改，否则与pretrained模型文件中的名字不一致，导致无法加载模型
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[Union[int, torch.Tensor]] = None,
    ):
        # prev rms norm
        # x: (batch_size, seq_len, hidden_dim)
        h = x + self.self_attn(self.input_layernorm(x), pos_embed, start_pos=start_pos)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device: torch.device):
        super().__init__()
        self.config = config
        base = config.rope_theta # 100w
        dim = config.hidden_size // config.num_attention_heads # head_dim=128
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            # inv_freq = base^(-2i/head_dim), shape: [head_dim//2]
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, pos):
        # inv_freq: [head_dim//2] -> [batch_size, head_dim//2, 1]
        inv_freq = self.inv_freq[None, :, None].float().expand(pos.shape[0], -1, 1)
        # pos: [batch_size=1, seq_len]
        # => [batch_size=1, 1, seq_len]
        pos = pos[:, None, :].float() # 中间插入一维
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            # x: (batch_size, seq_len, hidden_dim)
            # inv_freq: [batch_size, head_dim//2, 1]
            # pos: [batch_size=1, 1, seq_len] 
            # freqs: [batch_size, head_dim//2, seq_len]
            #  transpose => [batch_size, seq_len, head_dim//2]
            freqs = (inv_freq.float().to(x.device) @ pos.float()).transpose(1, 2)
            # emb: [batch_size, seq_len, head_dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # sin/cos: [batch_size, seq_len, head_dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Transformer(nn.Module):
    def __init__(self, params: Qwen2Config, device: torch.device):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.num_hidden_layers
        self.use_gradient_checkpointing = params.use_gradient_checkpointing

        self.embed_tokens = torch.nn.Embedding(params.vocab_size, params.hidden_size)
        with torch.device(device):
            self.rotary_emb = Qwen2RotaryEmbedding(config=params, device=device)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_hidden_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)
        if not params.tie_word_embeddings:
            self.lm_head = nn.Linear(params.hidden_size, params.vocab_size, bias=False)

    def lm_head_proj(self, x):
        if self.params.tie_word_embeddings:
            return x @ self.embed_tokens.weight.T
        else:
            return self.lm_head(x)

    def forward(self, token_ids: torch.Tensor):
        _bsz, seqlen = token_ids.shape
        h = self.embed_tokens(token_ids)
        # pos: [seq_len]
        pos = torch.arange(0, seqlen, device=token_ids.device, dtype=torch.int32)
        # h: (batch_size, seq_len, hidden_dim)
        # pos: [seq_len] -> [batch_size, seq_len]
        # pos_emb: (batch_size, seq_len, hidden_dim)
        pos_emb = self.rotary_emb(h, pos[None, :])

        if self.use_gradient_checkpointing:
            pipe_funcs = []
            for current_layer in self.layers:
                pipe_funcs.append(lambda x, layer=current_layer:  # layer的值固定为current_layer
                                            layer(x, pos_emb))
            pipe_funcs.append(self.norm.forward)
            pipe_funcs.append(self.lm_head_proj)
            return torch.utils.checkpoint.checkpoint_sequential(functions=pipe_funcs, segments=len(pipe_funcs), input=h, use_reentrant=False)
        else:
            # 不使用梯度检查点
            x = h
            for current_layer in self.layers:
                x = current_layer(x, pos_emb)
            # x: (batch_size, seq_len, hidden_dim)
            # output: (batch_size, seq_len, vocab_size)
            output = self.lm_head_proj(self.norm(x))
            return output

    def inference(self, tokens: torch.Tensor, start_pos: Union[int, torch.Tensor]):
        batch_size, seqlen = tokens.shape
        del batch_size
        h = self.embed_tokens(tokens)
        # pos:[batch_size=1, seq_len]
        pos = torch.arange(0, seqlen, device=tokens.device, dtype=torch.int32)[None, :]
        if isinstance(start_pos, torch.Tensor):
            pos = pos + start_pos[:, None]
        else:  # int
            pos.add_(start_pos)
        # h: (batch_size, seq_len, hidden_dim)
        # pos_emb: Tuple((batch_size, seq_len, hidden_dim), (batch_size, seq_len, hidden_dim)), 内含cos, sin
        pos_emb = self.rotary_emb.forward(h, pos)

        for layer in self.layers:
            h = layer(h, pos_emb, start_pos=start_pos)

        # only need the hidden state of the last token
        # to predict the next token
        # h: (batch_size, seq_len, hidden_dim), 只取最后一个token的hidden state
        # => h:[batch_size, seq_len=1, dim]
        h = h[:, -1:, :]
        h = self.norm(h)

        # output: [batch_size, seq_len=1, vocab_size]
        output = self.lm_head_proj(h)
        return output

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        # 分别初始化各层的kv cache
        for layer in self.layers:
            layer.self_attn.init_kv_cache(max_batch_size, max_seq_len, dtype=dtype, device=device)

    def del_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.del_kv_cache()

    @classmethod
    def from_pretrained(cls, ckpt_path, device: torch.device):
        config_file = Path(ckpt_path) / "config.json"
        with open(config_file, "r") as f:
            config = json.load(f)

        args = Qwen2Config(
            attention_dropout=config["attention_dropout"],
            bos_token_id=config["bos_token_id"],
            eos_token_id=config["eos_token_id"],
            hidden_act=config["hidden_act"],
            hidden_size=config["hidden_size"],
            initializer_range=config["initializer_range"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
            max_window_layers=config["max_window_layers"],
            model_type=config["model_type"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            vocab_size=config["vocab_size"],
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=config["rope_theta"],
            sliding_window=config["sliding_window"],
            use_sliding_window=config["use_sliding_window"],
            use_cache=config["use_cache"],
            tie_word_embeddings=config["tie_word_embeddings"],
            torch_dtype=config["torch_dtype"],
        )
        with torch.device("meta"):
            model = cls(params=args, device=device)

        import safetensors.torch

        model_weight_files = sorted(Path(ckpt_path).glob("model*.safetensors"))
        param_name_to_weights = {}
        for file in model_weight_files:
            param_name_to_weights.update(safetensors.torch.load_file(file, device="cpu"))
        # remove "model." prefix from keys
        param_name_to_weights = {k.replace("model.", ""): v for k, v in param_name_to_weights.items()}
        print("model weight:")
        param_name_to_weight_shape = {key: value.shape for key, value in param_name_to_weights.items()}
        print(json.dumps(param_name_to_weight_shape, indent=1))
        # NOTE:类中所有的变理名必须和safetensors中的参数名一致
        model.load_state_dict(param_name_to_weights, strict=True, assign=True)
        return model.to(device)

def test_transformer():
    conf = Qwen2Config()
    conf.num_hidden_layers=2
    conf.vocab_size=100
    conf.hidden_size=32
    conf.use_gradient_checkpointing=False

    device = torch.device("cpu")
    batch_size= 2
    seq_len = 16
    model =Transformer(params=conf, device=device)
    model.eval()
    model.init_kv_cache(batch_size, 512, device, torch.float32)

    token_ids = torch.randint(0, conf.vocab_size, (batch_size, seq_len), device=device)
    print(f'input ids:{token_ids}')
    result = model.forward(token_ids=token_ids)
    print(result.shape)
    print(f'results:{result}')

    # 模拟测试：验证每个lambda确实捕获了不同的层
    layers = ['layer1', 'layer2', 'layer3']
    functions = []

    for current_layer in layers:
        # 将两个参数的lambda变成只有一个参数的lambda
        #functions.append(lambda x, layer_obj=current_layer: f"{layer_obj} processed {x}")
        functions.append(lambda x, layer_obj=current_layer:  # layer_obj的值为current_layer
                            f"{layer_obj} processed {x}")

    # 测试每个函数
    for i, func in enumerate(functions):
        print(func("input"))  # 应该输出：layerX processed input

def test_load_qwen_2dot5_model():
    device = torch.device("cpu")
    pretrained_model_path = "/home/hkx/data/work/hf_data_and_model/models/Qwen/Qwen2.5-0.5B-Instruct/"
    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()
    print(model)


if __name__ == "__main__":
    test_load_qwen_2dot5_model()
    #test_transformer()