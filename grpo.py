import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List

import numpy as np
import torch

from data_types import Episode, MiniBatch
from qwen2_model import Transformer
from tokenizer import Tokenizer


@torch.no_grad()
def rollout(
    model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_func: Callable,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    # prefix_token_ids: [batch_size, seq_len]
    prefix_token_ids: List[List[int]] = batch.prefix_token_ids
    batch_size = len(batch.prefix) * num_answer_per_question
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len # 最大序列长度
    # 每次rollout时，都会重新分配kv cache
    model.init_kv_cache(
        max_batch_size=batch_size,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )
    # 先预分配一个最大长度的tokens内存, 默认都填padding
    # tokens: [batch_size*num_answer_per_question, max_gen_len + max_prompt_len]
    tokens = torch.full((batch_size, total_len), pad_token_id, dtype=torch.long, device=device)
    # prefix_token_ids: [batch_size, seq_len]
    for k, _prefix_token_ids in enumerate(prefix_token_ids):
        # 每个question生成多个answer
        question_offset = k * num_answer_per_question
        # 填充prefix
        for question_idx in range(num_answer_per_question):
            tokens[question_offset + question_idx, : len(_prefix_token_ids)] = torch.tensor(_prefix_token_ids, dtype=torch.long, device=device)

    # input_text_mask: [batch_size*num_answer_per_question, max_gen_len + max_prompt_len]
    input_text_mask = tokens != pad_token_id # 有效位置
    assert min_prompt_len < total_len
    # is_finished:[batch_size*num_answer_per_question]
    is_finished = torch.zeros((batch_size,), dtype=torch.bool, device=device) # 全为0

    prev_pos = 0
    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        with torch.autocast(device_type=device.type, dtype=dtype):
            # tokens: [batch_size*num_answer_per_question, max_gen_len + max_prompt_len]
            # logits: [batch_size*num_answer_per_question, 1, vocab_size]
            logits = model.inference(tokens=tokens[:, prev_pos:cur_pos], start_pos=prev_pos) # prev_pos:int
        # logits: [batch_size*num_answer_per_question, 1, vocab_size]
        # probs: [batch_size*num_answer_per_question, vocab_size]
        probs = torch.softmax(logits[:, -1], dim=-1)
        # next_token:[batch_size*num_answer_per_question, 1], 采样生成next_token
        next_token = torch.multinomial(input=probs, num_samples=1)
        # next_token:[batch_size*num_answer_per_question]
        next_token = next_token.reshape(-1)
        # 下一个位置用next_token
        # input_text_mask[:,cur_pos]: [batch_size*num_answer_per_question]
        # tokens[:,cur_pos]: [batch_size*num_answer_per_question]
        next_token = torch.where(condition=input_text_mask[:, cur_pos], input=tokens[:, cur_pos], other=next_token)
        # if an rollout is finished, we fill the rest of the tokens with pad_token_id
        # is_finished: [batch_size*num_answer_per_question]
        # next_token: [batch_size*num_answer_per_question]
        next_token = torch.where(condition=is_finished, input=pad_token_id, other=next_token) # finshed的地方补pad_token_id
        tokens[:, cur_pos] = next_token # 更新next_token到当前位置
        if end_token_id is not None:
            # is_end_token: [batch_size*num_answer_per_question]
            is_end_token = next_token == end_token_id # 是否结束符
            is_generated_token = ~input_text_mask[:, cur_pos] # 不是mask掉的token,是有效生成的token
            # 本次生成了eos
            is_finished = is_finished | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        if is_finished.all(): # 如果所有的都finished,则停止推理
            break
    model.del_kv_cache()
    gc.collect() # 回收cpu内存
    torch.cuda.empty_cache() # 回收gpu内存,以减小内存碎片
    is_finished_list = is_finished.tolist()
    # tokens_list: [batch_size*num_answer_per_question, max_gen_len + max_prompt_len], 转成list是便于处理
    tokens_list = tokens.tolist()

    # prepare the output episodes
    episodes = [] # episodes: [batch_size*num_answer_per_question]
    for question_idx in range(batch_size // num_answer_per_question):
        for j in range(num_answer_per_question):
            answer_idx = question_idx * num_answer_per_question + j
            # 只取生成token_id
            # generated_token_ids: [max_gen_len]
            generated_token_ids = tokens_list[answer_idx][len(batch.prefix_token_ids[question_idx]) :]
            # remove padding tokens
            if pad_token_id in generated_token_ids:
                # 只取去掉padding的部分
                generated_token_ids = generated_token_ids[: generated_token_ids.index(pad_token_id)]
            generated_text: str = tokenizer.detokenize(generated_token_ids)
            # 计算当前question的reward
            rewards = reward_func(
                response=generated_text,
                numbers=batch.numbers[question_idx],
                target=batch.target[question_idx],
                end_token=end_token,
            )
            # 每个answer都会生成一个episode, 而
            episode = Episode(
                prefix=batch.prefix[question_idx],
                text=batch.prefix[question_idx] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[question_idx],
                prefix_tokens=batch.prefix_tokens[question_idx],
                generated_token_ids=generated_token_ids, # 这里是有效tokenr id, 不包括padding的token
                is_finished=is_finished_list[answer_idx], # 是否输出了eos
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)
    # clear the output line
    print("\r", end=" " * 100, flush=True) # 光标回到行首，并清空行内字符
    # episodes: [batch_size*num_answer_per_question]
    return episodes

"""
没有使用 \r 的情况：
python
import time

for i in range(5):
    print(f"Progress: {i}/5")
    time.sleep(1)

# 输出：
# Progress: 0/5
# Progress: 1/5
# Progress: 2/5
# Progress: 3/5
# Progress: 4/5
使用 \r 的情况：
python
import time

for i in range(5):
    print(f"\rProgress: {i}/5", end="", flush=True)
    time.sleep(1)
print()  # 最后换行

# 输出（在同一行动态更新）：
# Progress: 4/5 （最终显示）

应用场景
def train_epoch(model, dataloader):
    total_batches = len(dataloader)
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # 训练逻辑...
        loss = model.train_step(inputs, targets)
        
        # 动态更新进度
        print(f"\rBatch: {batch_idx+1}/{total_batches} | Loss: {loss:.4f}", 
              end="", flush=True)
    
    # 清空行并返回结果
    print("\r", end=" " * 100, flush=True)
    print(f"\rEpoch completed! Average loss: {avg_loss:.4f}")
"""

def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """Normalize rewards per group. A group is defined by the prefix."""
    groups = defaultdict(list)
    for episode in episodes:
        # tuple("abc") -> ('a', 'b', 'c')
        # 相同question前缀的被认为是同一个group, 或者用索引更高效吧？
        groups[tuple(episode.prefix)].append(episode)
    output = []
    # 同一组内的reward减均值，除标准差
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            # 更新episode的reward
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # logits: [batch_size, seq_len, vocab_size]
    # probs: [batch_size, seq_len, vocab_size]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    """
    entropy = -sum_{i} {p_i*log(p_i)}
    logsumexp = log(sum(exp(x)))

    prob = softmax(logit) = exp(logit)/sum(exp(logit))

    推导entropy公式：
    entropy = -sum_{i} {p_i*log(p_i)} = -sum_{i} { p_i * log[ exp(logit_i)/sum(exp(logit)) ]}
    = - sum_{i} { p_i * logit_i - p_i * log(sum(exp(logit)))}
    = log(sum(exp(logit))) - sum_{i} { p_i * logit_i }  
     
    entropy = logsumexp - sum_{i} {p_i*log(p_i)} 
    NOTE: 这样变换的好处是，logsumexp是稳定的，而sum(p_i*log(p_i))可能不稳定，因为log(p_i)可能很小，导致乘以p_i后变成0，而logsumexp不会。
    """
    # entropy: [batch_size, seq_len]
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def update_policy(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Update the policy using the GRPO algorithm."""
    episodes = normalize_rewards_per_group(episodes)
    # sort episodes by token length for efficient (micro-)batching, 以减小seq_len不同导致的padding太多
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = 0.0

    for i in range(0, len(episodes), step=micro_batch_size):
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}", # 右对齐,最小宽度为2个字符, 十进制整数
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        # 取一部分episode
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids + episode.generated_token_ids + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        # batch_mask; [micro_batch_size, batch_max_length]
        batch_masks = [
            [0] * len(episode.prefix_token_ids) # prefix不需要计算loss
            + [1] * len(episode.generated_token_ids) # 生成部分要计算loss
            + [0] * (batch_max_length - batch_lengths[i]) # padding的地方不计算loss
            for i, episode in enumerate(batch_episodes)
        ]
        # batch_advantages: [micro_batch_size]
        batch_advantages = [episode.reward for episode in batch_episodes]
        # batch_token_ids: [micro_batch_size, batch_max_length]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        # batch_mask; [micro_batch_size, batch_max_length]
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        # batch_advantages: [micro_batch_size]
        batch_advantages = torch.tensor(batch_advantages, device=device, dtype=torch.float32)

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:] # label右移1位
            # target_masks; [micro_batch_size, batch_max_length-1]
            target_masks = batch_masks[:, 1:] # label右移1位
            # logits: [micro_batch_size, batch_max_length, vocab_size]
            logits = model.forward(input_token_ids).float()

        # logits: [micro_batch_size, batch_max_length-1, vocab_size]
        # target: [micro_batch_size, batch_max_length-1]
        # negative_log_probs: [micro_batch_size, batch_max_length-1]
        negative_log_probs = -torch.nn.functional.cross_entropy(
            input=logits.reshape(-1, logits.size(-1)),
            target=target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            # logits: [micro_batch_size, batch_max_length-1, vocab_size]
            # token_entropy: [micro_batch_size, batch_max_length-1]
            token_entropy = compute_entropy(logits)
            # target_masks; [micro_batch_size, batch_max_length-1]
            # 只计算有效token的entropy，排除了padding的entropy
            entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens

        # negative_log_probs: [micro_batch_size, batch_max_length-1]
        # batch_advantages: [micro_batch_size]
        # obj: [micro_batch_size, batch_max_length-1]
        obj = negative_log_probs * batch_advantages[:, None]
        # per-token objective
        # target_masks; [micro_batch_size, batch_max_length-1]
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()

    # NOTE: 注意：这里移除了deepseek grpo中的refrence model以及 kl penalty
    # update the policy
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }
