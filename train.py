import html
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from countdown_task import CountdownTasksDataset, reward_function
from data_types import Episode
from grpo import rollout, update_policy
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer


def evaluate(model, tokenizer, device, dtype, config):
    test_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="test",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device)
    # We reduce the batch size by half as we want to
    # generate twice as long trajectories.
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=config["training"]["batch_size"] // 2,
        drop_last=False,
    )
    success = []
    for batch in dataloader:
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"] * 2,
            num_answer_per_question=1,
            reward_func=reward_function,
            device=device,
            dtype=dtype,
        )
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])
    return np.mean(success)


def main(config_path: str):
    # 解释yaml文件
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])
    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    train_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="train",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device) # random seed, 是一个用于控制随机数生成的类, 控制随机种子：确保随机操作的可重复性,设备管理：让随机数生成在特定设备上.
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()

    optimizer: MemoryEfficientAdamW = MemoryEfficientAdamW(
        params=model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )

    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(train_dataloader, start=1):
        # 利用模型 + 当前batch的question生成answer
        episodes: List[Episode] = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
            reward_func=reward_function,
            device=device,
            dtype=dtype,
        )
        # 只要最后完整的episode，而不要强制结束的episodes
        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]

        policy_metrics = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=config["training"]["micro_batch_size"],
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=config["training"]["max_grad_norm"],
            device=device,
            dtype=dtype,
        )

        """
        GPU 的异步执行特性
        问题：GPU 操作是异步的
        import torch
        import time

        # GPU 操作是异步的
        start = time.time()
        x = torch.randn(10000, 10000).cuda()
        y = torch.randn(10000, 10000).cuda()
        z = x @ y  # 矩阵乘法（在GPU上异步执行）

        print(f"代码执行时间: {time.time() - start:.4f}s")  # 时间很短
        # 但此时GPU可能还在计算！
        解决方案：使用 synchronize()
        start = time.time()
        x = torch.randn(10000, 10000).cuda()
        y = torch.randn(10000, 10000).cuda()
        z = x @ y

        torch.cuda.synchronize()  # 等待GPU完成计算
        print(f"实际计算时间: {time.time() - start:.4f}s")  # 反映真实计算时间
        """
        torch.cuda.synchronize() # torch.cuda.synchronize() 的作用是强制等待 CUDA 设备完成所有先前提交的任务。

        end_time = time.time()
        duration = end_time - start_time # 每次一个batch的rollout+train的时间
        start_time = end_time

        # compute and log important metrics
        reward = [episode.reward for episode in episodes]
        formatted_reward = [ episode.reward_info["format_reward"] for episode in episodes ]
        answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
        num_finished_episodes = sum(episode.is_finished for episode in episodes)
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = policy_metrics["grad_norm"]
        entropy = policy_metrics["entropy"]
        lr = optimizer.param_groups[0]["lr"]
        loss = policy_metrics["loss"]
        mean_response_len = np.mean( [len(episode.generated_token_ids) for episode in episodes])

        print(f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
            f"train success_rate: {success_rate:.2f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"num_finished_episodes: {num_finished_episodes}, "
            f"mean_response_len: {mean_response_len:.2f}, "
            f"entropy: {entropy:.2f}")
        if step % config["training"]["eval_interval"] == 0:
            eval_success_rate = evaluate(model, tokenizer, device, dtype, config)
            print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
            tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)

        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("mean_reward", mean_reward, step)
        tb_writer.add_scalar("std_reward", std_reward, step) # reward的标准差, 监测方差，有助于了解reward的分散程度
        tb_writer.add_scalar("success_rate/train", success_rate, step)
        tb_writer.add_scalar("format_reward", format_reward, step)
        tb_writer.add_scalar("grad_norm", grad_norm, step)
        tb_writer.add_scalar("duration", duration, step)
        tb_writer.add_scalar("num_finished_episodes", num_finished_episodes, step)
        tb_writer.add_scalar("learning_rate", lr, step)
        tb_writer.add_scalar("mean_response_len", mean_response_len, step)
        tb_writer.add_scalar("entropy", entropy, step) # 模型输出的熵度，有助于了解模型的不确定性， 一般模型越来越置信

        # episodes: [batch_size*num_answer_per_question]
        for i, episode in enumerate(episodes):
            # TensorBoard treats text as markdown.
            text = html.escape(episode.text) # html.escape() 函数将字符串中的特殊字符转换为 HTML 实体，以防止 XSS 攻击。
            # 以question_index进行聚合更好，在tensorboard中更宜于查看, 可以看到该问题在每个iter中生成的内容
            """
            我们通过文本形式展示的文本发现，grpo如果前面没有sft, 即使经过2000多次迭代，其think内容也没有任何有效思考内容，所以需要保证grpo前有sft
            """
            tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        # save checkpoint
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            # 没有保存 optimizer 的状态。当前代码只保存了模型的参数权重
            torch.save(model.state_dict(), output_file) # 保存state_dict()
            print(f"Saved checkpoint to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    print(f"{torch.__version__=}")
    main(args.config)


"""
# 完整的检查点应该包含这些内容
checkpoint = {
    'step': step,                          # 当前训练步数
    'model_state_dict': model.state_dict(), # 模型参数
    'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,  # 学习率调度器
    'loss': current_loss,                  # 当前损失
    'config': config,                      # 训练配置
    'timestamp': datetime.now().isoformat() # 时间戳
}

torch.save(checkpoint, output_file)

修复后的完整保存代码
python
# save checkpoint
if step % config["training"]["ckpt_save_interval"] == 0:
    output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
    
    # 创建完整的检查点
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': current_loss  # 如果有当前损失值的话
    }
    
    # 如果有学习率调度器，也保存其状态
    if 'scheduler' in locals() or 'scheduler' in globals():
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, output_file)
    print(f"Saved checkpoint to {output_file}")

对应的加载代码
python
def load_checkpoint(model, optimizer, checkpoint_path, device='cuda'):
    # 加载完整的检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载学习率调度器状态（如果存在）
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 恢复训练步数
    start_step = checkpoint['step'] + 1  # 从下一步开始
    
    print(f"Loaded checkpoint from step {checkpoint['step']}")
    return start_step

# 使用示例
start_step = load_checkpoint(model, optimizer, "ckpt_010000.pt")

1. 同时保存最新和最佳检查点
python
# 保存最新检查点
latest_checkpoint = {
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': current_loss,
    'config': config
}
torch.save(latest_checkpoint, "latest.pt")

# 如果是最佳表现，额外保存
if current_loss < best_loss:
    best_loss = current_loss
    torch.save(latest_checkpoint, "best.pt")

2. 定期清理旧检查点
python
def cleanup_old_checkpoints(ckpt_dir, keep_last_n=5):
    #保留最近5个检查点，删除旧的
    checkpoints = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if len(checkpoints) > keep_last_n:
        for old_ckpt in checkpoints[:-keep_last_n]:
            old_ckpt.unlink()

-------------------------------------
查看 state_dict()
class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)  # 卷积层
        self.bn1 = nn.BatchNorm2d(16)                 # 批归一化层
        self.fc = nn.Linear(16 * 26 * 26, 10)         # 全连接层
        self.register_buffer('running_mean', torch.zeros(1))  # 注册缓冲区
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ComplexModel()
state_dict = model.state_dict()

print("复杂模型的 state_dict:")
for key, value in state_dict.items():
    print(f"{key}: shape {value.shape}")

python
state_dict = model.state_dict()
print("State dict keys:")
for key in state_dict.keys():
    print(f"  {key}")

print("\nState dict 内容:")
for key, value in state_dict.items():
    print(f"{key}: shape {value.shape}, dtype {value.dtype}")
输出结果：

State dict keys:
  linear1.weight
  linear1.bias
  linear2.weight
  linear2.bias

State dict 内容:
linear1.weight: shape torch.Size([5, 10]), dtype torch.float32
linear1.bias: shape torch.Size([5]), dtype torch.float32
linear2.weight: shape torch.Size([2, 5]), dtype torch.float32
linear2.bias: shape torch.Size([2]), dtype torch.float32
"""