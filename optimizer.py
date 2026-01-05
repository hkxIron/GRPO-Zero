import math
from typing import List

import torch
from torch.optim import AdamW


class MemoryEfficientAdamW(AdamW):
    """
    Memory Efficient AdamW optimizer that keeps parameters and gradients on GPU
    but optimizer states on CPU when enabled.
    When disabled, behaves exactly like standard AdamW.

    pin_memory（锁页内存） 指的是将内存页面锁定在物理RAM中，防止被操作系统交换到磁盘上。

    torch.zeros_like(p.data, device=device, pin_memory=pin_memory, dtype=dtype)
    当 pin_memory=True 时：张量数据会被分配到"锁页内存"中
    当 pin_memory=False 时：张量数据分配到普通内存中

    应该使用 pin_memory=True 的情况：
    # 数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        pin_memory=True,  # 加速数据到GPU的传输
        num_workers=4
    )

    # 优化器状态（如果可能在CPU/GPU间移动）
    optimizer = Adam(model.parameters())
    # 在初始化时使用pin_memory为将来可能的GPU训练做准备
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        pin_memory=True,
        enabled=True,
    ):
        super(MemoryEfficientAdamW, self).__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        self.pin_memory = pin_memory
        self.enabled = enabled

    """
    1. closure（闭包函数） 是一个可调用对象（通常是函数），它封装了计算损失所需的所有操作。在优化器步骤中，closure 负责：
    - 清零梯度
    - 前向传播计算损失
    - 反向传播计算梯度

    # 梯度累积
    accumulation_steps = 4

    def closure(step):
        optimizer.zero_grad()
        loss = model.compute_loss(inputs, targets)
        loss = loss / accumulation_steps  # 归一化损失
        loss.backward()
        return loss

    # 每 accumulation_steps 步才真正更新参数
    if (step + 1) % accumulation_steps == 0:
        optimizer.step(lambda: closure(step))
    else:
        closure(step)  # 只计算梯度，不更新参数

    2. 典型的 param_groups 结构
    param_groups = [
        {
            'params': [parameter1, parameter2, ...],  # 参数列表
            'lr': 0.001,                              # 学习率
            'weight_decay': 0.01,                     # 权重衰减
            'betas': (0.9, 0.999),                    # Adam 的 beta 参数
            # ... 其他优化器特定参数
        },
        {
            'params': [parameter3, parameter4, ...],
            'lr': 0.0001,                            # 不同的学习率
            'weight_decay': 0.0,                     # 不同的权重衰减
            # ... 其他配置
        }
    ]

    例子：
    import torch
    import torch.nn as nn

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(100, 200),
                nn.ReLU(),
                nn.Linear(200, 100)
            )
            self.classifier = nn.Linear(100, 10)
        
        def forward(self, x):
            x = self.backbone(x)
            return self.classifier(x)

    model = MyModel()

    ## 2.1 为不同部分设置不同的学习率
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-4},    # 主干网络：小学习率
        {'params': model.classifier.parameters(), 'lr': 1e-3}   # 分类器：大学习率
    ])

    print("参数组数量:", len(optimizer.param_groups))
    print("第一个组的学习率:", optimizer.param_groups[0]['lr'])
    print("第二个组的学习率:", optimizer.param_groups[1]['lr'])

    # 2.2 冻结 backbone，只训练 classifier
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])

    # 或者更精细的控制
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 0},      # 学习率为0，相当于冻结
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])

    ## 2.3 对权重和偏置使用不同的权重衰减
    weight_params = []
    bias_params = []

    for name, param in model.named_parameters():
        if 'bias' in name:
            bias_params.append(param)
        else:
            weight_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': weight_params, 'weight_decay': 0.01},
        {'params': bias_params, 'weight_decay': 0.0}    # 偏置通常不需要权重衰减
    ])
    """
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        # closure (Callable, optional): A closure that reevaluates the model and returns the loss.
        if not self.enabled:
            # Use the parent AdamW implementation when disabled
            return super(MemoryEfficientAdamW, self).step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # NOTE: param_groups 是一个字典列表，每个字典包含一组参数和对应的优化器配置。它允许对模型的不同部分使用不同的超参数。
        for group in self.param_groups:
            # 每个group中的参数与lr均不一样
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            # 每个p都会有自己的state
            for p in group["params"]:
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                grads.append(p.grad)

                # Initialize state if needed
                state = self.state[p] # 取出当前的参数p的状态字典, 若为空，则创建，由于是使用defaultdict初始化，因此可以取到任意key的值{}
                if len(state) == 0: # 初始化stat_dict
                    state["step"] = 0
                    # Store optimizer states on CPU with pinned memory
                    device = "cpu" # NOTE: 在cpu上存储优化器状态, 并不是在param所在的设备(如gpu)上存优化器状态, 但是会pin_memory, 即锁页内存加速访问,防止被交换到硬盘
                    pin_memory = self.pin_memory
                    # 32位存储优化器状态
                    dtype = torch.float32
                    # NOTE:将优化器状态存储在CPU上，以节省GPU内存
                    state["exp_avg"] = torch.zeros_like(p.data, device=device, pin_memory=pin_memory, dtype=dtype) # NOTE: 一阶动量,一般需要使用高精度fp32来存储
                    state["exp_avg_sq"] = torch.zeros_like(p.data, device=device, pin_memory=pin_memory, dtype=dtype) # NOTE: 二阶动量
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data, device=device, pin_memory=pin_memory, dtype=dtype)

                # Get state values
                exp_avgs.append(state["exp_avg"]) # NOTE: 一阶动量, exp_avg存在CPU内存中
                exp_avg_sqs.append(state["exp_avg_sq"]) # NOTE: 二阶动量, exp_avg_sq存在CPU内存中

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"]) # 最大二阶动量

                state["step"] += 1 # 步数加1
                state_steps.append(state["step"])

            # NOTE:计算好优化器状态后,原地更新训练参数p的值
            # Process all parameters in the group
            self._memory_efficient_update(
                params_with_grad,
                grads,
                exp_avgs, # NOTE: 一阶动量, exp_avg存在CPU内存中
                exp_avg_sqs, # NOTE: 二阶动量
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

        return loss

    def _memory_efficient_update(
        self,
        params:List[torch.Tensor],
        grads:List[torch.Tensor],
        exp_avgs:List[torch.Tensor], # NOTE: 一阶动量, exp_avg存在CPU内存中
        exp_avg_sqs:List[torch.Tensor], # NOTE: 二阶动量，exp_avg_sq存在CPU内存中
        max_exp_avg_sqs,
        state_steps,
        amsgrad,
        beta1,
        beta2,
        lr,
        weight_decay,
        eps,
    ):
        """
        Performs the AdamW parameter update on GPU with CPU-stored optimizer states.
        Uses pinned memory for efficient CPU-to-GPU transfer of optimizer states.
        """
        for i, param in enumerate(params):
            grad = grads[i] # NOTE: 取出params对应的梯度
            param_device = param.device

            # NOTE: 一阶动量, exp_avg存在CPU内存中, 需要移动到参数param所在的GPU中
            #   Access optimizer states - they'll transfer efficiently due to pin_memory, 禁用换页
            #  将m1/v1从cpu移到gpu, exp_avg为gpu上， exp_avgs[i]在cpu上
            exp_avg = exp_avgs[i].to(param_device, non_blocking=True) # 将优化器状态从CPU memory -> GPU memory
            exp_avg_sq = exp_avg_sqs[i].to(param_device, non_blocking=True)

            step = state_steps[i]

            # NOTE: GPU上原地计算, 不产生中间变量, 避免内存的申请与释放
            # Decay the first and second moment running averages
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1) # NOTE: m1 = m1 * beta1 + grad * (1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # NOTE: v1 = v1 * beta2 + grad^2 * (1 - beta2)

            if amsgrad:
                # Access max_exp_avg_sq - transfers efficiently with pin_memory
                max_exp_avg_sq = max_exp_avg_sqs[i].to(param_device, non_blocking=True)
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max for normalizing running avg of gradient
                denom = max_exp_avg_sq.sqrt().add_(eps)
                # Store back to CPU
                max_exp_avg_sqs[i].copy_(max_exp_avg_sq, non_blocking=True)
            else:
                denom = exp_avg_sq.sqrt().add_(eps) # denom = v1^0.5 + eps

            bias_correction1 = 1 - beta1**step  # bias_correction1 = 1 - beta1^step
            bias_correction2 = 1 - beta2**step # bias_correction2 = 1 - beta2^step
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1 # step_size = lr * (1 - beta2^step)^(1/2) / (1 - beta1^step)

            # Apply weight decay directly to the parameter (AdamW)
            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay) # param = param * (1 - lr * weight_decay)

            # NOTE: Update parameters (directly on GPU), 均为原地操作，原因是原地操作可以节省内存，不需要再分配GPU内存
            # NOTE:最重要的操作，原地更新参数p的值
            #  exp_avg即m1
            #  param = param - step_size * m1 / denorm = param - step_size * m1 / (v1 ^ 0.5 + eps)
            param.addcdiv_(exp_avg, denom, value=-step_size) #

            # Store optimizer states back to CPU
            # NOTE: 将优化器状态exp_avgs=m1从GPU memory -> CPU memory, 然后释放GPU memory
            #  将m1/v1从gpu移到cpu, exp_avg为gpu上， exp_avgs[i]在cpu上, 注意：即使不使用cpu/gpu之间的转换，仍需要将optimizer state缓存到cpu中，以减小gpu内存压力
            exp_avgs[i].copy_(src=exp_avg, non_blocking=True)
            exp_avg_sqs[i].copy_(src=exp_avg_sq, non_blocking=True)
