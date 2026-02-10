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

    """
    输出结果：这三者之间的差别有点大

    直接KL均值: 5.0214
    K3 estimator均值: 1.7834
    PyTorch KL均值: 0.0010

    直接KL是否有inf: False
    K3是否有inf: False
    """


if __name__ == "__main__":
    compare_kl_methods()