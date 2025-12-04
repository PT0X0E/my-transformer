import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # 计算频率倒数
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算正弦余弦缓存
        self.max_seq_len = max_seq_len
        self._compute_cos_sin_cache(max_seq_len)
    
    def _compute_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # outer product, [seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1) # [seq_len, dim]
        self.cos_cached = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
        self.sin_cached = emb.sin()[None, None, :, :]
    
    def rotate_half(self, x):
        """将输入的后半部分取负，实现旋转效果"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_emb(self, x, seq_len=None):
        # x shape: [batch, num_heads, seq_len, head_dim]
        if seq_len is None:
            seq_len = x.shape[2]  # 从输入获取 seq_len
            
        if seq_len > self.max_seq_len:
            self._compute_cos_sin_cache(seq_len)
            self.max_seq_len = seq_len
            
        # 确保维度匹配
        cos = self.cos_cached[:, :, :seq_len, :x.shape[-1]].to(x.device)
        sin = self.sin_cached[:, :, :seq_len, :x.shape[-1]].to(x.device)
        
        # 应用旋转公式: x_rotated = x * cos + rotate_half(x) * sin
        return (x * cos) + (self.rotate_half(x) * sin)

def demonstrate_rope_attention():
    """演示RoPE在注意力机制中的应用"""
    torch.manual_seed(42)
    
    # 模拟设置
    batch_size, seq_len, num_heads, head_dim = 1, 5, 1, 8
    dim = num_heads * head_dim
    
    # 初始化RoPE
    rope = RotaryPositionEmbedding(dim=head_dim)
    
    # 创建模拟的Query和Key（随机初始化）
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print("原始Query和Key（无位置信息）:")
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    
    # 应用RoPE
    Q_rotated = rope.apply_rotary_emb(Q, seq_len)
    K_rotated = rope.apply_rotary_emb(K, seq_len)

    print("\n应用RoPE后的Query和Key（包含位置信息）:")
    print(f"Q_rotated: {Q_rotated.shape}")
    print(f"K_rotated: {K_rotated.shape}")
    
    # 计算注意力分数
    # 原始注意力分数（无位置信息）
    attention_scores_original = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # 带RoPE的注意力分数
    attention_scores_rope = torch.matmul(Q_rotated, K_rotated.transpose(-2, -1)) / math.sqrt(head_dim)
    
    print("\n=== 注意力分数比较 ===")
    print("原始注意力分数（对角线显示自注意力）:")
    print(attention_scores_original[0, 0].detach().numpy())
    
    print("\nRoPE注意力分数（包含位置信息）:")
    print(attention_scores_rope[0, 0].detach().numpy())

    return Q, K, Q_rotated, K_rotated, attention_scores_original, attention_scores_rope


def test_rope_properties():
    """测试RoPE的关键数学性质"""
    print("=== 测试RoPE的数学性质 ===")
    
    dim = 8
    max_pos = 5
    rope = RotaryPositionEmbedding(dim=dim, max_seq_len=max_pos)
    
    # 测试1：相对位置依赖性
    print("\n1. 测试相对位置依赖性:")
    
    # 创建相同的Q和K向量
    q_vec = torch.randn(1, 1, 1, dim)
    k_vec = torch.randn(1, 1, 1, dim)
    
    # 为了获取特定位置的旋转，需要将向量放入序列中
    # 创建一个长度为 max_pos 的序列，每个位置放相同的向量
    q_seq = q_vec.expand(1, 1, max_pos, dim).clone()  # [1, 1, max_pos, dim]
    k_seq = k_vec.expand(1, 1, max_pos, dim).clone()
    
    # 应用 RoPE - 每个位置会得到不同的旋转
    q_rotated_all = rope.apply_rotary_emb(q_seq)  # [1, 1, max_pos, dim]
    k_rotated_all = rope.apply_rotary_emb(k_seq)
    
    # 位置m的Q和位置n的K
    m, n = 3, 1
    q_at_m = q_rotated_all[:, :, m, :]  # 提取位置 m
    k_at_n = k_rotated_all[:, :, n, :]  # 提取位置 n
    
    # 计算点积（模拟注意力分数）
    dot_product = torch.sum(q_at_m * k_at_n, dim=-1)
    print(f"位置{m}的Q · 位置{n}的K = {dot_product.item():.4f}")
    
    # 测试2：交换位置
    q_at_n = q_rotated_all[:, :, n, :]
    k_at_m = k_rotated_all[:, :, m, :]
    
    dot_product_symmetric = torch.sum(q_at_n * k_at_m, dim=-1)
    print(f"位置{n}的Q · 位置{m}的K = {dot_product_symmetric.item():.4f}")
    
    # 测试3：验证相对位置差的一致性
    # RoPE 的核心性质：Q_m · K_n 只依赖于 (m - n)
    # 所以 Q_3 · K_1 应该等于 Q_2 · K_0 (相对距离都是 2)
    m2, n2 = 4, 2  # 另一对相对距离为 2 的位置
    q_at_m2 = q_rotated_all[:, :, m2, :]
    k_at_n2 = k_rotated_all[:, :, n2, :]

    dot_product_same_rel = torch.sum(q_at_m2 * k_at_n2, dim=-1)
    print(f"位置{m2}的Q · 位置{n2}的K = {dot_product_same_rel.item():.4f}")
    
    print(f"\n验证 RoPE 的相对位置性质:")
    print(f"Q_{m} · K_{n} (rel={m-n}) = {dot_product.item():.6f}")
    print(f"Q_{m2} · K_{n2} (rel={m2-n2}) = {dot_product_same_rel.item():.6f}")
    print(f"这两个应该相等: 差值 = {abs(dot_product.item() - dot_product_same_rel.item()):.6f}")


def visualize_rope_effect():
    """可视化RoPE的旋转效果"""
    # 创建一个简单的2D向量来演示旋转
    vec = torch.tensor([1.0, 0.0])  # 初始向量指向x轴正方向
    
    # 初始化RoPE（维度为2）
    rope_2d = RotaryPositionEmbedding(dim=2, base=10000)
    
    # 测试不同位置的旋转
    positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    max_pos = max(positions) + 1
    
    # 创建一个序列，每个位置放相同的向量
    vec_seq = vec.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, max_pos, 2).clone()  # [1, 1, max_pos, 2]
    
    # 一次性应用 RoPE，每个位置会得到不同的旋转
    vec_rotated_all = rope_2d.apply_rotary_emb(vec_seq)  # [1, 1, max_pos, 2]
    
    # 提取各个位置的旋转结果
    rotated_vectors = [vec_rotated_all[0, 0, pos].detach().numpy() for pos in positions]
    
    # 绘制旋转效果
    plt.figure(figsize=(12, 4))
    
    # 子图1：向量旋转可视化
    plt.subplot(1, 2, 1)
    for i, (pos, rv) in enumerate(zip(positions, rotated_vectors)):
        plt.arrow(0, 0, rv[0], rv[1], head_width=0.05, head_length=0.1, 
                 length_includes_head=True, color=f'C{i}', label=f'pos {pos}')
        plt.text(rv[0]*1.1, rv[1]*1.1, f'pos={pos}', fontsize=8)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('RoPE Rotation Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')

    # 子图2：相对位置依赖关系
    plt.subplot(1, 2, 2)
    
    # 模拟不同相对位置的点积衰减
    max_rel_pos = 20
    relative_positions = range(0, max_rel_pos)
    dot_products = []
    
    # 创建两个相同的向量
    vec1 = torch.randn(64)  # 模拟64维向量
    
    rope_high_dim = RotaryPositionEmbedding(dim=64, max_seq_len=max_rel_pos)
    
    # 创建序列，每个位置放相同的向量
    vec_seq = vec1.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, max_rel_pos, 64).clone()
    
    # 一次性应用 RoPE
    vec_rotated_all = rope_high_dim.apply_rotary_emb(vec_seq)  # [1, 1, max_rel_pos, 64]
    
    # 位置 0 的向量作为基准
    vec_at_0 = vec_rotated_all[0, 0, 0]
    
    for rel_pos in relative_positions:
        # 计算位置 rel_pos 与位置 0 的点积
        vec_at_rel = vec_rotated_all[0, 0, rel_pos]
        dot_product = torch.dot(vec_at_rel, vec_at_0)
        dot_products.append(dot_product.item())
    
    plt.plot(relative_positions, dot_products, 'o-', linewidth=2, markersize=4)
    plt.xlabel('Relative Position Distance')
    plt.ylabel('Dot Product Similarity')
    plt.title('RoPE: Relative Position vs Attention Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.savefig("rope_visualization.png")


if __name__ == "__main__":
    # 运行演示
    demonstrate_rope_attention()
    
    # 测试数学性质
    test_rope_properties()
    
    # 可视化效果
    visualize_rope_effect()