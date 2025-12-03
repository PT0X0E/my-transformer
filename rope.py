import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # 计算频率倒数
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        print("inv_freq:", inv_freq)
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算正弦余弦缓存
        self.max_seq_len = max_seq_len
        self._compute_cos_sin_cache(max_seq_len)
    
    def _compute_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # outer product
        print("freqs:", freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        print("emb:", emb)
        self.cos_cached = emb.cos()[None, :, None, :]  # [1, seq_len, 1, dim]
        self.sin_cached = emb.sin()[None, :, None, :]
        print("emb.cos():", emb.cos())
        print("emb.sin():", emb.sin())
        print("cos_cached:", self.cos_cached)
        print("sin_cached:", self.sin_cached)
    
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
        cos = self.cos_cached[:, :seq_len, :, :x.shape[-1]].to(x.device)
        sin = self.sin_cached[:, :seq_len, :, :x.shape[-1]].to(x.device)
        
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
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    K = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    print("原始Query和Key（无位置信息）:")
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    
    # 应用RoPE
    Q_rotated = rope.apply_rotary_emb(Q.transpose(1, 2)).transpose(1, 2)
    K_rotated = rope.apply_rotary_emb(K.transpose(1, 2)).transpose(1, 2)

    print("\n应用RoPE后的Query和Key（包含位置信息）:")
    print(f"Q_rotated: {Q_rotated.shape}")
    print(f"K_rotated: {K_rotated.shape}")
    exit()

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

def visualize_rope_effect():
    """可视化RoPE的旋转效果"""
    # 创建一个简单的2D向量来演示旋转
    vec = torch.tensor([1.0, 0.0])  # 初始向量指向x轴正方向
    
    # 初始化RoPE（维度为2）
    rope_2d = RotaryPositionEmbedding(dim=2, base=10000)
    
    # 测试不同位置的旋转
    positions = [0, 1, 2, 3]
    rotated_vectors = []
    
    for pos in positions:
        # 扩展向量维度以匹配RoPE输入格式
        vec_expanded = vec.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]
        vec_rotated = rope_2d.apply_rotary_emb(vec_expanded, seq_len=pos+1)
        rotated_vectors.append(vec_rotated[0, 0, 0].detach().numpy())
    
    # 绘制旋转效果
    plt.figure(figsize=(12, 4))
    
    # 子图1：向量旋转可视化
    plt.subplot(1, 2, 1)
    for i, (pos, rv) in enumerate(zip(positions, rotated_vectors)):
        plt.arrow(0, 0, rv[0], rv[1], head_width=0.05, head_length=0.1, 
                 length_includes_head=True, color=f'C{i}', label=f'位置 {pos}')
        plt.text(rv[0]*1.1, rv[1]*1.1, f'pos={pos}', fontsize=8)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('RoPE旋转效果可视化')
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.legend()
    plt.axis('equal')

    # 子图2：相对位置依赖关系
    plt.subplot(1, 2, 2)
    
    # 模拟不同相对位置的点积衰减
    relative_positions = range(0, 20)
    dot_products = []
    
    # 创建两个相同的向量
    vec1 = torch.randn(64)  # 模拟64维向量
    vec2 = vec1.clone()
    
    rope_high_dim = RotaryPositionEmbedding(dim=64)
    
    for rel_pos in relative_positions:
        # 应用不同位置的旋转
        vec1_expanded = vec1.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        vec2_expanded = vec2.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        vec1_rotated = rope_high_dim.apply_rotary_emb(vec1_expanded, seq_len=rel_pos+1)
        vec2_rotated = rope_high_dim.apply_rotary_emb(vec2_expanded, seq_len=1)  # 固定位置0
        
        dot_product = torch.dot(vec1_rotated[0,0,0], vec2_rotated[0,0,0])
        dot_products.append(dot_product.item())
    
    plt.plot(relative_positions, dot_products, 'o-', linewidth=2, markersize=4)
    plt.xlabel('相对位置距离')
    plt.ylabel('点积相似度')
    plt.title('RoPE: 相对位置距离 vs 注意力分数')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_rope_properties():
    """测试RoPE的关键数学性质"""
    print("=== 测试RoPE的数学性质 ===")
    
    dim = 8
    rope = RotaryPositionEmbedding(dim=dim)
    
    # 测试1：相对位置依赖性
    print("\n1. 测试相对位置依赖性:")
    
    # 创建相同的Q和K向量
    q = torch.randn(1, 1, 1, dim)
    k = q.clone()
    
    # 位置m的Q和位置n的K
    m, n = 3, 1
    q_rotated_m = rope.apply_rotary_emb(q, seq_len=m+1)
    k_rotated_n = rope.apply_rotary_emb(k, seq_len=n+1)
    
    # 计算点积（模拟注意力分数）
    dot_product = torch.sum(q_rotated_m * k_rotated_n, dim=-1)
    print(f"位置{m}的Q · 位置{n}的K = {dot_product.item():.4f}")
    
    # 测试2：交换位置
    q_rotated_n = rope.apply_rotary_emb(q, seq_len=n+1) 
    k_rotated_m = rope.apply_rotary_emb(k, seq_len=m+1)
    
    dot_product_symmetric = torch.sum(q_rotated_n * k_rotated_m, dim=-1)
    print(f"位置{n}的Q · 位置{m}的K = {dot_product_symmetric.item():.4f}")
    
    # 测试3：验证相对位置差的一致性
    relative_pos = m - n
    q_rotated_relative = rope.apply_rotary_emb(q, seq_len=relative_pos+1)
    k_rotated_0 = rope.apply_rotary_emb(k, seq_len=1)  # 位置0
    
    dot_product_relative = torch.sum(q_rotated_relative * k_rotated_0, dim=-1)
    print(f"相对位置{relative_pos}的点积 = {dot_product_relative.item():.4f}")
    
    print(f"\n验证: 三个点积应该相等（或非常接近）")
    print(f"差值1: {abs(dot_product.item() - dot_product_symmetric.item()):.6f}")
    print(f"差值2: {abs(dot_product.item() - dot_product_relative.item()):.6f}")

if __name__ == "__main__":
    # 运行演示
    Q, K, Q_rot, K_rot, scores_orig, scores_rope = demonstrate_rope_attention()
    
    # 测试数学性质
    test_rope_properties()
    
    # 可视化效果
    visualize_rope_effect()