import math
import torch
from torch import nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F

class MyMultiHeadAttention(nn.Module):
	def __init__(self, 
			  embed_dim, 
			  num_heads,
			  dropout=0.0, 
			  bias=True, 
			  add_bias_kv=False,
			  add_zero_attn=False, 
			  kdim=None, 
			  vdim=None, 
			  batch_first=True, 
			  device=None, 
			  dtype=None):
		super().__init__()
		self.embed_dim = embed_dim
		self.kdim = kdim if kdim else embed_dim
		self.vdim = vdim if vdim else embed_dim
		self.num_head = num_heads
		
		self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.k_proj = nn.Linear(kdim, embed_dim, bias=bias)
		self.v_proj = nn.Linear(vdim, embed_dim, bias=bias)
		self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.dropout = nn.Dropout(dropout)

	def forward(self, query, key, value, key_padding_mask, attn_mask, is_casual):
		batch_size, seq_len, _ = query.shape
		head_dim = self.embed_dim // self.num_head
		if self.embed_dim % self.num_head != 0:
			raise ValueError(f"embed_dim {self.embed_dim} should be divisible by num_heads {self.num_head}")

		Q = self.q_proj(query)
		K = self.k_proj(key)
		V = self.v_proj(value)
		
		Q = Q.view(batch_size, seq_len, self.num_head, head_dim).transpose(1, 2) # (b, h, s, d/h)
		K = K.view(batch_size, seq_len, self.num_head, head_dim).transpose(1, 2) # (b, h, s, d/h)
		V = V.view(batch_size, seq_len, self.num_head, head_dim).transpose(1, 2) # (b, h, s, d/h)

		atten = torch.matmul(Q, K.transpose(-2, -1))  # (b, h, s, s)
		atten /= math.sqrt(head_dim)

		if key_padding_mask is not None and key_padding_mask.dtype == torch.bool:
			key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, s)
			atten = atten.masked_fill(key_padding_mask, float('-inf'))
		if attn_mask is not None and attn_mask.dtype == torch.bool:
			atten = atten.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
		if is_casual and self.training:
			casual_mask = torch.triu(torch.ones(seq_len, seq_len, device=atten.device), diagonal=1).bool()
			atten = atten.masked_fill(casual_mask, float('-inf'))

		atten = F.softmax(atten, dim=-1)
		atten = self.dropout(atten)

		x_att = torch.matmul(atten, V)  # (b, h, s, d/h)
		x_att = x_att.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim) # (b, s, d)

		x_att = self.out_proj(x_att)
		x_att = self.dropout(x_att)
		return x_att
