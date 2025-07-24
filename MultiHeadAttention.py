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
		Q = self.q_proj(query)
		K = self.k_proj(key)
		V = self.v_proj(value)
		Q = torch.chunk(Q, self.num_head, dim=-1)  # (b, s, d/h) * h
		K = torch.chunk(K, self.num_head, dim=-1)  # (b, s, d/h) * h
		V = torch.chunk(V, self.num_head, dim=-1)  # (b, s, d/h) * h

		#TODO: use torch.view instead of chunk and loop
		x_att = []
		for n in range(self.num_head):
			atten = torch.matmul(Q[n], # (b,s,d/h)
								torch.transpose(K[n], -1, -2) # (b,d/h,s)
			) # (b,s,s)
			atten /= math.sqrt(self.embed_dim / self.num_head)
			if key_padding_mask is not None and key_padding_mask.dtype == torch.bool: # (b, s)
				key_padding_mask = key_padding_mask.unsqueeze(1) # (b, 1, s)
				atten = atten.masked_fill(key_padding_mask, float('-inf'))
			if attn_mask is not None and attn_mask.dtype == torch.bool: # (b, s, s)
				atten = atten.masked_fill(attn_mask, float('-inf'))
			if is_casual and self.training:
				seq_len = atten.size(-1)
				casual_mask = torch.triu(torch.ones(seq_len, seq_len, device=atten.device), diagonal=1).bool()
				atten = atten.masked_fill(casual_mask, float('-inf'))
			atten = torch.softmax(atten, dim=-1)  # -inf will be 0
			atten = self.dropout(atten)
			x_att.append(
				torch.matmul(atten, # (b,s,s)
							V[n], # (b,s,d/h)
			)) # (b,s,d/h)

		x_att = torch.cat(x_att, dim=-1)
		x_att = self.out_proj(x_att)
		x_att = self.dropout(x_att)
		return x_att
