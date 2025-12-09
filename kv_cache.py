import time
import torch
from torch import nn
import random
import numpy as np
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)

class SimpleMHA(nn.Module):
	def __init__(self, d, kv_cache_en):
		super().__init__()
		self.q_emb = nn.Linear(d, d)
		self.k_emb = nn.Linear(d, d)
		self.v_emb = nn.Linear(d, d)
		self.out_emb = nn.Linear(d, d)
		self.d = d
		self.kv_cache_en = kv_cache_en
		self.k_cache = None
		self.v_cache = None

	def forward(self, x): # x: initial tokens (b, s1, d), subsequent tokens (b, 1, d)
		q = self.q_emb(x)
		k = self.k_emb(x)
		v = self.v_emb(x)

		if self.kv_cache_en and (self.k_cache is not None and self.v_cache is not None):
			k = torch.cat((self.k_cache, k), dim=1) # (b, s, d)
			v = torch.cat((self.v_cache, v), dim=1) # (b, s, d)
		self.k_cache = k
		self.v_cache = v

		att_wt = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.d)) # (b, 1, d) * (b, d, s) => (b, 1, s), computation b*d*s, s times smaller than no kv cache

		# add causual mask for initial inference, because otherwise it will be different behaviour with training (and autogressive)
		if q.shape[1] > 1:
			casual_mask = torch.triu(torch.ones(q.shape[1], k.shape[1], device=att_wt.device), diagonal=1).bool()
			att_wt = att_wt.masked_fill(casual_mask, float('-inf'))
		# print("attention weight shape: ", att_wt.shape)
		# print(att_wt)

		att_wt = torch.softmax(att_wt, dim=-1)
		x = torch.matmul(att_wt, v) # (b, 1, d)
		x = self.out_emb(x)
		return x


class SimpleDecoderLayer(nn.Module):
	def __init__(self, d, kv_cache_en):
		super().__init__()
		self.mha = SimpleMHA(d, kv_cache_en)
		self.norm1 = nn.RMSNorm(d)
		self.norm2 = nn.RMSNorm(d)
		self.ffn = nn.Sequential(
			nn.Linear(d, 4*d),
			nn.GELU(),
			nn.Linear(4*d, d)
		)

	def forward(self, x): # x: initial tokens (b, s1, d), subsequent tokens (b, 1, d)
		residual = x
		x = self.norm1(x)
		x = self.mha(x) + residual

		residual = x
		x = self.norm2(x)
		x = self.ffn(x) + residual
		return x


MAX_SEQ = 100
class CasualLM(nn.Module):
	def __init__(self, d, layers, vocab_size, kv_cache_en):
		super().__init__()
		self.input_emb = nn.Linear(vocab_size, d) # vocab -> hidden dim
		self.layers = nn.Sequential(
			*[SimpleDecoderLayer(d, kv_cache_en) for _ in range(layers)]
		)
		self.output_emb = nn.Linear(d, vocab_size)
		self.pos_emb = torch.arange(0, MAX_SEQ).unsqueeze(0).unsqueeze(-1) # (1, 100, 1)
		self.num_token = 0
		self.kv_cache_en = kv_cache_en

	def forward(self, x):
		seq_len = x.shape[1] # inital s, subsequent 1

		x = self.input_emb(x)
		if self.kv_cache_en:
			x = x + self.pos_emb[:, self.num_token:self.num_token+seq_len, :]
			self.num_token += seq_len
		else:
			x = x + self.pos_emb[:, 0:seq_len, :]

		x = self.layers(x) # (b, s or 1, d)
		x = x[:, -1:, :] # (b, 1, d)
		x = self.output_emb(x) # (b, 1, vocab)
		# return torch.argmax(x, dim=-1) # return index in real pred
		return x # return vector for kvcache demo


INITIAL_TOKEN = 10
GENERATE_STEP = 50
VOCAB_SIZE = 10

def seed_everything(seed=42):
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

print("Without KV cache:")
seed_everything(42)
all_tokens = torch.rand((1, INITIAL_TOKEN, VOCAB_SIZE))
model = CasualLM(d=512, layers=1, vocab_size=VOCAB_SIZE, kv_cache_en=False)
model.eval()
t1 = time.time()
with torch.no_grad():
	for i in range(GENERATE_STEP):
		cur_token = model(all_tokens)  # use all tokens
		all_tokens = torch.concat([all_tokens, cur_token], dim=1)
print("Time: ", time.time() - t1)

print("With KV cache:")
seed_everything(42)
all_tokens = torch.rand((1, INITIAL_TOKEN, VOCAB_SIZE))
model = CasualLM(d=512, layers=1, vocab_size=VOCAB_SIZE, kv_cache_en=True)
model.eval()
t1 = time.time()
with torch.no_grad():
	cur_token = model(all_tokens)
	for i in range(GENERATE_STEP - 1):
		cur_token = model(cur_token)  # use one token
		all_tokens = torch.cat([all_tokens, cur_token], dim=1)
print("Time: ", time.time() - t1)


"""
Without KV cache:
attention weight shape:  torch.Size([1, 2, 2])
tensor([[[ 0.1181,    -inf],
         [-0.0039,  0.1466]]])
attention weight shape:  torch.Size([1, 3, 3])
tensor([[[ 0.1181,    -inf,    -inf],
         [-0.0039,  0.1466,    -inf],
         [-0.0834,  0.1021,  0.2259]]])
attention weight shape:  torch.Size([1, 4, 4])
tensor([[[ 0.1181,    -inf,    -inf,    -inf],
         [-0.0039,  0.1466,    -inf,    -inf],
         [-0.0834,  0.1021,  0.2259,    -inf],
         [-0.1373,  0.0813,  0.2302,  0.2296]]])
attention weight shape:  torch.Size([1, 5, 5])
tensor([[[ 0.1181,    -inf,    -inf,    -inf,    -inf],
         [-0.0039,  0.1466,    -inf,    -inf,    -inf],
         [-0.0834,  0.1021,  0.2259,    -inf,    -inf],
         [-0.1373,  0.0813,  0.2302,  0.2296,    -inf],
         [-0.1545,  0.0706,  0.2260,  0.2252,  0.2176]]])
attention weight shape:  torch.Size([1, 6, 6])
tensor([[[ 0.1181,    -inf,    -inf,    -inf,    -inf,    -inf],
         [-0.0039,  0.1466,    -inf,    -inf,    -inf,    -inf],
         [-0.0834,  0.1021,  0.2259,    -inf,    -inf,    -inf],
         [-0.1373,  0.0813,  0.2302,  0.2296,    -inf,    -inf],
         [-0.1545,  0.0706,  0.2260,  0.2252,  0.2176,    -inf],
         [-0.1677,  0.0642,  0.2248,  0.2236,  0.2162,  0.2135]]])

With KV cache:
attention weight shape:  torch.Size([1, 2, 2])
tensor([[[ 0.1181,    -inf],
         [-0.0039,  0.1466]]])
attention weight shape:  torch.Size([1, 1, 3])
tensor([[[-0.0834,  0.1021,  0.2259]]])
attention weight shape:  torch.Size([1, 1, 4])
tensor([[[-0.1373,  0.0813,  0.2302,  0.2296]]])
attention weight shape:  torch.Size([1, 1, 5])
tensor([[[-0.1545,  0.0706,  0.2260,  0.2252,  0.2176]]])
attention weight shape:  torch.Size([1, 1, 6])
tensor([[[-0.1677,  0.0642,  0.2248,  0.2236,  0.2162,  0.2135]]])
"""

"""
Without KV cache:
Time:  0.05321383476257324
With KV cache:
Time:  0.022008419036865234
"""
