import math
import torch
from torch import nn
import torch.nn.functional as F

torch_impl = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=512*4,
    dropout=0.1,
    activation=F.gelu,
	batch_first=True,
    norm_first=True,
    bias=True,
    device=torch.device('cpu')
)

ifm = torch.rand(32, 10, 512)
ofm = torch_impl(ifm)
print("torch impl:", ofm.shape)

class MyTransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: callable = F.gelu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device = torch.device("cpu")
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias)
        self.k_proj = nn.Linear(d_model, d_model, bias)
        self.v_proj = nn.Linear(d_model, d_model, bias)
        self.out_proj = nn.Linear(d_model, d_model, bias)
        self.ffn1 = nn.Linear(d_model, dim_feedforward, bias)
        self.ffn2 = nn.Linear(dim_feedforward, d_model, bias)
        self.layernorm1 = nn.LayerNorm(d_model, layer_norm_eps, bias)
        self.layernorm2 = nn.LayerNorm(d_model, layer_norm_eps, bias)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.nhead = nhead
        self.activation = activation
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.device = device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (b, s, d) or (s, b, d)
        if not self.batch_first:
            x = torch.transpose(x, 0, 1)

        # Pre-Norm
        if self.norm_first:
            x = self.layernorm1(x)

        # Multihead Self Attention
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = torch.chunk(Q, self.nhead, dim=-1)  # (b, s, d/h) * h
        K = torch.chunk(K, self.nhead, dim=-1)  # (b, s, d/h) * h
        V = torch.chunk(V, self.nhead, dim=-1)  # (b, s, d/h) * h
        # torch.split: 2nd parameter is size of each chunk
        # torch.chunk: 2nd parameter is number of chunks

        x_att = []
        for n in range(self.nhead):
            atten = torch.matmul(Q[n], # (b,s,d/h)
                                torch.transpose(K[n], -1, -2) # (b,d/h,s)
                                # transpose only accepts two arguments except for tensor
            ) # (b,s,s)
            atten /= math.sqrt(self.d_model / self.nhead)
            atten = torch.softmax(atten, dim=-1)
            atten = self.dropout(atten)
            x_att.append(
                torch.matmul(atten, # (b,s,s)
                            V[n], # (b,s,d/h)
            )) # (b,s,d/h)

        x_att = torch.cat(x_att, dim=-1)
        x_att = self.out_proj(x_att)
        x_att = self.dropout(x_att)

        # Add
        x = x + x_att # (b,s,d)

        # Post-Norm
        if not self.norm_first:
            x = self.layernorm1(x)

        # Pre-Norm
        if self.norm_first:
            x = self.layernorm2(x)

        # FFN
        x_ffn = self.ffn1(x)
        x_ffn = self.activation(x_ffn)
        x_ffn = self.dropout(x_ffn)
        x_ffn = self.ffn2(x_ffn)
        x_ffn = self.dropout(x_ffn)

        # Add
        x = x + x_ffn

        # Post-Norm
        if not self.norm_first:
            x = self.layernorm2(x)

        if not self.batch_first:
            x = torch.transpose(x, 0, 1)
        return x

my_impl = MyTransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=512*4,
    dropout=0.1,
    activation=F.gelu,
	batch_first=True,
    norm_first=True,
    bias=True,
    device=torch.device('cpu')
)

ifm = torch.rand(32, 10, 512)
ofm = my_impl(ifm)
print("my impl:", ofm.shape)
