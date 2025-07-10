import torch
from torch import nn
import torch.nn.functional as F

torch_impl = nn.TransformerDecoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=512*4,
    dropout=0.1,
    activation=F.gelu,
    norm_first=True,
    bias=True,
    device=torch.device('cpu')
)

ifm = torch.rand((32, 10, 512), pin_memory=True)

ofm = torch_impl(ifm)
print(ofm.shape)

# class MyTransformerDecoder(nn.Module):
#     def __init__(self, 