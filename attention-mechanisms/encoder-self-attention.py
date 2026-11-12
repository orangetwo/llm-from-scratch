# Encoder self-attention
import torch
import torch.nn as nn
from typing import Optional

class EncoderSelfAttention(nn.Module):
	"""
	多头自注意力: x 为 [B, L, E], mask 为 [B, L] 的 bool tensor
	pad_mask 中 true 表示 padding, 每条序列中至少包含一个有效 token
	"""
	def __init__(self, embed_size: int, head_num: int, qkv_bias: bool=False):
		super().__init__()
		assert head_num >0 and embed_size % head_num == 0, f"维度必须能被头数整除"
		self.head_num = head_num
		self.head_dim = embed_size // head_num

		self.W_q = nn.Linear(embed_size, embed_size, bias=qkv_bias)
		self.W_k = nn.Linear(embed_size, embed_size, bias=qkv_bias)
		self.W_v = nn.Linear(embed_size, embed_size, bias=qkv_bias)
		self.W_o = nn.Linear(embed_size, embed_size)

	def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
		B, L, E = x.shape # B: batch size, L: sequence length, E: embeding dimension

		# 1. 同一个 x 分别生成 Q、K、V：[B, L, E]
		# 2. 拆成 H 个头：[B, L, E] -> [B, L, H, D] -> [B, H, L, D]
		q = self.W_q(x).reshape(B, L, self.head_num, self.head_dim).transpose(1, 2)
		k = self.W_k(x).reshape(B, L, self.head_num, self.head_dim).transpose(1, 2)
		v = self.W_v(x).reshape(B, L, self.head_num, self.head_dim).transpose(1, 2)

		# 3. 每个头计算 token 之间的匹配分数：[B, H, L, L]
		attn_score = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
		if pad_mask is not None:
			assert (~pad_mask).any(dim=-1).all(), "每条序列至少需要一个有效 token"
			assert pad_mask.dtype == torch.bool and pad_mask.shape == (B, L), f" "
			# mask 为 True 的位置表示 padding token, 这些位置的分数设为 -inf
			attn_score = attn_score.masked_fill(pad_mask[:, None, None, :], float('-inf'))

		# 4. 按行归一化，再对 V 加权求和：[B, H, L, D]
		attn_weight = torch.softmax(attn_score, dim=-1)
		context = attn_weight @ v

		# 5. 将多头拼接回原来的维度：[B, H, L, D] -> [B, L, E]
		context = context.transpose(1, 2).reshape(B, L, E)

		return self.W_o(context)



if __name__ == "__main__":
	mat = torch.rand(4, 4)
	print(mat)
	mask = torch.triu(
		torch.ones(mat.shape[1], mat.shape[1], dtype=torch.bool),
		diagonal=1,
	)
	print(mask)