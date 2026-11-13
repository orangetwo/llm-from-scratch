# Decoder-only self-attention
import torch
import torch.nn as nn
from typing import Optional

class CausalSelfAttention(nn.Module):
	"""
	多头自注意力: x 为 [B, L, E], pad_mask 为 [B, L] 的 bool tensor
	pad_mask 中 true 表示 padding, 每条序列中至少包含一个有效 token
	"""
	def __init__(self, embed_size: int, head_num: int, max_context_length: int, qkv_bias: bool=False):
		super().__init__()
		assert head_num >0 and embed_size % head_num == 0, f"维度必须能被头数整除"
		self.head_num = head_num
		self.head_dim = embed_size // head_num
		self.max_context_length = max_context_length

		self.W_q = nn.Linear(embed_size, embed_size, bias=qkv_bias)
		self.W_k = nn.Linear(embed_size, embed_size, bias=qkv_bias)
		self.W_v = nn.Linear(embed_size, embed_size, bias=qkv_bias)
		self.W_o = nn.Linear(embed_size, embed_size)

		self.register_buffer(
			"causal_mask", 
			torch.triu(torch.ones(self.max_context_length, self.max_context_length, dtype=torch.bool), diagonal=1)	
		)

	def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
		B, L, E = x.shape # B: batch size, L: sequence length, E: embeding dimension
		if L > self.max_context_length:
			raise ValueError(f"序列长度 {L} 超过最大上下文长度 {self.max_context_length}")

		# 1. 同一个 x 分别生成 Q、K、V：[B, L, E]
		# 2. 拆成 H 个头：[B, L, E] -> [B, L, H, D] -> [B, H, L, D]
		q = self.W_q(x).reshape(B, L, self.head_num, self.head_dim).transpose(1, 2)
		k = self.W_k(x).reshape(B, L, self.head_num, self.head_dim).transpose(1, 2)
		v = self.W_v(x).reshape(B, L, self.head_num, self.head_dim).transpose(1, 2)

		# 3. 每个头计算 token 之间的匹配分数：[B, H, L, L], 注意 mask 为 [L, L], 所以增加前两个维度
		attn_score = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
		mask_bool = self.causal_mask[:L, :L]  # 只取前 L 个 token 的 mask
		if pad_mask is not None:
			assert pad_mask.device == x.device, "pad_mask 必须与 x 在同一设备"
			assert pad_mask.dtype == torch.bool and pad_mask.shape == (B, L), f"pad_mask 必须是 bool pad_mask 形状应为 {(B, L)}，实际 {tuple(pad_mask.shape)}"
			assert (~pad_mask).any(dim=-1).all(), "每条序列至少需要一个有效 token"
			# mask 为 True 的位置表示 padding token, 这些位置的分数设为 -inf
			mask_bool = pad_mask[:, None, None, :] | mask_bool[None, None, :, :]
		attn_score = attn_score.masked_fill(mask_bool, float('-inf'))

		# 4. 按行归一化，再对 V 加权求和：[B, H, L, D]
		attn_weight = torch.softmax(attn_score, dim=-1)
		context = attn_weight @ v

		# 5. 将多头拼接回原来的维度：[B, H, L, D] -> [B, L, E]
		context = context.transpose(1, 2).reshape(B, L, E)		

		return self.W_o(context)

if __name__ == "__main__":
	"""
	对于 softmax、`reshape`、加法这类运算，只要直接对 Tensor 做计算，结果通常就在输入所在的设备上。 这个设备信息就是 x.device。

	a = torch.softmax(x, dim=-1)   # 在 GPU 上，直接对 x 做运算
	b = torch.ones_like(x)         # 在 GPU 上，使用 x 的形状和设备
	c = torch.ones(x.shape)        # 使用默认设备，通常是 CPU
	最后一行虽然写了 x，但传给 torch.ones 的只有 x.shape。形状包含各维度的大小，不包含设备信息。

	"""
	pass