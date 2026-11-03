# Decoder-only self-attention
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
	def __init__(self, embed_size: int, head_num: int, context_length: int, qkv_bias: bool=False):
		super().__init__()
		assert head_num >0 and embed_size % head_num == 0, f"维度必须能被头数整除"
		self.head_num = head_num
		self.head_dim = embed_size // head_num
		self.context_length = context_length

		self.W_q = nn.Linear(embed_size, embed_size, bias=qkv_bias)
		self.W_k = nn.Linear(embed_size, embed_size, bias=qkv_bias)
		self.W_v = nn.Linear(embed_size, embed_size, bias=qkv_bias)
		self.W_o = nn.Linear(embed_size, embed_size)

		self.register_buffer(
			"mask", 
		    torch.triu(torch.ones(self.context_length, self.context_length, dtype=torch.bool), diagonal=1)	
		)

	def forward(self, x: torch.Tensor):
		B, L, E = x.shape # B: batch size, L: sequence length, E: embeding dimension
		if L > self.context_length:
			raise ValueError(f"序列长度 {L} 超过最大上下文长度 {self.context_length}")

		# 1. 同一个 x 分别生成 Q、K、V：[B, L, E]
		q = self.W_q(x)
		k = self.W_k(x)
		v = self.W_v(x)

		# 2. 拆成 H 个头：[B, L, E] -> [B, L, H, D] -> [B, H, L, D]
		q = q.reshape(B, L, self.head_num, self.head_dim).transpose(1, 2)
		k = k.reshape(B, L, self.head_num, self.head_dim).transpose(1, 2)
		v = v.reshape(B, L, self.head_num, self.head_dim).transpose(1, 2)

		# 3. 每个头计算 token 之间的匹配分数：[B, H, L, L], 注意 mask 为 [L, L], 所以增加前两个维度
		attn_score = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
		mask_bool = self.mask.bool()[:L, :L]  # 只取前 L 个 token 的 mask
		attn_score = attn_score.masked_fill(mask_bool[None, None, :, :], float('-inf'))

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