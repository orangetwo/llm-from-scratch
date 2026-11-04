"""GQA 因果自注意力：多个 Query head 分组共享 K/V。"""

import math
import torch
from torch import nn


class GroupedQueryAttention(nn.Module):
	def __init__(self, embed_size: int, head_num: int, group_size: int, max_context_length: int, qkv_bias: bool=False):
		"""
		embed_size：输入、输出的特征维度 E。
		head_num：Query head 的数量 H，每个 head 的维度 D = E // H。
		group_size：KV head 的数量 G，也就是组数；每组有 H // G 个 Query head。
		    G = H 时为 MHA，G = 1 时为 MQA，其余合法取值为分组共享的 GQA。
		max_context_length：预先创建的因果遮罩所支持的最大序列长度。
		qkv_bias：是否为 Q、K、V 的线性投影添加偏置。
		"""
		super().__init__()
		assert embed_size > 0 and head_num > 0 and embed_size % head_num == 0, "embed_size、head_num 必须大于零，且 embed_size 必须能被 head_num 整除"
		assert group_size > 0 and head_num % group_size == 0, "组数大于零且 head_num 必须能被 group_size 整除"
		assert max_context_length > 0, "max_context_length 必须大于零"

		self.head_num = head_num
		self.head_dim = embed_size // head_num
		self.group_size = group_size

		# Q 保留 H 个 head；K/V 只生成 G 个 head，因此投影维度为 G * D。
		self.W_q = nn.Linear(embed_size, embed_size, bias=qkv_bias)
		self.W_k = nn.Linear(embed_size, self.group_size * self.head_dim, bias=qkv_bias)
		self.W_v = nn.Linear(embed_size, self.group_size * self.head_dim, bias=qkv_bias)
		self.W_o = nn.Linear(embed_size, embed_size)

		# 上三角为 True，表示需要屏蔽的未来位置；对角线保留，允许关注自身。
		# buffer 不参与训练，但会随 model.to(device) 移动，并保存在 state_dict 中。
		self.register_buffer(
			"mask",
			torch.triu(torch.ones(max_context_length, max_context_length, dtype=torch.bool), diagonal=1)
		)

	def forward(self, x: torch.Tensor):
		B, L, E = x.shape  # batch size、序列长度、特征维度
		if L > self.mask.size(0):
			raise ValueError(f"序列长度 {L} 超过 max_context_length={self.mask.size(0)}")

		# 1. Q/K/V 来自同一个 x，因此是 self-attention。
		q = self.W_q(x)  # [B, L, E]，其中 E = H * D
		k = self.W_k(x)  # [B, L, G * D]
		v = self.W_v(x)  # [B, L, G * D]

		# 2. 拆分 head 并将 head 维度移到前面，方便批量矩阵乘法。
		q = q.reshape(B, L, self.head_num, self.head_dim).transpose(1, 2)  # [B, H, L, D]
		k = k.reshape(B, L, self.group_size, self.head_dim).transpose(1, 2)  # [B, G, L, D]
		v = v.reshape(B, L, self.group_size, self.head_dim).transpose(1, 2)  # [B, G, L, D]

		# 3. 每组 Query head 共享 K/V，将 K/V 展开为 [B, H, L, D]。
		# 例如 H = 4、G = 2：[K0, K1] -> [K0, K0, K1, K1]。
		# repeat_interleave 保留计算图，各副本的梯度会自动累加到原始 K/V。
		k = k.repeat_interleave(self.head_num // self.group_size, dim=1)
		v = v.repeat_interleave(self.head_num // self.group_size, dim=1)

		# 4. QKᵀ 得到每个 Query 对所有 Key 的分数，除以 sqrt(D) 控制数值尺度。
		scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L, L]

		# 5. [L, L] 遮罩广播到所有 batch 和 head；-inf 经 softmax 后变为 0。
		scores = scores.masked_fill(self.mask[:L, :L], float('-inf'))
		attn = torch.softmax(scores, dim=-1)  # 沿 Key 维度归一化，每行权重之和为 1
		context = attn @ v  # 对 Value 加权求和：[B, H, L, D]

		# 6. 拼接所有 head：[B, H, L, D] -> [B, L, H, D] -> [B, L, E]。
		# transpose 后可能不连续，reshape 会在需要时复制数据，并保留梯度连接。
		context = context.transpose(1, 2).reshape(B, L, E)

		return self.W_o(context)  # 输出投影融合各 head 的信息，形状仍为 [B, L, E]
