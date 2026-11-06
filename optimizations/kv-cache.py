
"""带 KV cache 的多头因果自注意力，支持提示词预填充和增量解码。

每次只计算新 token 的 Q、K、V，并复用历史 K、V。
forward 保留梯度，文件末尾的示例通过 inference_mode 运行推理验证。
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


# 缓存为 (K, V)，两者的形状均为 [B, H, L, D]。
# B 为批大小，H 为头数，L 为已缓存长度，D 为每个头的维度。
KVCache = tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttentionWithKVCache(nn.Module):
	"""适用于 batch 内有效长度相同、无 padding 的输入。

	每个注意力层、每批序列各自维护缓存，位置编码由调用方处理。
	缓存对应生成它的模型参数和序列前缀，参数或前缀变化后需要重新构建。
	"""

	def __init__(self, 
	    embed_size: int, 
		head_num: int, 
		max_context_length: int = 1024, 
		qkv_bias: bool = False):
		super().__init__()
		assert head_num > 0 and embed_size % head_num == 0, "embed_size 必须能被 head_num 整除"
		self.head_num = head_num
		self.head_dim = embed_size // head_num
		# 上限约束历史缓存与本轮输入的总长度。
		self.max_context_length = max_context_length

		self.W_q = nn.Linear(embed_size, embed_size, bias=qkv_bias)
		self.W_k = nn.Linear(embed_size, embed_size, bias=qkv_bias)
		self.W_v = nn.Linear(embed_size, embed_size, bias=qkv_bias)
		self.W_o = nn.Linear(embed_size, embed_size)

		# 行对应 Query 的位置，列对应 Key 的位置。严格上三角为未来位置。
		# 此处 True 表示屏蔽，传给 SDPA 时需要取反。
		# 注册为 buffer 后会随模型迁移设备、进入 state_dict，并作为固定掩码使用。
		self.register_buffer(
			"mask",
			torch.triu(torch.ones(self.max_context_length, self.max_context_length, dtype=torch.bool), diagonal=1),
		)

	def forward(
		self,
		x: torch.Tensor,
		kv_cache: Optional[KVCache] = None,
	) -> tuple[torch.Tensor, KVCache]:
		"""处理本轮新输入，返回这些位置的输出和更新后的 KV cache。

		x 的形状为 [B, new_L, E]，E 为 embed_size。已有缓存时只传新 token。
		kv_cache 为上一轮返回的缓存，首次调用传 None。
		输出形状为 [B, new_L, E]，缓存形状为 [B, H, past_len + new_L, D]。
		"""
		B, new_L, E = x.shape
		if new_L > self.max_context_length:
			raise ValueError(f"序列长度 {new_L} 超过最大上下文长度 {self.max_context_length}")

		# 将新 token 投影为 Q、K、V，并拆分注意力头。
		# [B, new_L, E] -> [B, new_L, H, D] -> [B, H, new_L, D]。
		q = self.W_q(x).view(B, new_L, self.head_num, self.head_dim).transpose(1, 2)
		k = self.W_k(x).view(B, new_L, self.head_num, self.head_dim).transpose(1, 2)
		v = self.W_v(x).view(B, new_L, self.head_num, self.head_dim).transpose(1, 2)

		# 沿序列维追加新 K、V。Q 只保留本轮位置，后续计算会读取完整的 K、V。
		# torch.cat 每轮都会复制缓存，长序列推理可改用预分配缓存。
		past_len = 0
		if kv_cache is not None:
			past_k, past_v = kv_cache
			assert past_k.shape == past_v.shape, "KV cache 的 K 和 V 形状必须一致"
			past_len = past_k.shape[2]
			k = torch.cat((past_k, k), dim=-2)
			v = torch.cat((past_v, v), dim=-2)

		# 所有解码方式都检查总长度，包括每轮只新增一个 token 的情况。
		if (past_len + new_L) > self.max_context_length:
			raise ValueError(
				f"总序列长度 {past_len + new_L} 超过最大上下文长度 {self.max_context_length}"
			)

		# 有历史缓存且 new_L=1 时，全部 Key 都位于当前 Query 的左侧或自身，均可读取。
		# 有历史缓存且 new_L>1 时，还需要屏蔽本轮输入内部的未来位置。
		mask = None
		if past_len > 0 and new_L > 1:
			# 新 Query 的位置从 past_len 开始，因此取对应的行和所有已处理 Key 的列。
			# 取反后 True 表示允许读取，形状 [new_L, past_len + new_L] 会广播到各 batch 和头。
			mask = ~self.mask[past_len: past_len + new_L, :past_len + new_L]

		# 首次调用时 Q、K 等长，使用 SDPA 自带的标准因果掩码。
		# 有缓存时 Q、K 长度不同，使用上面构造的可见性规则。
		out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=past_len==0)

		# 合并注意力头并做输出投影，返回当前新位置的结果以及完整缓存。
		out = out.transpose(1, 2).reshape(B, new_L, E)
		return self.W_o(out), (k, v)

@torch.inference_mode()
def demo():
    """比较全量计算与两种分块计算，逐位置验证输出一致。"""
    torch.manual_seed(7)
    attention = CausalSelfAttentionWithKVCache(embed_size=32, head_num=4).double().eval()
    x = torch.randn(2, 8, 32, dtype=torch.float64)
    # 使用相同的输入和模型权重，以一次性全序列计算作为基准。
    expected, _ = attention(x)

    # 都先预填充 3 个 token，分别验证逐 token 解码和多 token 分块解码。
    for sizes in ((3, 1, 1, 1, 1, 1), (3, 2, 3)):
        # 每种分块方案从空缓存开始。
        cache = None
        parts = []
        for block in x.split(sizes, dim=1):
            output, cache = attention(block, cache)
            parts.append(output)
        actual = torch.cat(parts, dim=1)
        # 不同分块可能产生浮点舍入差异，使用适合当前 float64 示例的误差容限。
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        error = (actual - expected).abs().max().item()
        print(f"输入分块 {sizes}，最大绝对误差 {error:.2e}，K 形状 {tuple(cache[0].shape)}")


if __name__ == "__main__":
    demo()
			
