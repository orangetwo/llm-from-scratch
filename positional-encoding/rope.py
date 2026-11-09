
import torch

# 推荐的视频: https://www.bilibili.com/video/BV1FjrCBdESo/?spm_id_from=333.337.search-card.all.click&vd_source=16a807f67f2b79552e3a4b294873de82

"""理论上"""
# RoPE（Rotary Position Embedding）旋转位置编码矩阵
#
# R^d_{θ,m} =
#
# ⎡ cos(mθ₀)  -sin(mθ₀)       0          0        ⋯         0                0        ⎤
# ⎢ sin(mθ₀)   cos(mθ₀)       0          0        ⋯         0                0        ⎥
# ⎢    0           0       cos(mθ₁)  -sin(mθ₁)    ⋯         0                0        ⎥
# ⎢    0           0       sin(mθ₁)   cos(mθ₁)    ⋯         0                0        ⎥
# ⎢    ⋮           ⋮          ⋮          ⋮        ⋱         ⋮                ⋮        ⎥
# ⎢    0           0          0          0        ⋯   cos(mθ₍d/2-1₎)  -sin(mθ₍d/2-1₎) ⎥
# ⎣    0           0          0          0        ⋯   sin(mθ₍d/2-1₎)   cos(mθ₍d/2-1₎) ⎦
#
# 其中：
# d   : 向量维度（必须为偶数）
# m   : Token 的位置索引
# θᵢ  : 第 i 个二维子空间的旋转频率
#
# θᵢ = 10000^(-2i/d), i = 0, 1, ..., d/2-1
#
# 整个矩阵由 d/2 个 2×2 旋转矩阵构成。
# 每两个相邻维度组成一个二维子空间，按照角度 mθᵢ 进行旋转。

"""实际实现"""
# RoPE 旋转位置编码：
#
# R^d_(θ,m) x =
#
#   ⎡  x₀  ⎤   ⎡ cos(mθ₀) ⎤   ⎡ -x₁ ⎤   ⎡ sin(mθ₀) ⎤
#   ⎢  x₁  ⎥   ⎢ cos(mθ₀) ⎥   ⎢  x₀ ⎥   ⎢ sin(mθ₀) ⎥
#   ⎢  x₂  ⎥   ⎢ cos(mθ₁) ⎥   ⎢ -x₃ ⎥   ⎢ sin(mθ₁) ⎥
#   ⎢  x₃  ⎥ ⊙ ⎢ cos(mθ₁) ⎥ + ⎢  x₂ ⎥ ⊙ ⎢ sin(mθ₁) ⎥
#   ⎢  ⋮   ⎥   ⎢    ⋮     ⎥   ⎢  ⋮   ⎥   ⎢    ⋮     ⎥
#   ⎣  ⋮   ⎦   ⎣    ⋮     ⎦   ⎣  ⋮   ⎦   ⎣    ⋮     ⎦
#
# 其中：
# ⊙：Hadamard 积（逐元素乘法）
# m：Token 的位置索引
# θᵢ：第 i 个二维分组的旋转频率
#
# 对应 PyTorch 实现：
# x_rope = x * cos + rotate_half(x) * sin


def precompute_freqs(dim: int, seq_len: int, base: float = 10000.0):
	"""
    预计算 RoPE 所需的 cos 和 sin。

    Args:
        dim: 每个注意力头的维度，必须为偶数
        seq_len: Token 序列长度
        theta: 旋转频率的基数，默认 10000

    Returns:
        cos: [seq_len, dim]
        sin: [seq_len, dim]
    """
	assert dim % 2 == 0, "dim 必须为偶数"

	# 1. 计算每个二维分组对应的旋转频率
    # θ_i = 10000^(-2i/d)
    # shape: [dim // 2]
	freqs = 1.0 / (
          base ** (torch.arange(0, dim, 2).float() / dim)
	)

	## 2. 生成 Token 的位置索引
    # m = 0, 1, ..., seq_len - 1
    # shape: [seq_len]
	positions = torch.arange(seq_len)

	# 3. 计算每个 Token 在每个二维分组中的旋转角度
    # angles[m, i] = m * θ_i
    # shape: [seq_len, dim // 2]
	angles = torch.outer(positions, freqs)

	# 4. 计算 cos 和 sin
    # 每个二维分组的两个两个元素使用相同的旋转角度
    # repeat_interleave(2) 将每个角度重复两次，得到与原始维度相同的形状
    # ex:
    # [cos(θ0), cos(θ1)]
    # 变为
    # [cos(θ0), cos(θ0), cos(θ1), cos(θ1)]
    # shape: [seq_len, dim]
    # 举个例子：
	# 假设 dim = 4, seq_len = 3
	# cos.shape = [3, 4]
	#
	# cos = [
	#     [cos(0θ₀), cos(0θ₀), cos(0θ₁), cos(0θ₁)],
	#     [cos(1θ₀), cos(1θ₀), cos(1θ₁), cos(1θ₁)],
	#     [cos(2θ₀), cos(2θ₀), cos(2θ₁), cos(2θ₁)]
	# ]
	#
	# sin.shape = [3, 4]
	#
	# sin = [
	#     [sin(0θ₀), sin(0θ₀), sin(0θ₁), sin(0θ₁)],
	#     [sin(1θ₀), sin(1θ₀), sin(1θ₁), sin(1θ₁)],
	#     [sin(2θ₀), sin(2θ₀), sin(2θ₁), sin(2θ₁)]
	# ]
	cos = torch.cos(angles).repeat_interleave(2, dim=-1)
	sin = torch.sin(angles).repeat_interleave(2, dim=-1)


	return cos, sin

def rotate_half(x: torch.Tensor):
	"""
	对向量中相邻的两个元素进行旋转变换。

	输入：
		[x0, x1, x2, x3, ...]

	输出：
		[-x1, x0, -x3, x2, ...]

	注意：这里还没有真正完成 RoPE 旋转。
	"""
	assert x.shape[-1] % 2 == 0

	# 每两个相邻元素组成一组
	# [..., dim] -> [..., dim // 2, 2]
	x_pairs = x.reshape(*x.shape[:-1], -1, 2)

	# 分别提取每组的两个元素
	x_even = x_pairs[..., 0]
	x_odd = x_pairs[..., 1]

	# (-x1, x0), (-x3, x2), ...
	rotated = torch.stack(
		[-x_odd, x_even],
		dim=-1
	)

	# 恢复原来的形状
	return rotated.flatten(-2)
    

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    对输入向量应用 RoPE。

    Args:
        x: [seq_len, dim]
        cos: [seq_len, dim]
        sin: [seq_len, dim]

    Returns:
        旋转后的向量：[seq_len, dim]
    """

    # R_m x = x ⊙ cos(mθ) + rotate_half(x) ⊙ sin(mθ)
    return x * cos + rotate_half(x) * sin


if __name__ == "__main__":

    # 假设序列包含 3 个 Token，每个 Token 的 Query 维度为 4
    seq_len = 3
    dim = 4

    # 这里假设 Query 已经经过 W_q 线性投影
    q = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0]
    ])

    # 预计算旋转因子
    cos, sin = precompute_freqs(dim, seq_len)

    # 对 Query 应用 RoPE
    q_rope = apply_rope(q, cos, sin)

    print("原始 Query:")
    print(q)

    print("\ncos:")
    print(cos)

    print("\nsin:")
    print(sin)

    print("\n旋转后的 Query:")
    print(q_rope)
	