from typing import Optional

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """多头交叉注意力。decoder 提供 Q，encoder 提供 K 和 V。

    decoder 是解码器传入的隐藏状态，形状为 [B, T, E]，T 是目标序列长度。
    encoder 是编码器输出的隐藏状态，形状为 [B, S, E]，S 是源序列长度。
    交叉注意力层为每个目标位置返回一个 E 维向量，因此该层的输出形状为 [B, T, E]。
    可选的 encoder_padding_mask 为 [B, S] 的布尔张量，True 表示忽略该位置。
    """

    def __init__(self, embed_size: int, head_num: int, qkv_bias: bool = False):
        super().__init__()
        if embed_size <= 0 or head_num <= 0 or embed_size % head_num != 0:
            raise ValueError("embed_size 和 head_num 必须为正数，且 embed_size 必须能被 head_num 整除")
        self.head_num = head_num
        self.head_dim = embed_size // head_num

        self.W_q = nn.Linear(embed_size, embed_size, bias=qkv_bias)
        self.W_k = nn.Linear(embed_size, embed_size, bias=qkv_bias)
        self.W_v = nn.Linear(embed_size, embed_size, bias=qkv_bias)
        self.W_o = nn.Linear(embed_size, embed_size)

    def forward(
        self,
        decoder: torch.Tensor,
        encoder: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, E = decoder.shape
        batch_size, S, encoder_dim = encoder.shape
        if B != batch_size or E != encoder_dim or E != self.W_q.in_features:
            raise ValueError("decoder 和 encoder 的批大小必须一致，末维必须等于 embed_size")
        if S == 0:
            raise ValueError("encoder 序列不能为空")

        # 1. 拆头。Q 保留目标长度 T，K 和 V 保留源长度 S。
        q = self.W_q(decoder).reshape(B, T, self.head_num, self.head_dim).transpose(1, 2)
        k = self.W_k(encoder).reshape(B, S, self.head_num, self.head_dim).transpose(1, 2)
        v = self.W_v(encoder).reshape(B, S, self.head_num, self.head_dim).transpose(1, 2)

        # 2. 每个目标位置都可以查看所有有效源位置，分数形状为 [B, H, T, S]。
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if encoder_padding_mask is not None:
            if encoder_padding_mask.dtype != torch.bool or encoder_padding_mask.shape != (B, S):
                raise ValueError("encoder_padding_mask 必须是形状为 [B, S] 的布尔张量")
            if encoder_padding_mask.all(dim=-1).any():
                raise ValueError("每条 encoder 序列至少需要一个有效 token")
            scores = scores.masked_fill(encoder_padding_mask[:, None, None, :], float("-inf"))

        # 3. 沿源序列维度 S 归一化，对 V 加权求和，再拼回 [B, T, E]。
        weights = torch.softmax(scores, dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(B, T, E)
        return self.W_o(context)


if __name__ == "__main__":
    attention = CrossAttention(embed_size=16, head_num=4)
    decoder = torch.randn(2, 3, 16)
    encoder = torch.randn(2, 5, 16)
    mask = torch.tensor([[False, False, False, True, True],
                         [False, False, False, False, True]])
    print(attention(decoder, encoder, mask).shape)  # torch.Size([2, 3, 16])
