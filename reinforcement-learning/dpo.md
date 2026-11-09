# 从一对回答看懂 DPO 训练

给定同一个问题 `1+1=`，我们希望模型更偏好回答 `2`。同目录的 [dpo.py](./dpo.py) 用 PyTorch 实现了这个训练过程。入口 `train_dpo` 接收单设备上的 decoder-only 语言模型和已经分词的偏好样本，原地更新模型参数。

## 一对回答怎样组成 batch

示例用 `0` 表示 EOS，`1` 表示整个问题 `1+1=`，`2` 和 `3` 分别表示两个候选答案。一条训练样本写成这样。

```python
pair = {"prompt_ids": [1], "chosen_ids": [2, 0], "rejected_ids": [3, 0]}
```

`collate_pairs` 把同一份 prompt 分别接到两个回答前面，得到 `[1, 2, 0]` 和 `[1, 3, 0]`。一个 batch 有 `B` 对样本，前 `B` 行放 chosen，后 `B` 行按相同顺序放 rejected。这样每个模型只需做一次前向计算，再用 `.chunk(2)` 就能取出相互配对的分数。

对于真实对话，`prompt_ids` 包含模型开始生成前的完整上下文。两段回答都从各自的第一个回答 token 开始，按模型的模板带上回合结束符或 EOS。输入样本按原始长度提供，右侧填充由 `collate_pairs` 完成。

## 每条回答怎样得到一个分数

`response_logps` 计算整条回答的条件对数概率。对于 chosen，分数是 `log P(2 | 1+1=) + log P(EOS | 1+1=, 2)`。prompt 提供上下文，求和从第一个回答 token 开始。

假设同批还有更长的序列，chosen 这行需要在右侧补一个位置。即使 EOS 和 PAD 共用 ID `0`，按真实长度生成的掩码也能区分它们。

| 位置含义 | 问题 `1+1=` | 回答 `2` | EOS | PAD |
| --- | ---: | ---: | ---: | ---: |
| `input_ids` | 1 | 2 | 0 | 0 |
| `attention_mask` | 1 | 1 | 1 | 0 |
| `response_mask` | 0 | 1 | 1 | 0 |

问题位置的 logits 给 `2` 打分，`2` 位置的 logits 给 EOS 打分。因此 logits 取到倒数第二个位置，目标 token 和回答掩码都从第二个位置开始取。

```python
log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
targets = input_ids[:, 1:]
token_logps = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
return token_logps.masked_fill(~response_mask[:, 1:], 0.0).sum(dim=-1)
```

## DPO 的目标函数与参数更新

记 $x$ 为 prompt，$y_w$ 为 chosen，$y_l$ 为同一 prompt 下的 rejected。给定偏好数据集 $\mathcal{D}$ 和固定的参考模型 $\pi_{\mathrm{ref}}$，训练目标是调整策略模型 $\pi_\theta$ 的参数 $\theta$，最小化下面的期望损失。[DPO 原论文公式（7）](https://arxiv.org/html/2305.18290v3#S4.E7)

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
= -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[\log\sigma\!\left(
  \beta\left(
    \log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
    - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
  \right)
\right)\right].
$$

两个对数比分别对应代码中的 `chosen_gain` 和 `rejected_gain`，它们的差是 `margin`。令 $m$ 表示这个差值，$\beta>0$ 为缩放系数，$\sigma(z)=1/(1+e^{-z})$。偏好模型给 chosen 胜出的概率为 $\sigma(\beta m)$，所以单对样本的损失为 $-\log\sigma(\beta m)$。

`train_dpo` 在训练开始时复制一份 `policy`，冻结为 `reference`。两个模型都关闭 dropout。参考模型的前向计算使用 `torch.no_grad()`，策略模型的前向计算保留梯度。

```python
optimizer.zero_grad(set_to_none=True)
with torch.no_grad():
    ref_chosen, ref_rejected = response_logps(reference, *batch).chunk(2)
pi_chosen, pi_rejected = response_logps(policy, *batch).chunk(2)

chosen_gain = pi_chosen - ref_chosen
rejected_gain = pi_rejected - ref_rejected
margin = chosen_gain - rejected_gain
loss = -F.logsigmoid(beta * margin).mean()

loss.backward()
optimizer.step()
```

`.mean()` 用当前 batch 中偏好对的平均损失估计公式里的期望。`loss.backward()` 计算损失对 $\theta$ 的梯度，`optimizer.step()` 据此更新 `policy`。

## 运行示例并读懂输出

在这两个文件所在的目录运行以下命令。脚本自带一个词表大小为 4 的二元语言模型，可以离线运行。

```sh
python dpo.py
```

```text
step=1  loss=0.6931  margin=+0.0000
step=2  loss=0.6685  margin=+0.4998
step=3  loss=0.6445  margin=+0.9976
step=4  loss=0.6213  margin=+1.4918
step=5  loss=0.5990  margin=+1.9808
```

每行记录该步**参数更新前**的分数。第一行中两份模型相同，所以 `margin = 0`，`loss = ln(2)`。此时单样本损失对 `margin` 的导数是 `-beta / 2`，第一次反向传播就有学习信号。第二行中，`exp(0.4998) ≈ 1.65`，表示一次更新后，完整回答 `2, EOS` 相对于 `3, EOS` 的概率比变成了初始值的约 1.65 倍。
