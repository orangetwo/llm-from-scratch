"""面向单设备、decoder-only 语言模型的 DPO 教学实现。直接运行本文件即可看示例。"""


from collections.abc import Iterable, Sequence
from copy import deepcopy
from functools import partial
from typing import Literal, TypedDict, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class PreferencePair(TypedDict):
	"""一条样本是一个字典。每个 token 用它在模型词表里的整数 ID 表示。"""
	
	# 生成前已知的上下文，同一对回答共享
	prompt_ids: Sequence[int]
	# 标注者更偏好的回答，从第一个回答 token 到结束符
	chosen_ids: Sequence[int]
	# 同一问题下用于比较的另一条回答，也含结束符
	rejected_ids: Sequence[int]


class StepMetrics(TypedDict):
	"""每个训练 step 的指标。"""
	
	# 本批偏好对的平均损失
	loss: float
	# 与训练起点相比，chosen 对 rejected 的对数优势平均增加了多少
	margin: float

Batch = tuple[Tensor, Tensor, Tensor]  # input_ids, attention_mask, response_mask

def collate_pairs(pairs: Sequence[PreferencePair], *, pad_token_id: int) -> Batch:
	"""
	把 B 对样本整理成三个 [2B, T] 张量, T 是本批中最长的 token 序列长度。padding 用 pad_token_id 填充。

	input_ids 为 torch.long, 保存实际送入模型的 token ID
	attention_mask 标记可作为上下文的有效位置, response_mask 标记计入回答分数的位置
	"""

	sequences: list[Tensor] = []
	prompt_lengths: list[int] = []

	# 假定 一批数据包含 A,B 两个问题, 那包装为 4 个序列, 具体为:
	# [prompt_A + chosen_A, prompt_B + chosen_B, prompt_A + rejected_A, prompt_B + rejected_B]
	answer_keys = ("chosen_ids", "rejected_ids")

	for answer_key in answer_keys:
		for pair in pairs:
			prompt, answer = pair["prompt_ids"], pair[answer_key]
			if len(prompt) == 0 or len(answer) == 0:
				raise ValueError("prompt_ids 和 chosen/rejected_ids 都不能为空")
			# 假定 prompt 序列为 [1] 拼上 answer 序列 [2, 0] 得到 [1, 2, 0]。记下 prompt 长度 1，
            # 就知道这行从下标 1 开始才是要打分的回答。
			# [*A, *B] 是 Python 3.5+ 的语法糖, 相当于 A + B, 但更通用, 即使 A 和 B 非同一类型
			sequences.append(torch.tensor([*prompt, *answer], dtype=torch.long))
			prompt_lengths.append(len(prompt))

	# 短序列右侧补 PAD，让所有行等长。batch_first=True 让第一维表示行，第二维表示位置。
	input_ids = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
	# positions 形状为 [1, T]，lengths 为 [2B, 1]。比较时 PyTorch 会自动广播，
    # 相当于把同一排位置编号分别与每行自己的真实长度比较。
	positions = torch.arange(input_ids.size(1))[None, :]
	lengths = torch.tensor([len(seq) for seq in sequences])[:, None]
    # 假设示例中的一行被补到四格，EOS（回答结束）和 PAD（补齐）恰好都用 ID 0。
    # input_ids       [1, 2, 0, 0]    问题、回答、EOS、PAD
    # attention_mask  [1, 1, 1, 0]    按真实长度保留前三格
    # response_mask   [0, 1, 1, 0]    再按 prompt 长度选中回答和 EOS
	attention_mask = positions < lengths
	response_mask = attention_mask & (positions >= torch.tensor(prompt_lengths)[:, None])
	return input_ids, attention_mask, response_mask


def response_logprobs(
		model: nn.Module, 
        input_ids: Tensor, 
		attention_mask: Tensor, 
		response_mask: Tensor
	) -> Tensor:
	"""计算每条回答的条件对数概率 log P(answer | prompt)，每行返回一个数，形状为 [2B]。

    模型输出的 logits 形状为 [2B, T, V], V 是词表大小。logits[i, t, v] 表示
    模型读到第 i 样本位置 t 时，给“下一个 token 是 v”的原始分数。
    """
	
	# 样本已给出完整回答，因果模型可以一次前向为每个前缀预测下一个 token。
    # use_cache=False 省去逐步生成时才会复用的键值缓存。
	logits: Tensor = model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).logits

	# logits 去掉末位，target ID 对应 input ID 去掉首位。log_softmax 沿词表维把原始分数
    # 转成对数概率，float() 让这一步使用 32 位浮点数。
	log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
	targets = input_ids[:, 1:]
    # gather 从每个位置的 V 个候选分数中取出真实目标的那一个。例如目标 ID 为 2，
    # 就取词表下标 2 的值。unsqueeze 补上取值所需的索引维，squeeze 去掉该维，
    # 得到 [2B, T-1]，每格都是对应目标 token 的对数概率。
	token_logps = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # 掩码也跟随目标取 [:, 1:]。把 prompt 和 PAD 对应的分数置零，只累加回答及 EOS。
    # 各步条件概率相乘得到整条回答的概率，取对数后就能用求和来算。在示例中得到
    # log P(2 | 1+1=) + log P(EOS | 1+1=, 2)。
	return token_logps.masked_fill(~response_mask[:, 1:], 0.0).sum(dim=-1)


def train_dpo(
		policy: nn.Module,
		pairs: Sequence[PreferencePair],
		pad_token_id: int,
		beta: float = 0.1,
		lr: float = 1e-6,
		batch_size: int = 4,
		epochs: int = 1
	) -> list[StepMetrics]:

	if beta <= 0 or epochs < 1:
		raise ValueError("beta 必须为正, epochs 至少为 1")

    # DataLoader 以“对”为单位打乱并取样，两条回答始终一起交给 collate_pairs。
    # partial 预先填好补齐用的 ID。cast 只给类型检查器提示，运行时使用传入的样本序列。
	loader: Iterable[Batch] = DataLoader(
        cast(Dataset[PreferencePair], pairs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_pairs, pad_token_id=pad_token_id),
    )

	device = next(policy.parameters()).device

	# 在第一次更新前复制一份独立权重，作为整个训练的比较基准。
    # reference 负责回答“训练起点给这条回答多少分”，优化器只接收 policy 的参数。
	reference = deepcopy(policy).eval()
	reference.requires_grad_(False)
	optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)

	# eval() 关闭 dropout 的随机性，让分数可以直接比较。PyTorch 单独控制梯度记录，
    # 因此处在 eval 模式的 policy 仍然可以通过 backward() 学习。
	# tips: eval() 不会影响 layerNorm 参数的更新
	original_modes = [(module, module.training) for module in policy.modules()]
	policy.eval()
	history: list[StepMetrics] = []

	try:
		for _ in range(epochs):
			for input_ids, attention_mask, response_mask in loader:
                # 把 ID 和两个掩码一起搬到模型所在设备，供两份模型使用相同的输入。
				batch = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    response_mask.to(device),
                )

				# PyTorch 默认累加梯度。每批各做一次更新，所以先清掉上一批的梯度。
				optimizer.zero_grad()
				# *batch 展开三个张量。一次打分得到 [2B]，chunk(2) 切出两个 [B]。
                # 参考分数是固定基准，no_grad() 跳过梯度记录。policy 的打分保留计算图。
				# 显式声明 no_grad : ① 防御性编程 ② 语义清晰
				with torch.no_grad():
					ref_chosen, ref_rejected = response_logprobs(reference, *batch).chunk(2)
				pi_chosen, pi_rejected = response_logprobs(policy, *batch).chunk(2)

				# 相减的两个数都是对数概率，因此 gain = log(P_policy / P_reference)。
                # chosen 和 rejected 各有一个 gain，按同一问题逐对相减。
				chosen_gain = pi_chosen - ref_chosen
				rejected_gain = pi_rejected - ref_rejected

				# margin > 0 表示策略模型给 chosen 与 rejected 的概率比高于训练起点。
				margin = chosen_gain - rejected_gain
				# sigmoid(beta * margin) 是 DPO 预测这一对中 chosen 胜出的概率。
                # logsigmoid 稳定地计算该概率的对数。取负后，预测概率越高，单对损失越小。
                # 最后的 mean() 在这批 B 对回答之间取平均。
				loss = -F.logsigmoid(beta * margin).mean()

				# backward() 将损失对各个 policy 参数的梯度写入 .grad，step() 据此更新权重。
				loss.backward()
				optimizer.step()
				# 记录的是本步前向时的值。detach() 把指标与计算图分开，item() 取出普通数值。
				history.append(
                    {"loss": loss.item(), "margin": margin.detach().mean().item()}
                )

	finally:
		# 把模型切回调用者原先使用的训练或评估模式。
		for module, was_training in original_modes:
			if module.training != was_training:
				module.train(was_training)

	return history


if __name__ == "__main__":
    from types import SimpleNamespace

    class BigramLM(nn.Module):
        """示例用二元语言模型，表的每行存放给定当前 token 时下一 token 的 logits。"""

        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            # 词表大小为 4 时，这是一张 4×4 的可训练表。行号是当前 token 的 ID，
            # 列号是下一 token 的 ID。初始分数全为 0，所以每行四个候选的概率都是 1/4。
            self.table = nn.Embedding(vocab_size, vocab_size)
            nn.init.zeros_(self.table.weight)

        def forward(
            self, input_ids: Tensor, attention_mask: Tensor, use_cache: bool = False
        ) -> SimpleNamespace:
            # 对每个输入 ID 查一行表，就得到 [行数, 序列长度, 词表大小] 的 logits。
            # 放到 .logits 字段后，这个小模型就能复用上面的训练函数。
            return SimpleNamespace(logits=self.table(input_ids))

    # 词表用 0、1、2、3 分别表示 <eos>、"1+1="、"2"、"3"。样本偏好回答 "2"。
    pairs: list[PreferencePair] = [
        {"prompt_ids": [1], "chosen_ids": [2, 0], "rejected_ids": [3, 0]}
    ]
    policy = BigramLM(vocab_size=4)
    # 这里只有一对回答，每轮正好更新一次。示例用 lr=0.1，让五步之内的变化容易观察。
    history = train_dpo(policy, pairs, pad_token_id=0, epochs=5, lr=0.1)
    # 第一行记录初始模型。两份权重相同，margin 为 0，loss 为 -log(0.5) ≈ 0.6931。
    # 第二行就能看到第一次参数更新后，同一对回答的偏好分差。
    for step, row in enumerate(history, start=1):
        print(f"step={step}  loss={row['loss']:.4f}  margin={row['margin']:+.4f}")