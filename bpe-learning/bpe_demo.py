from typing import List, Tuple, Optional
from collections import Counter, deque
# from functools import lru_cache

class BPETokenizerSimple:
	def __init__(self):
		self.idx2token = {}
		self.token2idx = {}
		self.bpe_merges = {}

	def train(self, text: str, vocab_size: int, allowed_special={"<|endoftext|>"}) -> None:
		"""
		训练 BPE 分词器
		每个实例仅支持训练一次
		Args:
			text: 训练文本
			vocab_size: 设定的词典大小
			allowed_special: 一组特殊词元
		Returns:
			None		
		"""
		# 预处理: 用'Ġ'替换空格, 'Ġ'是GPT-2 BPE实现中的特殊字符
		# 举个例子: 在GPT-2 BPE中, "Hello world" 可能被分为 ["Hello", "Ġworld"]
		# 而在GPT-4BPE 中, 则有可能被分为 ["Hello", " world"]
		processed_text = []
		for i, char in enumerate(text):
			if " " == char and 0 != i:
				processed_text.append("Ġ")
			if " " != char:
				processed_text.append(char)
		processed_text = "".join(processed_text)   # type: str

		# 预置 U+0000～U+00FF 的 256 个字符, 再补充训练文本中的其他字符
		unique_chars = [chr(i) for i in range(256)]
		unique_chars.extend(char for char in sorted(set(processed_text)) if char not in unique_chars)

		# 针对 text 是一个不含空格的字符串的情况(也挺怪异的)
		if 'Ġ' not in unique_chars:
			unique_chars.append('Ġ')

		self.idx2token = {idx: char for idx, char in enumerate(unique_chars)}
		self.token2idx = {char: idx for idx, char in self.idx2token.items()}

		# 特殊 token 更新进 self.idx2token 和 self.token2idx
		if allowed_special:
			for token in allowed_special:
				if token not in self.token2idx:
					new_id = len(self.idx2token)
					self.idx2token[new_id] = token
					self.token2idx[token] = new_id

		# 预处理后的文本转成 token id
		processed_text_token_ids = [self.token2idx[char] for char in processed_text]

		# BPE 算法实现: 反复查找并替换频次最高的 token_id pair
		for new_id in range(len(self.idx2token), vocab_size):
			if len(processed_text_token_ids) < 2:
				break
			# 找到频次最高的 token_id pair
			pair_id = self.find_freq_pair(processed_text_token_ids, mode='most') # type: tuple
			# 利用找到的 token_id pair 合并 processed_text_token_ids, 合并后的结果 merged_processed_text_token_ids
			merged_processed_text_token_ids = self.replace_pair(processed_text_token_ids, pair_id, new_id)

			processed_text_token_ids = merged_processed_text_token_ids
			self.bpe_merges[pair_id] = new_id

		# 需要注意的是, 按合并 ID 递增构建词表, 确保子 token 已存在
		# 按合并 id 递增构建词表，确保组成新 token 的 subtoken 已经存在
		# 使用 for (p0, p1), new_id in self.bpe_merges.items() 可能在 3.6 以下版本会出问题
		for (p0, p1), new_id in sorted(self.bpe_merges.items(), key=lambda item: item[1]):
			merged_token = self.idx2token[p0] + self.idx2token[p1]
			self.idx2token[new_id] = merged_token
			self.token2idx[merged_token] = new_id

	def encode(self, text: str) -> List[int]:
		"""
		把输入文本转化为 token id
		Args:
			text: 输入的文本
		Returns:
			List[int]: 文本对应的 token id 数组
		"""
		tokens = []
		words = text.split()

		# 把空格标识出来
		for i, word in enumerate(words):
			if i > 0 :
				tokens.append("Ġ" + word)
			else:
				tokens.append(word)

		token_ids = []
		for token in tokens:
			if token in self.token2idx:
				token_id = self.token2idx[token]
				token_ids.append(token_id)
			else:
				sub_token_ids = self.tokenize_with_bpe(token)
				token_ids.extend(sub_token_ids)
		return token_ids

	def tokenize_with_bpe(self, token: str) -> List[int]:
		"""
		按训练阶段记录的合并优先级, 编码预处理后的普通文本片段
		bpe = BPETokenizerSimple()
		bpe.train("bcXbcYbcZabWab", 260)
		# 学到: b+c -> 258, a+b -> 259

		bpe.encode("abc")
		# 如果仅从左往后合并, 会出现 ["ab", "c"]
		# 正确应为 ["a", "bc"], 因为 b+c 的优先级更高, 注意先合并的优先级高

		Args:
			token: 输入的字符串
		Returns:
			List[int]: token id list
		"""
		missing_chars = [char for char in token if char not in self.token2idx]
		if missing_chars:
			raise ValueError(f"Characters not found in vocab: {missing_chars}")
		token_ids = [self.token2idx[char] for char in token]

		while len(token_ids) > 1:
			# 只考虑当前相邻, 且存在已学习合并规则的 pair
			candidates = (pair for pair in zip(token_ids, token_ids[1:]) if pair in self.bpe_merges)

			# 新 id 按训练顺序递增, 因此最小 id 对应最高合并优先级
			best_pair = min(candidates, key=lambda pair: self.bpe_merges[pair], default=None)

			# 当前序列中已没有可应用的合并规则
			if best_pair is None:
				break

			# 从左到右, 替换该 pair 的所有非重叠出现位置
			new_id = self.bpe_merges[best_pair]
			token_ids = self.replace_pair(token_ids, best_pair, new_id)

		return token_ids
		


	def decode(self, token_ids: List[int]) -> str:
		"""
		把一组 token id list 转化为文本
		Args:
			token_ids: token id list
		Returns:
			str: token id list 对应的文本
		"""
		decoded_string = ""
		for token_id in token_ids:
			if token_id not in self.idx2token:
				raise ValueError(f"Token ID {token_id} not found in vocab.")
			token = self.idx2token[token_id]
			if token.startswith("Ġ"):
				tmp = " " + token[1:]
				decoded_string += tmp
			else:
				decoded_string += token
		return decoded_string


	@staticmethod
	def find_freq_pair(token_ids: List[int], mode="most") -> Optional[Tuple[int, int]]:
		"""
		找到频次最高/最低的token_id pair

		Args:
			token_ids: 接受的token id list
			mode: "most" 表示返回频次最高的 pair, "least" 表示返回频次最低的 pair
		Returns:
			Tuple[int, int]: 频次最高/最低的pair
		"""
		# 示例：
		# token_ids = [1, 2, 2, 3]
		# Counter(zip(token_ids, token_ids[1:]))
		# 结果：Counter({(1, 2): 1, (2, 2): 1, (2, 3): 1})
		pairs = Counter(zip(token_ids, token_ids[1:]))

		if "most" == mode:
			return max(pairs.items(), key=lambda x: x[1])[0]
		elif "least" == mode:
			return min(pairs.items(), key=lambda x: x[1])[0]
		else:
			raise ValueError("Invalid mode. Choose 'most' or 'least'.")


	@staticmethod
	def replace_pair(processed_text_token_ids: List[int], pair_id: Tuple[int, int], new_id: int) -> List[int]:
		"""
		给定pair_id, 使用 new_id, 在 processed_text_token_ids 中进行替换
		Args:
			processed_text_token_ids: 待替换的 token id 序列
			pair_id: 要替换的 pair id
			new_id: 替换后的新 token id
		Returns:
			List[int]: 替换指定 pair 后的新 token id 序列
		"""
		# 构建队列
		dq = deque(processed_text_token_ids)
		replaced = []

		while dq:
			current = dq.popleft() # 左侧弹出, dq长度 - 1
			if dq and (current, dq[0]) == pair_id:
				replaced.append(new_id)
				dq.popleft()
			else:
				replaced.append(current)
		return replaced


if __name__ == "__main__":

	path = "/Users/orangetwo/Downloads/LLMs-from-scratch-main/ch02/01_main-chapter-code/the-verdict.txt"

	with open(path, "r", encoding="utf-8") as f: # added ../01_main-chapter-code/
		text = f.read()

	tokenizer = BPETokenizerSimple()
	text = "a ba <|endoftext|> b"
	tokenizer.train(text, vocab_size=1000, allowed_special={"<|endoftext|>"})	
	print(len(tokenizer.token2idx))
	print(len(tokenizer.bpe_merges))
	input_text = text
	token_ids = tokenizer.encode(input_text)
	print(token_ids)
	print(tokenizer.decode(token_ids))
	# for token_id in token_ids:
	# 	print(f"{token_id} -> {tokenizer.decode([token_id])}")
