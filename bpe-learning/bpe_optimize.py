from collections import Counter, deque
from typing import Tuple, List, Optional, Union
import re
import json


class BPETokenizerOpt:
	def __init__(self):
		self.index2token = {}
		self.token2index = {}
		self.bpe_merges = {}

	def train(self, text: str, vocab_size: int = 1_000, allowed_special={"<|endoftext|>"}) -> None:

		tokens = self.pretokenize_text_opt(text)

		unique_chars = [chr(i) for i in range(256)]
		unique_chars.extend(
			char for char in sorted({char for token in tokens for char in token})
			if char not in unique_chars
		)

		if "Ġ" not in unique_chars:
			unique_chars.append("Ġ")

		self.index2token = {idx: char for idx, char in enumerate(unique_chars)}
		self.token2index = {char: idx for idx, char in self.index2token.items()}

		if allowed_special:
			for token in allowed_special:
				if token not in self.token2index:
					new_id = len(self.index2token)
					self.index2token[new_id] = token
					self.token2index[token] = new_id

		current_token_id_sequences = [
			[self.token2index[char] for char in token]
			for token in tokens
		]

		# 不断合并频次最高的 pair id， 并且记录每次合并的 pair id
		for new_id in range(len(self.index2token), vocab_size):
			pair_id = self.find_freq_pair(current_token_id_sequences)
			if pair_id is None:
				break
			current_token_id_sequences = self.replace_pair(current_token_id_sequences, pair_id, new_id)
			self.bpe_merges[pair_id] = new_id
 
		for (p0, p1), new_id in sorted(self.bpe_merges.items(), key=lambda item: item[1]):
			merged_token = self.index2token[p0] + self.index2token[p1]
			self.index2token[new_id] = merged_token
			self.token2index[merged_token] = new_id


	def encode_opt(self, text: str, allowed_special=None) -> List[int]:

		# None 和空集合都表示: 不允许任何特殊 token
		if allowed_special is None:
			allowed_special = set()

		# 1. 检查文本中是否包含未获准的特殊 token。
		disallowed = [
			tok for tok in self.token2index
			if tok.startswith("<|") and tok.endswith("|>")
			and tok in text and tok not in allowed_special
		]

		if disallowed:
			raise ValueError(
            f"Disallowed special tokens encountered in text: {disallowed}"
        )

		# 2. 拆分文本, 保留匹配到的特殊 token
		if allowed_special:
			special_pattern = "(" + "|".join(
				re.escape(tok)
				for tok in sorted(allowed_special, key=len, reverse=True)
        	) + ")"

			parts = re.split(special_pattern, text)
		else:
			parts = [text]

		# 3. 按原文顺序编码每一段
		token_ids = []
		for part in parts:
			if part in allowed_special:
				# 特殊 token：直接查词表，加入一个 ID。
				if part not in self.token2index:
					raise ValueError(
                    	f"Special token {part} not found in vocabulary."
                	)
				token_ids.append(self.token2index[part])
			else:
				# 普通文本：先预切分，再查词表或执行 BPE。
				for tok in self.pretokenize_text_opt(part):
					if tok in self.token2index:
						token_ids.append(self.token2index[tok])
					else:
						token_ids.extend(self.tokenize_with_bpe(tok))

		return token_ids
		


	def decode(self, token_ids:List[int]) -> str:
		out = []
		for token_id in token_ids:
			if token_id not in self.index2token:
				raise ValueError(f"Token ID {token_id} not found in vocab.")

			token = self.index2token[token_id]

			if token == "\n":
				out.append("\n")
			elif token == "\r":
				out.append("\r")
			elif token.startswith("Ġ"):
				out.append(" " + token[1:])
			else:
				out.append(token)

		return "".join(out)


	def tokenize_with_bpe(self, token) -> List[int]:

		
		token_ids = [self.token2index.get(char, None) for char in token]

		# 处理 字符 异常情况
		if None in token_ids:
			missing_chars = [char for char, token_id in zip(token, token_ids) if token_id is None]
			raise ValueError(f"Characters not found in vocab: {missing_chars}")


		while len(token_ids) > 1:
			candidates = (pair for pair in zip(token_ids, token_ids[1:]) if pair in self.bpe_merges)

			best_pair = min(candidates, key=lambda pair: self.bpe_merges[pair], default=None)

			if best_pair is None:
				break
			
			new_id = self.bpe_merges[best_pair]
			token_ids = self.replace_pair(token_ids, best_pair, new_id)

		return token_ids


	@staticmethod
	def pretokenize_text_opt(text: str) -> List[str]:
		tokens = []

		for part in re.split(r"([\r\n])", text):
			if part in ("\r", "\n"):
				tokens.append(part)
			else:
				part = re.sub(r" [^\S ]+", " ", part)
				tokens.extend(part.replace(" ", " Ġ").split())

		return tokens


	@staticmethod
	def find_freq_pair(token_id_sequences: List[List[int]], mode='most') -> Optional[Tuple[int, int]]:

		pairs = Counter(
			pair for token_ids in token_id_sequences
			for pair in zip(token_ids, token_ids[1:])
		)

		if not pairs:
			return None

		if mode == "most":
			return max(pairs.items(), key=lambda x: x[1])[0]
		elif mode == "least":
			return min(pairs.items(), key=lambda x: x[1])[0]
		else:
			raise ValueError("Invalid mode. Choose 'most' or 'least'.")

		
	@staticmethod
	def replace_pair(current_token_id_sequences: Union[List[int], List[List[int]]], pair_id: Tuple[int, int], new_id: int) -> Union[List[int], List[List[int]]]:

		# 判断输入是否为一维列表
		is_1d = (
			not current_token_id_sequences
			or isinstance(current_token_id_sequences[0], int)
		)

		 # 统一转换为二维列表
		if is_1d:
			current_token_id_sequences = [current_token_id_sequences]

		replaced_sequences = []
		for token_ids in current_token_id_sequences:
			dq = deque(token_ids)

			replaced = []
			while dq:
				current = dq.popleft()
				if dq and (current, dq[0]) == pair_id:
					replaced.append(new_id)
					dq.popleft()
				else:
					replaced.append(current)
			replaced_sequences.append(replaced)

		# 根据原始输入类型返回
		return replaced_sequences[0] if is_1d else replaced_sequences

	

def lists_equal(a: List[str], b: List[str]) -> bool:
    return a == b


if __name__ == "__main__":
	text = "温柔懒懒的海风    风海的懒懒柔温\r\r\n吹到高高的山峰    \n\r\r    温的风\r\n山的风\n吹到高高的山峰"
	# text = "hello \tworld"

	bpe = BPETokenizerOpt()
	# res1 = bpe.pretokenize_text(text)
	# print(res1)

	# res2 = bpe.pretokenize_text_opt(text)
	# print(res2)

	bpe.train(text)
