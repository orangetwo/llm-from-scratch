
import re
from collections import Counter, deque
from typing import List, Optional, Tuple, Union


class BPETokenizerOpt:
    """从文本训练 BPE 分词器，并按训练时的合并优先级进行编码。"""

    def __init__(self):
        """初始化双向词表和合并规则。"""
        # 词元 ID 到字符串的映射，例如 {258: "ab"}。
        self.index2token = {}
        # 字符串到词元 ID 的映射，例如 {"ab": 258}。
        self.token2index = {}
        # 相邻词元的合并规则，例如 {(97, 98): 258}。
        self.bpe_merges = {}

    def train(
        self,
        text: str,
        vocab_size: int = 1_000,
        allowed_special={"<|endoftext|>"},
    ) -> None:
        """
        从文本训练 BPE 分词器，每个实例仅支持训练一次。

        参数：
            text: 训练文本。
            vocab_size: 目标词表大小；没有可合并的相邻词元时会提前结束。
            allowed_special: 需要加入初始词表的一组特殊词元。

        返回：
            无返回值，训练结果记录在双向词表和合并规则中。
        """
        # 先拆成普通文本片段、空格标记和换行字符，合并不跨越片段边界。
        tokens = self.pretokenize_text_opt(text)

        # 预置 U+0000～U+00FF 的 256 个字符，再补充训练文本中的其他字符。
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            char
            for char in sorted({char for token in tokens for char in token})
            if char not in unique_chars
        )

        # 即使训练文本没有空格，也保留空格标记。
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.index2token = {idx: char for idx, char in enumerate(unique_chars)}
        self.token2index = {char: idx for idx, char in self.index2token.items()}

        # 为特殊词元预留 ID；这里只登记词表，训练文本仍按普通字符拆分。
        if allowed_special:
            for token in allowed_special:
                if token not in self.token2index:
                    new_id = len(self.index2token)
                    self.index2token[new_id] = token
                    self.token2index[token] = new_id

        # 每个片段分别转成字符 ID 序列，保留二维结构以隔开合并边界。
        current_token_id_sequences = [
            [self.token2index[char] for char in token]
            for token in tokens
        ]

        # 反复选出频次最高的相邻词元对，替换后记录合并规则。
        for new_id in range(len(self.index2token), vocab_size):
            pair_id = self.find_freq_pair(current_token_id_sequences)
            if pair_id is None:
                break
            current_token_id_sequences = self.replace_pair(
                current_token_id_sequences, pair_id, new_id
            )
            self.bpe_merges[pair_id] = new_id

        # 按新 ID 递增构建词表，确保组成新词元的两个子词元已经存在。
        for (p0, p1), new_id in sorted(
            self.bpe_merges.items(), key=lambda item: item[1]
        ):
            merged_token = self.index2token[p0] + self.index2token[p1]
            self.index2token[new_id] = merged_token
            self.token2index[merged_token] = new_id

    def encode_opt(self, text: str, allowed_special=None) -> List[int]:
        """
        将文本编码为词元 ID 序列，并单独处理获准的特殊词元。

        参数：
            text: 待编码的文本。
            allowed_special: 本次允许使用的特殊词元集合。
                为 None 或空集合时，不允许使用词表中登记的特殊词元。

        返回：
            List[int]: 按原文顺序排列的词元 ID 列表。
        """
        # 统一把 None 视为空集合，便于后续检查和匹配。
        if allowed_special is None:
            allowed_special = set()

        # 先检查词表中以 <| 开头、以 |> 结尾的特殊词元是否获准使用。
        disallowed = [
            tok
            for tok in self.token2index
            if tok.startswith("<|")
            and tok.endswith("|>")
            and tok in text
            and tok not in allowed_special
        ]

        if disallowed:
            raise ValueError(
                f"Disallowed special tokens encountered in text: {disallowed}"
            )

        # 较长的特殊词元优先匹配；捕获组让拆分结果保留特殊词元本身。
        if allowed_special:
            special_pattern = (
                "("
                + "|".join(
                    re.escape(tok)
                    for tok in sorted(allowed_special, key=len, reverse=True)
                )
                + ")"
            )
            parts = re.split(special_pattern, text)
        else:
            parts = [text]

        # 按原文顺序处理各段，避免特殊词元与两侧的普通文本参与合并。
        token_ids = []
        for part in parts:
            if part in allowed_special:
                # 特殊词元直接查词表，整个字符串对应一个 ID。
                if part not in self.token2index:
                    raise ValueError(
                        f"Special token {part} not found in vocabulary."
                    )
                token_ids.append(self.token2index[part])
            else:
                # 普通文本先预切分；已有词元直接查表，其余片段再执行 BPE。
                for tok in self.pretokenize_text_opt(part):
                    if tok in self.token2index:
                        token_ids.append(self.token2index[tok])
                    else:
                        token_ids.extend(self.tokenize_with_bpe(tok))

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        按词表将词元 ID 序列转回文本，并还原空格标记。

        参数：
            token_ids: 待解码的词元 ID 列表。

        返回：
            str: 拼接各词元并还原空格标记后得到的文本。
        """
        out = []
        for token_id in token_ids:
            if token_id not in self.index2token:
                raise ValueError(f"Token ID {token_id} not found in vocab.")

            token = self.index2token[token_id]

            # 根据词元内容保留回车和换行，不依赖其他词表的固定编号。
            if token == "\n":
                out.append("\n")
            elif token == "\r":
                out.append("\r")
            elif token.startswith("Ġ"):
                # 独立的 Ġ 表示一个空格，Ġword 则表示空格加上 word。
                out.append(" " + token[1:])
            else:
                out.append(token)

        return "".join(out)

    def tokenize_with_bpe(self, token) -> List[int]:
        r"""
        按训练阶段记录的合并优先级，编码一个预切分后的普通文本片段。

        示例：
            训练文本为 "bc\nbc\nab"，目标词表大小为 260 时，
            先学到 b + c -> 258，再学到 a + b -> 259。
            编码 "abc" 应先合并 b、c，得到 ["a", "bc"]。
            如果仅从左往右选择可合并的词元对，则会得到 ["ab", "c"]。

        参数：
            token: 一个预切分后的普通文本片段。

        返回：
            List[int]: 应用已有合并规则后得到的词元 ID 列表。
        """
        # 先从单字符 ID 开始，后续只使用训练阶段学到的合并规则。
        token_ids = [self.token2index.get(char, None) for char in token]

        # 提前检查缺失字符，避免将无效 ID 带入合并过程。
        if None in token_ids:
            missing_chars = [
                char
                for char, token_id in zip(token, token_ids)
                if token_id is None
            ]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        while len(token_ids) > 1:
            # 只考虑当前相邻、且在合并表中存在的词元对。
            candidates = (
                pair
                for pair in zip(token_ids, token_ids[1:])
                if pair in self.bpe_merges
            )

            # 新 ID 随训练顺序递增，因此最小 ID 对应最早学到的合并规则。
            best_pair = min(
                candidates,
                key=lambda pair: self.bpe_merges[pair],
                default=None,
            )

            # 当前序列已经没有可应用的合并规则。
            if best_pair is None:
                break

            # 从左到右替换该词元对的所有非重叠出现位置，再重新寻找候选。
            new_id = self.bpe_merges[best_pair]
            token_ids = self.replace_pair(token_ids, best_pair, new_id)

        return token_ids

    @staticmethod
    def pretokenize_text_opt(text: str) -> List[str]:
        r"""
        预切分文本，保留普通空格、回车和换行，其他空白不保留。

        示例：
            "a  b\r\n" -> ["a", "Ġ", "Ġb", "\r", "\n"]
            一个空格可附在后面的词上，多余或末尾的空格用独立的 Ġ 表示。

        参数：
            text: 需要预切分的文本。

        返回：
            List[str]: 普通文本片段、空格标记和换行字符组成的列表。
        """
        tokens = []

        # 捕获组保留分隔符；连续的回车和换行会分别形成独立片段。
        for part in re.split(r"([\r\n])", text):
            if part in ("\r", "\n"):
                tokens.append(part)
            else:
                # 去掉紧跟普通空格的其他空白，让空格标记仍能附在后面的词上。
                part = re.sub(r" [^\S ]+", " ", part)
                # 先插入 Ġ 再拆分，连续普通空格因此不会被 split() 丢掉。
                tokens.extend(part.replace(" ", " Ġ").split())

        return tokens

    @staticmethod
    def find_freq_pair(
        token_id_sequences: List[List[int]],
        mode="most",
    ) -> Optional[Tuple[int, int]]:
        """
        统计各片段内部的相邻词元对，返回频次最高或最低的一对。

        参数：
            token_id_sequences: 多个片段对应的词元 ID 序列。
            mode: "most" 表示选择最高频，"least" 表示选择最低频。

        返回：
            Optional[Tuple[int, int]]: 选中的词元对，没有相邻词元时返回 None。
                频次相同时，返回在输入中最先出现的词元对。
        """
        # 示例：[[1, 2, 1], [1, 2]] 得到 {(1, 2): 2, (2, 1): 1}。
        # 每个片段单独取相邻元素，不统计两个片段交界处的词元对。
        pairs = Counter(
            pair
            for token_ids in token_id_sequences
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
    def replace_pair(
        current_token_id_sequences: Union[List[int], List[List[int]]],
        pair_id: Tuple[int, int],
        new_id: int,
    ) -> Union[List[int], List[List[int]]]:
        """
        将指定的相邻词元对替换成新 ID，支持一维和二维输入。

        参数：
            current_token_id_sequences: 单个词元 ID 序列，或多个片段的序列。
            pair_id: 要替换的两个相邻词元 ID。
            new_id: 合并后的新词元 ID。

        返回：
            Union[List[int], List[List[int]]]: 替换结果，保持输入的列表层级。
        """
        # 编码时传入一维序列，训练时传入二维序列；空列表按一维处理。
        is_1d = (
            not current_token_id_sequences
            or isinstance(current_token_id_sequences[0], int)
        )

        # 临时统一成二维列表，让训练和编码共用同一套替换逻辑。
        if is_1d:
            current_token_id_sequences = [current_token_id_sequences]

        replaced_sequences = []
        for token_ids in current_token_id_sequences:
            # 使用队列从左侧依次取出词元，不修改调用方传入的列表。
            dq = deque(token_ids)
            replaced = []
            while dq:
                current = dq.popleft()
                if dq and (current, dq[0]) == pair_id:
                    replaced.append(new_id)
                    # 第一个词元已弹出，再移除第二个，保证替换位置不重叠。
                    dq.popleft()
                else:
                    replaced.append(current)
            replaced_sequences.append(replaced)

        # 一维输入取出唯一的结果序列，二维输入保留外层列表。
        return replaced_sequences[0] if is_1d else replaced_sequences


if __name__ == "__main__":
    # 用包含连续空格、回车和换行的中文文本演示训练。
    text = (
        "温柔懒懒的海风    风海的懒懒柔温\r\r\n"
        "吹到高高的山峰    \n\r\r    温的风\r\n"
        "山的风\n吹到高高的山峰"
    )

    bpe = BPETokenizerOpt()
    bpe.train(text)
