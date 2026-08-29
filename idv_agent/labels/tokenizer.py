"""简易分词器 + 动作骨架词表。

- SimpleWordTokenizer：BC 阶段轻量文本分支用（word 级切分），预留 Qwen tokenizer
  兼容接口（encode/decode/pad_id/...）便于后期换真 tokenizer。
- SkeletonVocab：扫描 JSONL 收集所有 action_skeleton，FastController 用 nn.Embedding
  查表。空骨架固定映射 0 号。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional


_TOKEN_RE = re.compile(
    r"[A-Za-z]+|"
    r"\d+(?:\.\d+)?|"
    r"[\u4e00-\u9fff]|"
    r"[+\-_(),:.\[\]/]"
)


def _basic_tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


class SimpleWordTokenizer:
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"

    def __init__(self, vocab: Optional[list[str]] = None):
        specials = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self._tokens: list[str] = list(specials)
        self._token_to_id: dict[str, int] = {t: i for i, t in enumerate(self._tokens)}
        if vocab:
            for tok in vocab:
                if tok not in self._token_to_id:
                    self._token_to_id[tok] = len(self._tokens)
                    self._tokens.append(tok)

    @property
    def vocab_size(self) -> int:
        return len(self._tokens)

    @property
    def pad_id(self) -> int:
        return self._token_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self._token_to_id[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self._token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._token_to_id[self.EOS_TOKEN]

    def encode(self, text: str, add_special: bool = False, max_len: Optional[int] = None) -> list[int]:
        toks = _basic_tokenize(text)
        ids = []
        if add_special:
            ids.append(self.bos_id)
        for t in toks:
            ids.append(self._token_to_id.get(t, self.unk_id))
        if add_special:
            ids.append(self.eos_id)
        if max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        out = []
        for i in ids:
            t = self._tokens[i]
            if skip_special_tokens and t in (self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN):
                continue
            out.append(t)
        return "".join(out)

    def build_vocab_from_texts(self, texts: Iterable[str]) -> None:
        seen = set()
        for t in texts:
            for tok in _basic_tokenize(t):
                if tok not in self._token_to_id:
                    seen.add(tok)
        for tok in sorted(seen):
            self._token_to_id[tok] = len(self._tokens)
            self._tokens.append(tok)


_NONE_SKELETONS = {"", "(无)", "(无动作)", "NONE"}


class SkeletonVocab:
    """动作骨架词表。空骨架固定 0 号。"""

    def __init__(self, skeletons: Optional[list[str]] = None):
        self._id_to_sk: list[str] = ["(无)"]   # 0 = 空骨架
        self._sk_to_id: dict[str, int] = {"(无)": 0}
        if skeletons:
            for sk in skeletons:
                self.add(sk)

    def add(self, skeleton: str) -> int:
        sk = skeleton or "(无)"
        if sk in self._sk_to_id:
            return self._sk_to_id[sk]
        idx = len(self._id_to_sk)
        self._id_to_sk.append(sk)
        self._sk_to_id[sk] = idx
        return idx

    def encode(self, skeleton: str) -> int:
        sk = skeleton or "(无)"
        return self._sk_to_id.get(sk, 0)

    def __getitem__(self, idx: int) -> str:
        return self._id_to_sk[idx]

    def __len__(self) -> int:
        return len(self._id_to_sk)

    @classmethod
    def scan_jsonl(cls, jsonl_path: Path) -> "SkeletonVocab":
        vocab = cls()
        if not jsonl_path.exists():
            return vocab
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sk = obj.get("slow_system_output", {}).get("action_skeleton", "")
                vocab.add(sk)
        return vocab

    @classmethod
    def build_from_jsonls(cls, jsonl_paths) -> "SkeletonVocab":
        """合并多个 JSONL 的骨架为一张词表。"""
        vocab = cls()
        for jp in jsonl_paths:
            jp = Path(jp)
            if not jp.exists():
                continue
            with jp.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    sk = obj.get("slow_system_output", {}).get("action_skeleton", "")
                    vocab.add(sk)
        return vocab
