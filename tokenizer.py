import json
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment
from tokenizers import Encoding
# 依赖于tokenizer包
from tokenizers import Tokenizer as TokenizerBase

"""
Tokenizers
Provides an implementation of today's most used tokenizers, with a focus on performance and versatility.

Bindings over the Rust implementation. If you are interested in the High-level design, you can go check it there.

Otherwise, let's dive in!

Main features:
Train new vocabularies and tokenize using 4 pre-made tokenizers (Bert WordPiece and the 3 most common BPE versions).
Extremely fast (both training and tokenization), thanks to the Rust implementation. Takes less than 20 seconds to tokenize a GB of text on a server's CPU.
Easy to use, but also extremely versatile.
Designed for research and production.
Normalization comes with alignments tracking. It's always possible to get the part of the original sentence that corresponds to a given token.
Does all the pre-processing: Truncate, Pad, add the special tokens your model needs.
"""

class Tokenizer:
    """Tokenizer with chat template supported using jinja2 engine"""

    def __init__(self, tokenizer_path: str):
        super().__init__()
        tokenizer_config_path = Path(tokenizer_path).parent / "tokenizer_config.json"
        self.tokenizer_config = json.load(open(tokenizer_config_path))
        self.tokenizer = TokenizerBase.from_file(tokenizer_path)
        self.chat_template = Environment().from_string(
            self.tokenizer_config["chat_template"]
        )
        self.eos_token = self.tokenizer_config["eos_token"]
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
        self.pad_token = self.tokenizer_config["pad_token"]
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)

    def encode_chat(self, messages: List[Dict[str, str]]) -> str:
        return self.chat_template.render(messages=messages, add_generation_prompt=True)

    def encode_chat_with_response_prompt(
        self, messages: List[Dict[str, str]], prompt: str
    ) -> str:
        return self.encode_chat(messages) + prompt

    def tokenize(self, text: str) -> Encoding:
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
