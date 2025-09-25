import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset

from data_types import MiniBatch
from tokenizer import Tokenizer

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


"""
示例数据如下:
```
Question: Given 1 2 3 4 and a target number 11. Show an expression that evaluates to 11.
Answer: 1 + (2 * 3) + 4
```
"""
class CountdownTasksDataset(Dataset):
    """Prepare Countdown Tasks for training"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        data = pd.read_parquet(Path(data_path) / "data")
        # use the last `test_size` examples for testing
        self.data = (
            data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    """
    示例数据：
    {   
        'target': 58, 
        'nums': [34,  2, 16, 80], 
        'prefix': '<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\nUsing the numbers [34  2 16 80], create an equation that equals 58. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>', 
        'prefix_tokens': ['<|im_start|>', 'system', 'Ċ', 'You', 'Ġare', 'Ġa', 'Ġhelpful', 'Ġassistant', '.', 'ĠYou', 'Ġfirst', 'Ġthink', 'Ġabout', 'Ġthe', 'Ġreasoning', 'Ġprocess', 'Ġin', 'Ġyour', 'Ġmind', 'Ġand', 'Ġthen', 'Ġprovide', 'Ġthe', 'Ġuser', 'Ġwith', 'Ġthe', 'Ġanswer', '.', '<|im_end|>', 'Ċ', '<|im_start|>', 'user', 'Ċ', 'Using', 'Ġthe', 'Ġnumbers', 'Ġ[', '3', '4', 'Ġ', 'Ġ', '2', 'Ġ', '1', '6', 'Ġ', '8', '0', '],', 'Ġcreate', 'Ġan', 'Ġequation', 'Ġthat', 'Ġequals', 'Ġ', '5', '8', '.', 'ĠYou', 'Ġcan', 'Ġuse', 'Ġbasic', 'Ġarithmetic', 'Ġoperations', 'Ġ(+', ',', 'Ġ-,', 'Ġ*,', 'Ġ/', ')', 'Ġand', 'Ġeach', 'Ġnumber', 'Ġcan', 'Ġonly', 'Ġbe', 'Ġused', 'Ġonce', '.', 'ĠShow', 'Ġyour', 'Ġwork', 'Ġin', 'Ġ<', 'think', '>', 'Ġ</', 'think', '>', 'Ġtags', '.', 'ĠAnd', 'Ġreturn', 'Ġthe', 'Ġfinal', 'Ġanswer', 'Ġin', 'Ġ<', 'answer', '>', 'Ġ</', 'answer', '>', 'Ġtags', ',', 'Ġfor', 'Ġexample', 'Ġ<', 'answer', '>', 'Ġ(', '1', 'Ġ+', 'Ġ', '2', ')', 'Ġ/', 'Ġ', '3', 'Ġ</', 'answer', '>.', '<|im_end|>', 'Ċ', '<|im_start|>', 'assistant', 'Ċ', 'Let', 'Ġme', 'Ġsolve', 'Ġthis', 'Ġstep', 'Ġby', 'Ġstep', '.Ċ', '<th', 'ink', '>'],
        'prefix_token_ids': [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 1446, 1156, 1744, 911, 279, 32711, 1882, 304, 697, 3971, 323, 1221, 3410, 279, 1196, 448, 279, 4226, 13, 151645, 198, 151644, 872, 198, 16429, 279, 5109, 508, 18, 19, 220, 220, 17, 220, 16, 21, 220, 23, 15, 1125, 1855, 458, 23606, 429, 16819, 220, 20, 23, 13, 1446, 646, 990, 6770, 34784, 7525, 17973, 11, 85922, 11777, 608, 8, 323, 1817, 1372, 646, 1172, 387, 1483, 3055, 13, 6928, 697, 975, 304, 366, 26865, 29, 690, 26865, 29, 9492, 13, 1597, 470, 279, 1590, 4226, 304, 366, 9217, 29, 690, 9217, 29, 9492, 11, 369, 3110, 366, 9217, 29, 320, 16, 488, 220, 17, 8, 608, 220, 18, 690, 9217, 14276, 151645, 198, 151644, 77091, 198, 10061, 752, 11625, 419, 3019, 553, 3019, 624, 13708, 766, 29]
    }
    """
    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        item.update(self.encode_prefix(item["nums"], item["target"]))
        return item

    def encode_prefix(self, numbers: List[int], target: int):
        """Prefix is the *actual* input to the model."""
        user_message = USER_TEMPLATE.format(numbers=numbers, target=target)
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        #print(f"prefix:\n {prefix}")
        """
        prefix的示例输出如下, 注意只给了个think的开头：

        <|im_start|>system
        You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.<|im_end|>
        <|im_start|>user
        Using the numbers [16 22 85], create an equation that equals 79. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>
        <|im_start|>assistant
        Let me solve this step by step.
        <think>
        """

        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix, # text
            "prefix_tokens": tokens.tokens, # token
            "prefix_token_ids": tokens.ids, # token id
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        numbers = [item["nums"] for item in batch]
        target = [item["target"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return MiniBatch(
            numbers=numbers,
            target=target,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    # Strip end token if present
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    """
    re.DOTALL 的作用是让点号 . 匹配包括换行符在内的所有字符。
    默认情况下，正则表达式的点号 . 匹配除换行符 \n 以外的任何字符。
    """
    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL) # 既有think,又有answer格式正确

    if full_format_match:
        return 1.0

    reward = 0.0

    if think_match:
        reward += 0.1

    if answer_match:
        reward += 0.5

    return reward


def answer_reward_function(
    response: str, numbers: List[int] = None, target: int = None
) -> float:
    """
    Checks if the answer uses all numbers exactly once and evaluates to the target
    """
    answer_regex = r"<answer>(.*?)<\/answer>"
    answer_match = re.search(answer_regex, response, re.DOTALL)
    # answer格式不对，返回0分
    if not answer_match:
        return 0.0
    # answer内容为空，返回0分
    answer_content = answer_match.group(1)
    if not answer_content:
        return 0.0
    # 只要数字 +-*/ 与 括号
    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # Check if the answer uses all numbers exactly once
    # 是否使用了不在numbers中的数字
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    """
    eval(answer_content, {"__builtins__": None}, {}) 中的三个参数：
    第一个参数：要执行的字符串表达式
    第二个参数：全局命名空间（globals）
    第三个参数：局部命名空间（locals）
    {"__builtins__": None} 的作用是清空内置命名空间

    危险的使用方式：
    python
    # 假设用户输入恶意代码
    malicious_code = "__import__('os').system('rm -rf /')"  # 删除系统文件！
    result = eval(malicious_code)  #  catastrophic! 
    """
    # Check if the answer evaluates to the target
    try:
        # 使用最严格的安全配置
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except:
        pass

    return 0.0


def reward_function(
    response: str,
    numbers: List[int] = None,
    target: int = None,
    end_token: str = None,
) -> Dict[str, Any]:
    """Reward function for Countdown Tasks.

    Total reward = 0.1 * format_reward + answer_reward
    """
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, numbers, target)
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }


if __name__ == "__main__":
    countdown_path = "/home/hkx/data/work/hf_data_and_model/datas/Jiayi-Pan/Countdown-Tasks-3to4/"
    tokenizer = Tokenizer(str( "qwen_tokenizer/tokenizer.json"))
    test_dataset = CountdownTasksDataset(
        data_path=countdown_path,
        tokenizer=tokenizer,
        split="test",
        test_size=100,
    )
    print(test_dataset[0])
