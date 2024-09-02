import re
from typing import Any, Dict, List, Tuple

from .__base__ import MinimalTemplate


class XSumMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "xsum_minimal"

    def get_answer_choices_list(self, example: Any) -> List[str]:
        return example["summary"]

    def apply(self, example: Any) -> Tuple[str]:
        demon = f"Article: {example['document']}\nShort summary:"
        label = example["summary"]
        return (demon, label)
