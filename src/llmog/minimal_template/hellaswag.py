# Implmentation Refs : lm-evaluation-harness/lm-eval/tasks/hellaswag.py
import re
from typing import Any, List, Tuple

from .__base__ import MinimalTemplate


class HellaswagMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "hellaswag_minimal"

    def apply(self, example: Any) -> Tuple[str]:
        ctx = example["ctx_a"] + " " + example["ctx_b"].capitalize()
        demon = self.preprocess(example["activity_label"] + ": " + ctx)
        label = self.preprocess(example["endings"][int(example["label"])])
        return (demon, label)

    def get_answer_choices_list(self, example: Any) -> List[str]:
        return [self.preprocess(ending) for ending in example["endings"]]

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text
