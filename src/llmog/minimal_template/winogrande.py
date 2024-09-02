# Implementation Refs : lm-evaluation-harness/lm-eval/tasks/winogrande.py
from typing import Any, List, Tuple

from .__base__ import MinimalTemplate


class WinograndeMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "winogrande_minimal"
        self.ans_str = ["1", "2"]

    def apply(self, example: Any) -> Tuple[str]:
        demon = self.partial_context(example)
        label = self.partial_target(example, example["option" + example["answer"]])
        return (demon, label)

    def get_answer_choices_list(self, example: Any) -> List[str]:
        options = ["option" + ans for ans in self.ans_str]
        return [self.partial_target(example, example[option]) for option in options]

    def get_ctx_choices_list(self, example: Any) -> List[str]:
        return [self.partial_context(example, example["option" + ans]) for ans in self.ans_str]

    @classmethod
    def partial_context(cls, doc):
        # Substitute the pronoun in the sentence without the specified option
        # and ignore everything after.
        pronoun_loc = doc["sentence"].index("_")
        return doc["sentence"][:pronoun_loc]

    @classmethod
    def partial_target(cls, doc, option):
        # The target is everything after the document specified pronoun.
        # with the specified option
        pronoun_loc = doc["sentence"].index("_") + 1
        return option + " " + doc["sentence"][pronoun_loc:].strip()
