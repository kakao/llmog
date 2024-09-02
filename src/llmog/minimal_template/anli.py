# Implementation Refs : lm-evaluation-harness/lm-eval/tasks/anli.py
from typing import Any, List, Tuple

from .__base__ import MinimalTemplate


class AnliMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "anli_minimal"

    def apply(self, example: Any) -> Tuple[str]:
        # OA does this a bit weirdly: they prepend "anli 1:  anli 1:  " to the beginning
        # of the prompt (yes, repeating it!). also, " True, False, or Neither?" is directly
        # appended onto the question, with no "Answer:" or even a newline. Do we *really*
        # want to do it exactly as OA did?
        demon = example["premise"] + "\nQuestion: " + example["hypothesis"] + " True, False, or Neither?\nAnswer:"
        label = ["True", "Neither", "False"][example["label"]]
        return (demon, label)

    def get_answer_choices_list(self, example: Any) -> List[str]:
        return ["True", "Neither", "False"]
