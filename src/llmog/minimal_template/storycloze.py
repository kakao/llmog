# Implementation Refs : lm-evaluation-harness/lm-eval/tasks/storycloze.py
from typing import Any, List, Tuple

from .__base__ import MinimalTemplate


class StoryclozeMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "storycloze_minimal"

    def apply(self, example: Any) -> Tuple[str]:
        demon = " ".join(
            [
                example["input_sentence_1"],
                example["input_sentence_2"],
                example["input_sentence_3"],
                example["input_sentence_4"],
            ]
        )
        clozes = [example["sentence_quiz1"], example["sentence_quiz2"]]
        label = clozes[example["answer_right_ending"] - 1]
        return (demon, label)

    def get_answer_choices_list(self, example: Any) -> List[str]:
        return [example["sentence_quiz1"], example["sentence_quiz2"]]
