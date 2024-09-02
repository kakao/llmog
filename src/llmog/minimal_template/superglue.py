# Implementation Refs : lm-evaluation-harness/lm-eval/tasks/superglue.py
# For Rte template : lm-evaluation-harness/lm-eval/tasks/glue.py

import re
from typing import Any, Dict, List, Tuple

from .__base__ import MinimalTemplate


def yesno(x):
    if x:
        return "yes"
    else:
        return "no"


def general_detokenize(string):
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')
    string = re.sub(r" (['.,])", r"\1", string)
    return string


class BoolqMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "boolq_minimal"

    def get_answer_choices_list(self, example: Any) -> List[str]:
        return ["no", "yes"]

    def apply(self, example: Any) -> Tuple[str]:
        demon = f"{example['passage']}\nQuestion: {example['question']}?\nAnswer:"
        label = yesno(example["label"])
        return (demon, label)


class CopaMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "copa_minimal"

    def get_answer_choices_list(self, example: Any) -> List[str]:
        return [example["choice1"], example["choice2"]]

    def apply(self, example: Any) -> Tuple[str]:
        connector = {
            "cause": "because",
            "effect": "therefore",
        }[example["question"]]
        demon = example["premise"].strip()[:-1] + f" {connector}"
        label = example["choice1"] if example["label"] == 0 else example["choice2"]
        return (demon, label)


class CbMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "cb_minimal"

    def get_answer_choices_list(self, example: Any) -> List[str]:
        return ["True", "False", "Neither"]

    def apply(self, example: Any) -> Tuple[str]:
        demon = "{}\nQuestion: {}. True, False or Neither?\nAnswer:".format(
            example["premise"],
            example["hypothesis"],
        )
        label = {0: "True", 1: "False", 2: "Neither"}[example["label"]]
        return (demon, label)


class WicMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "wic_minimal"

    def apply(self, example: Any) -> Tuple[str]:
        demon = (
            "Sentence 1: {}\nSentence 2: {}\nQuestion: Is the word '{}' used in the same way in the"
            " two sentences above?\nAnswer:".format(
                example["sentence1"],
                example["sentence2"],
                example["sentence1"][example["start1"] : example["end1"]],
            )
        )
        label = yesno(example["label"])
        return (demon, label)

    def get_answer_choices_list(self, example: Any) -> List[str]:
        return ["no", "yes"]


class WscMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "wsc_minimal"

    def apply(self, example: Any) -> Tuple[str]:
        raw_passage = example["text"]
        # NOTE: HuggingFace span indices are word-based not character-based.
        pre = " ".join(raw_passage.split()[: example["span2_index"]])
        post = raw_passage[len(pre) + len(example["span2_text"]) + 1 :]
        passage = general_detokenize(pre + " *{}*".format(example["span2_text"]) + post)
        noun = example["span1_text"]
        pronoun = example["span2_text"]
        demon = (
            f"Passage: {passage}\n"
            + f'Question: In the passage above, does the pronoun "*{pronoun}*" refer to "*{noun}*"?\n'
            + "Answer:"
        )
        label = yesno(example["label"])
        return (demon, label)

    def get_answer_choices_list(self, example: Any) -> List[str]:
        return ["no", "yes"]


class RteMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "rte_minimal"

    def apply(self, example: Any) -> Tuple[str]:
        demon = "{}\nQuestion: {} True or False?\nAnswer:".format(
            example["premise"],
            example["hypothesis"],
        )
        label = {0: "True", 1: "False"}[example["label"]]
        return (demon, label)

    def get_answer_choices_list(self, example: Any) -> List[str]:
        return ["True", "False"]


class MultircMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "multirc_minimal"

    def apply(self, example: Any) -> Tuple[str]:
        demon = f"{example['paragraph']}\nQuestion: {example['question']}\nAnswer: {example['answer']}\nIs the answer correct?"
        label = yesno(example["label"])
        return (demon, label)

    def get_answer_choices_list(self, example: Any) -> List[str]:
        return ["no", "yes"]


class RecordMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "record_minimal"

    def _process_doc(self, example: Any) -> Dict[str, Any]:
        return {
            "passage": example["passage"],
            "query": example["query"],
            "entities": sorted(list(set(example["entities"]))),
            "answers": sorted(list(set(example["answers"]))),
        }

    def _format_ans(self, query, subs, repl):
        return f"  - {query}".replace(subs, repl)

    def apply(self, example: Any) -> Tuple[str]:
        doc = self._process_doc(example)
        initial_text, *highlights = doc["passage"].strip().split("\n@highlight\n")
        demon = initial_text + "\n\n"
        for highlight in highlights:
            demon += f"  - {highlight}.\n"
        labels = [self._format_ans(doc["query"], "@placeholder", ans) for ans in doc["answers"]]
        return (demon, labels)

    def get_answer_choices_list(self, example: Any) -> List[str]:
        doc = self._process_doc(example)
        return [self._format_ans(doc["query"], "@placeholder", ent) for ent in doc["entities"]]
