import json
import os
import random
import re
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer


def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def camelToSnake(s):
    underscorer1 = re.compile(r"(.)([A-Z][a-z]+)")
    underscorer2 = re.compile("([a-z])([A-Z0-9])")
    return underscorer2.sub(r"\1_\2", underscorer1.sub(r"\1_\2", s)).lower()


def all_identical_elems(elements: Iterable) -> bool:
    assert len(elements) != 1, "Unnecessary checking for single elem iterable"
    return len(set(elements)) == 1


def get_train_valid_dataset(dataset_name, subtask_name, train_path, valid_path) -> Tuple[Dataset]:
    if dataset_name in ["super_glue", "hellaswag", "winogrande"]:
        if subtask_name in ["axb", "axg"]:
            raise NotImplementedError()
        dataset = load_dataset(dataset_name, subtask_name, num_proc=os.cpu_count() // 2)
        return (dataset["train"], dataset["validation"])
    elif dataset_name.startswith("anli"):
        key_postfix = f"_r{dataset_name[-1]}"
        dataset = load_dataset(dataset_name[:-1], num_proc=os.cpu_count() // 2)
        return (dataset[f"train{key_postfix}"], dataset[f"dev{key_postfix}"])
    elif dataset_name.startswith("story_cloze"):
        if train_path == None or valid_path == None:
            raise ValueError(
                "The story_cloze task requires manual data. You can download it by filling out the Google form. (https://goo.gl/forms/aQz39sdDrO)"
            )
        dataset = load_dataset(
            "csv", data_files={"train": train_path, "validation": valid_path}, num_proc=os.cpu_count() // 2
        )
        dataset = dataset.rename_columns(
            {column_name: camelToSnake(column_name) for column_name in dataset["validation"].column_names}
        )
        dataset = dataset.rename_columns(
            {"random_fifth_sentence_quiz_1": "sentence_quiz1", "random_fifth_sentence_quiz_2": "sentence_quiz2"}
        )
        return (dataset["train"], dataset["validation"])
    elif dataset_name == "enriched_web_nlg":
        dataset = load_dataset("enriched_web_nlg", "en", num_proc=os.cpu_count() // 2)
        return (dataset["dev"], dataset["test"])
    elif dataset_name in ["xsum", "e2e_nlg_cleaned"]:
        dataset = load_dataset(dataset_name, subtask_name, num_proc=os.cpu_count() // 2)
        return (dataset["validation"], dataset["test"])
    else:
        raise NotImplementedError()


def reshape_tensor_for_decoding(sample: torch.Tensor, replace_id: int):
    if sample.dim() == 3:
        return torch.where(sample < 0, replace_id, sample).squeeze()
    else:
        return torch.where(sample < 0, replace_id, sample)


def logging_decoded_samples(tokenizer: PreTrainedTokenizer, *samples):
    for sample in samples:
        if isinstance(sample[0], list):
            for s in sample:
                print(tokenizer.batch_decode(reshape_tensor_for_decoding(s, tokenizer.pad_token_id)))
        else:
            print(tokenizer.batch_decode(reshape_tensor_for_decoding(sample, tokenizer.pad_token_id)))


def write_results(
    results: Dict[str, float],
    template_name: str,
    args: Dict[str, Any],
    generation_kwargs: Optional[Dict[str, Any]] = None,
):

    results_dict = dict()
    results_dict["template_name"] = template_name
    results_dict["args"] = vars(args)
    if generation_kwargs is not None:
        write_gen_kwargs = dict()
        for key, value in generation_kwargs.items():
            if value is not None:
                write_gen_kwargs[key] = value
        results_dict["generation_hp"] = write_gen_kwargs
    results_dict["results"] = results
    json_string = json.dumps(results_dict, indent=4, ensure_ascii=False)

    with open(args.output_file, "a") as f:
        f.write(json_string)


def get_element_indices(lst, elements):
    return [i for i, x in enumerate(lst) if x in elements]
