from functools import partial
from typing import Dict, List, Optional, Tuple, Union

from datasets import Dataset
from promptsource.templates import DatasetTemplates, Template
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .minimal_template.__base__ import MinimalTemplate
from .minimal_template.anli import AnliMinimal
from .minimal_template.hellaswag import HellaswagMinimal
from .minimal_template.storycloze import StoryclozeMinimal
from .minimal_template.superglue import (
    BoolqMinimal,
    CbMinimal,
    CopaMinimal,
    MultircMinimal,
    RecordMinimal,
    RteMinimal,
    WicMinimal,
    WscMinimal,
)
from .minimal_template.winogrande import WinograndeMinimal
from .minimal_template.xsum import XSumMinimal
from .utils import all_identical_elems, get_element_indices

MINIMAL_TEMPLATE_MAPPER = {
    "super_glue-boolq": BoolqMinimal,
    "super_glue-copa": CopaMinimal,
    "super_glue-cb": CbMinimal,
    "super_glue-rte": RteMinimal,
    "super_glue-wic": WicMinimal,
    "super_glue-wsc.fixed": WscMinimal,
    "super_glue-multirc": MultircMinimal,
    "super_glue-record": RecordMinimal,
    "anli1": AnliMinimal,
    "anli2": AnliMinimal,
    "anli3": AnliMinimal,
    "hellaswag": HellaswagMinimal,
    "story_cloze-2016": StoryclozeMinimal,
    "winogrande-winogrande_xl": WinograndeMinimal,
    "winogrande-winogrande_m": WinograndeMinimal,
    "xsum": XSumMinimal,
}


def get_prompt_template(dataset_name, subset_name) -> Tuple[Template, str]:
    """
    Choose a prompt template from bigscience-workshop/template source.
    You can do inference simply by typing a template index from terminal.
    """
    if dataset_name.startswith("anli"):
        dataset_name = dataset_name[:-1]
    prompts = DatasetTemplates(dataset_name, subset_name)
    prompts_type = prompts.name_to_id_mapping
    prompts_type_indices = {idx: key for idx, key in enumerate(prompts_type.keys())}
    for idx, key in prompts_type_indices.items():
        print(f"[{idx}] {key}")
    full_dataset_name = "-".join([dataset_name, subset_name]) if subset_name else dataset_name
    if full_dataset_name in MINIMAL_TEMPLATE_MAPPER.keys():
        minimal_template = MINIMAL_TEMPLATE_MAPPER[full_dataset_name]()
        print(f"[{idx + 1}] {minimal_template.get_name()}")
    chosen_idx = int(input("Choose Template Number: "))
    if chosen_idx == idx + 1:
        return minimal_template
    else:
        template = prompts.templates[prompts_type[prompts_type_indices[chosen_idx]]]
        return template


def get_template_mappings(dataset_name, subset_name) -> Tuple[Dict[int, str], DatasetTemplates]:
    if dataset_name.startswith("anli"):
        dataset_name = dataset_name[:-1]
    prompts = DatasetTemplates(dataset_name, subset_name)
    prompts_type = prompts.name_to_id_mapping
    return (prompts_type, prompts)


def get_minimal_template(dataset_name, subset_name) -> MinimalTemplate:
    template_key = dataset_name if subset_name is None else "-".join([dataset_name, subset_name])
    template = MINIMAL_TEMPLATE_MAPPER[template_key]()
    return template


def wrap_concat_ids(
    pre: Optional[List[int]],
    demon: Optional[List[int]],
    test: Optional[List[int]],
    post: Optional[List[int]],
    is_encoder: bool = True,
    test_data_to_decoder: bool = False,
):
    if test_data_to_decoder and is_encoder:
        return pre + demon + post
    elif not test_data_to_decoder and is_encoder:
        return pre + demon + test + post
    elif test_data_to_decoder and not is_encoder:
        return pre + test
    else:
        return pre


def k_shot_ds_for_classification(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:
    # TODO: Complete docstring with some of I/O format examples.
    """
    Create dataset in the format of k-shot classification.
    Get most probable answer from labels by comparing the perplexity (loss).
    """

    if train_dataset:
        demonstrations = []
        for idx in tqdm(range(len(train_dataset))):
            demon, label = template.apply(train_dataset[idx])
            if isinstance(label, list):
                label = label[0]
            demonstrations.append(" ".join([demon, label]))

        if "idx" in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(["idx"])
        valid_dataset = valid_dataset.add_column("idx", list(range(len(valid_dataset))))

    decoder_prefix_id = [decoder_start_id] if not use_sentinel else [decoder_start_id, first_sentinel_id]
    encoder_postfix_id = [] if not use_sentinel else [first_sentinel_id]
    encoder_prefix_id = (
        [] if denoiser_prefix is None else tokenizer(denoiser_prefix, add_special_tokens=False)["input_ids"]
    )

    # Map function for mapping k-shot format dataset
    def template_mapper_(sample):
        # Non-fixed sampels case
        if train_dataset and len(train_dataset) != num_k:
            start_idx = sample["idx"] * num_k % len(train_dataset)
            end_idx = (sample["idx"] + 1) * num_k % len(train_dataset)
            selected_demons: List[str] = (
                demonstrations[start_idx:end_idx]
                if start_idx < end_idx
                else demonstrations[start_idx:] + demonstrations[:end_idx]
            )
            joined_demons: str = tokenizer.eos_token.join(selected_demons) + tokenizer.eos_token
            tokenized_demons: List[int] = tokenizer(joined_demons, add_special_tokens=False)["input_ids"]
        # Fixed-samples case
        elif train_dataset and len(train_dataset) == num_k:
            joined_demons: str = tokenizer.eos_token.join(demonstrations) + tokenizer.eos_token
            tokenized_demons: List[int] = tokenizer(joined_demons, add_special_tokens=False)["input_ids"]
        # Zero-shot case
        else:
            tokenized_demons = []

        test_input, test_label = template.apply(sample)
        answer_choices: List[str] = template.get_answer_choices_list(sample)
        if isinstance(test_label, list):  # for multi-reference cases
            answer: List[int] = get_element_indices(answer_choices, test_label)
        else:
            answer: int = answer_choices.index(test_label)
        tokenized_ans: List[List[int]] = [
            tokenizer(ans, add_special_tokens=False)["input_ids"] for ans in answer_choices
        ]
        tokenized_input = tokenizer(test_input, add_special_tokens=False)["input_ids"]
        encoder_input_ids = (
            encoder_prefix_id + tokenized_demons + encoder_postfix_id
            if test_data_to_decoder
            else encoder_prefix_id + tokenized_demons + tokenized_input + encoder_postfix_id
        )
        decoder_prefix_id_ = decoder_prefix_id + tokenized_input if test_data_to_decoder else decoder_prefix_id

        input_ids = [[encoder_input_ids] for _ in range(len(answer_choices))]
        decoder_input_ids = [[decoder_prefix_id_ + ans] for ans in tokenized_ans]
        last_index_label = tokenizer.eos_token_id if add_eos_loss else -100
        labels = [
            [[-100 for _ in range(len(decoder_prefix_id_) - 1)] + ans + [last_index_label]] for ans in tokenized_ans
        ]  # Calculate perplexity (loss) only for the answer spans
        target = answer

        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "target": target,
        }

    return valid_dataset.map(
        template_mapper_, num_proc=num_proc, batch_size=1, remove_columns=valid_dataset.column_names
    )


def k_shot_ds_for_generation(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:

    if train_dataset:
        demonstrations = []
        for idx in tqdm(range(len(train_dataset))):
            example = template.apply(train_dataset[idx])
            demonstrations.append(" ".join(example))

        if "idx" in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(["idx"])
        valid_dataset = valid_dataset.add_column("idx", list(range(len(valid_dataset))))

    decoder_prefix_id = [decoder_start_id] if not use_sentinel else [decoder_start_id, first_sentinel_id]
    encoder_postfix_id = [] if not use_sentinel else [first_sentinel_id]
    encoder_prefix_id = (
        [] if denoiser_prefix is None else tokenizer(denoiser_prefix, add_special_tokens=False)["input_ids"]
    )

    # Map function for mapping k-shot format dataset
    def template_mapper_(sample):
        # Non-fixed samples case
        if train_dataset and len(train_dataset) != num_k:
            start_idx = sample["idx"] * num_k % len(train_dataset)
            end_idx = (sample["idx"] + 1) * num_k % len(train_dataset)
            selected_demons: List[str] = (
                demonstrations[start_idx:end_idx]
                if start_idx < end_idx
                else demonstrations[start_idx:] + demonstrations[:end_idx]
            )
            joined_demons: str = tokenizer.eos_token.join(selected_demons) + tokenizer.eos_token
            tokenized_demons: List[int] = tokenizer(joined_demons, add_special_tokens=False)["input_ids"]
        # Fixed-samples case
        elif train_dataset and len(train_dataset) == num_k:
            joined_demons: str = tokenizer.eos_token.join(demonstrations) + tokenizer.eos_token
            tokenized_demons: List[int] = tokenizer(joined_demons, add_special_tokens=False)["input_ids"]
        # Zero-shot case
        else:
            tokenized_demons = []

        test_input, test_label = template.apply(sample)
        tokenized_input = tokenizer(test_input, add_special_tokens=False)["input_ids"]
        encoder_input_ids = (
            encoder_prefix_id + tokenized_demons + encoder_postfix_id
            if test_data_to_decoder
            else encoder_prefix_id + tokenized_demons + tokenized_input + encoder_postfix_id
        )
        decoder_input_ids = decoder_prefix_id + tokenized_input if test_data_to_decoder else decoder_prefix_id

        return {
            "input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "answer_start_idx": len(decoder_input_ids) - 1,
            "target": test_label,
        }

    return valid_dataset.map(
        template_mapper_, num_proc=num_proc, batch_size=1, remove_columns=valid_dataset.column_names
    )


def fid_k_shot_ds_for_classification(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:
    """Same function as 'k_shot_ds_for_classification()' in Fusion-in-Decoder fomat."""

    if not train_dataset:
        raise ValueError("Fusion-in-Decoder style few-shot does not support zero-shot learning")
    else:
        demonstrations = []
        for idx in tqdm(range(len(train_dataset))):
            demon, label = template.apply(train_dataset[idx])
            if isinstance(label, list):
                label = label[0]
            demonstrations.append(" ".join([demon, label]) + tokenizer.eos_token)

        if "idx" in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(["idx"])
        valid_dataset = valid_dataset.add_column("idx", list(range(len(valid_dataset))))

    decoder_prefix_id = [decoder_start_id] if not use_sentinel else [decoder_start_id, first_sentinel_id]
    encoder_postfix_id = [] if not use_sentinel else [first_sentinel_id]
    encoder_prefix_id = (
        [] if denoiser_prefix is None else tokenizer(denoiser_prefix, add_special_tokens=False)["input_ids"]
    )

    # Map function for mapping k-shot format dataset
    def template_mapper_(sample):
        # Non-fixed samples case
        if train_dataset and len(train_dataset) != num_k:
            start_idx = sample["idx"] * num_k % len(train_dataset)
            end_idx = (sample["idx"] + 1) * num_k % len(train_dataset)
            selected_demons: List[str] = (
                demonstrations[start_idx:end_idx]
                if start_idx < end_idx
                else demonstrations[start_idx:] + demonstrations[:end_idx]
            )
            tokenized_demons: List[List[int]] = tokenizer(selected_demons, add_special_tokens=False)["input_ids"]
        # Fixed-samples case
        else:
            tokenized_demons: List[List[int]] = tokenizer(demonstrations, add_special_tokens=False)["input_ids"]

        test_input, test_label = template.apply(sample)
        answer_choices: List[str] = template.get_answer_choices_list(sample)
        if isinstance(test_label, list):  # for multi-reference cases
            answer: List[int] = get_element_indices(answer_choices, test_label)
        else:
            answer: int = answer_choices.index(test_label)
        tokenized_ans: List[List[int]] = [
            tokenizer(ans, add_special_tokens=False)["input_ids"] for ans in answer_choices
        ]
        tokenized_input = tokenizer(test_input, add_special_tokens=False)["input_ids"]

        encoder_input_ids = [
            encoder_prefix_id + demon + encoder_postfix_id
            if test_data_to_decoder
            else encoder_prefix_id + demon + tokenized_input + encoder_postfix_id
            for demon in tokenized_demons
        ]

        decoder_prefix_id_ = decoder_prefix_id + tokenized_input if test_data_to_decoder else decoder_prefix_id

        input_ids = [encoder_input_ids for _ in range(len(answer_choices))]
        decoder_input_ids = [[decoder_prefix_id_ + ans] for ans in tokenized_ans]
        last_index_label = tokenizer.eos_token_id if add_eos_loss else -100
        labels = [
            [[-100 for _ in range(len(decoder_prefix_id_) - 1)] + ans + [last_index_label]] for ans in tokenized_ans
        ]  # Calculate perplexity (loss) only for the answer spans
        target = answer

        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "target": target,
        }

    return valid_dataset.map(
        template_mapper_, num_proc=num_proc, batch_size=1, remove_columns=valid_dataset.column_names
    )


def fid_k_shot_ds_for_generation(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:

    if not train_dataset:
        raise ValueError("Fusion-in-Decoder style few-shot does not support zero-shot learning")
    else:
        demonstrations = []
        for idx in tqdm(range(len(train_dataset))):
            example = template.apply(train_dataset[idx])
            demonstrations.append(" ".join(example) + tokenizer.eos_token)

        if "idx" in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(["idx"])
        valid_dataset = valid_dataset.add_column("idx", list(range(len(valid_dataset))))

    decoder_prefix_id = [decoder_start_id] if not use_sentinel else [decoder_start_id, first_sentinel_id]
    encoder_postfix_id = [] if not use_sentinel else [first_sentinel_id]
    encoder_prefix_id = (
        [] if denoiser_prefix is None else tokenizer(denoiser_prefix, add_special_tokens=False)["input_ids"]
    )

    # Map function for mapping k-shot format dataset
    def template_mapper_(sample):
        # Non-fixed samples case
        if train_dataset and len(train_dataset) != num_k:
            start_idx = sample["idx"] * num_k % len(train_dataset)
            end_idx = (sample["idx"] + 1) * num_k % len(train_dataset)
            selected_demons: List[str] = (
                demonstrations[start_idx:end_idx]
                if start_idx < end_idx
                else demonstrations[start_idx:] + demonstrations[:end_idx]
            )
            tokenized_demons: List[List[int]] = tokenizer(selected_demons, add_special_tokens=False)["input_ids"]
        # Fixed-samples case
        else:
            tokenized_demons: List[List[int]] = tokenizer(demonstrations, add_special_tokens=False)["input_ids"]

        test_input, test_label = template.apply(sample)
        tokenized_input = tokenizer(test_input, add_special_tokens=False)["input_ids"]
        encoder_input_ids = [
            encoder_prefix_id + demon + encoder_postfix_id
            if test_data_to_decoder
            else encoder_prefix_id + demon + tokenized_input + encoder_postfix_id
            for demon in tokenized_demons
        ]

        decoder_input_ids = decoder_prefix_id + tokenized_input if test_data_to_decoder else decoder_prefix_id

        return {
            "input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "answer_start_idx": len(decoder_input_ids) - 1,
            "target": test_label,
        }

    return valid_dataset.map(
        template_mapper_, num_proc=num_proc, batch_size=1, remove_columns=valid_dataset.column_names
    )


def rag_k_shot_ds_for_classification(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:
    """Same function as 'k_shot_ds_for_classification()' in RAG fomat."""

    if not train_dataset:
        raise ValueError("RAG style few-shot does not support zero-shot learning")
    else:
        demonstrations = []
        for idx in tqdm(range(len(train_dataset))):
            demon, label = template.apply(train_dataset[idx])
            if isinstance(label, list):
                label = label[0]
            demonstrations.append(" ".join([demon, label]) + tokenizer.eos_token)

        if "idx" in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(["idx"])
        valid_dataset = valid_dataset.add_column("idx", list(range(len(valid_dataset))))

    decoder_prefix_id = [decoder_start_id] if not use_sentinel else [decoder_start_id, first_sentinel_id]
    encoder_postfix_id = [] if not use_sentinel else [first_sentinel_id]
    encoder_prefix_id = (
        [] if denoiser_prefix is None else tokenizer(denoiser_prefix, add_special_tokens=False)["input_ids"]
    )

    # Map function for mapping k-shot format dataset
    def template_mapper_(sample):
        # Non-fixed samples case
        if train_dataset and len(train_dataset) != num_k:
            start_idx = sample["idx"] * num_k % len(train_dataset)
            end_idx = (sample["idx"] + 1) * num_k % len(train_dataset)
            selected_demons: List[str] = (
                demonstrations[start_idx:end_idx]
                if start_idx < end_idx
                else demonstrations[start_idx:] + demonstrations[:end_idx]
            )
            tokenized_demons: List[List[int]] = tokenizer(selected_demons, add_special_tokens=False)["input_ids"]
        # Fixed-samples case
        else:
            tokenized_demons: List[List[int]] = tokenizer(demonstrations, add_special_tokens=False)["input_ids"]

        test_input, test_label = template.apply(sample)
        answer_choices: List[str] = template.get_answer_choices_list(sample)
        if isinstance(test_label, list):  # for multi-reference cases
            answer: List[int] = get_element_indices(answer_choices, test_label)
        else:
            answer: int = answer_choices.index(test_label)
        tokenized_ans: List[List[int]] = [
            tokenizer(ans, add_special_tokens=False)["input_ids"] for ans in answer_choices
        ]
        tokenized_input = tokenizer(test_input, add_special_tokens=False)["input_ids"]

        encoder_input_ids = [
            encoder_prefix_id + demon + encoder_postfix_id
            if test_data_to_decoder
            else encoder_prefix_id + demon + tokenized_input + encoder_postfix_id
            for demon in tokenized_demons
        ]

        decoder_prefix_id_ = decoder_prefix_id + tokenized_input if test_data_to_decoder else decoder_prefix_id

        input_ids = [encoder_input_ids for _ in range(len(answer_choices))]
        decoder_input_ids = [[decoder_prefix_id_ + ans] for ans in tokenized_ans]
        last_index_label = tokenizer.eos_token_id if add_eos_loss else -100
        labels = [
            [[-100 for _ in range(len(decoder_prefix_id_) - 1)] + ans + [last_index_label]] for ans in tokenized_ans
        ]  # Calculate perplexity (loss) only for the answer spans
        target = answer

        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "target": target,
        }

    return valid_dataset.map(
        template_mapper_, num_proc=num_proc, batch_size=1, remove_columns=valid_dataset.column_names
    )


def rag_k_shot_ds_for_generation(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:

    if not train_dataset:
        raise ValueError("RAG style few-shot does not support zero-shot learning")
    else:
        demonstrations = []
        for idx in tqdm(range(len(train_dataset))):
            example = template.apply(train_dataset[idx])
            demonstrations.append(" ".join(example) + tokenizer.eos_token)

        if "idx" in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(["idx"])
        valid_dataset = valid_dataset.add_column("idx", list(range(len(valid_dataset))))

    decoder_prefix_id = [decoder_start_id] if not use_sentinel else [decoder_start_id, first_sentinel_id]
    encoder_postfix_id = [] if not use_sentinel else [first_sentinel_id]
    encoder_prefix_id = (
        [] if denoiser_prefix is None else tokenizer(denoiser_prefix, add_special_tokens=False)["input_ids"]
    )

    # Map function for mapping k-shot format dataset
    def template_mapper_(sample):
        # Non-fixed samples case
        if train_dataset and len(train_dataset) != num_k:
            start_idx = sample["idx"] * num_k % len(train_dataset)
            end_idx = (sample["idx"] + 1) * num_k % len(train_dataset)
            selected_demons: List[str] = (
                demonstrations[start_idx:end_idx]
                if start_idx < end_idx
                else demonstrations[start_idx:] + demonstrations[:end_idx]
            )
            tokenized_demons: List[List[int]] = tokenizer(selected_demons, add_special_tokens=False)["input_ids"]
        # Fixed-samples case
        else:
            tokenized_demons: List[List[int]] = tokenizer(demonstrations, add_special_tokens=False)["input_ids"]

        test_input, test_label = template.apply(sample)
        tokenized_input = tokenizer(test_input, add_special_tokens=False)["input_ids"]
        encoder_input_ids = [
            encoder_prefix_id + demon + encoder_postfix_id
            if test_data_to_decoder
            else encoder_prefix_id + demon + tokenized_input + encoder_postfix_id
            for demon in tokenized_demons
        ]

        decoder_input_ids = decoder_prefix_id + tokenized_input if test_data_to_decoder else decoder_prefix_id

        return {
            "input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "answer_start_idx": len(decoder_input_ids) - 1,
            "target": test_label,
        }

    return valid_dataset.map(
        template_mapper_, num_proc=num_proc, batch_size=1, remove_columns=valid_dataset.column_names
    )
