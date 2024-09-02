from typing import Dict, List

import evaluate
import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.data.data_collator import _torch_collate_batch

from llmog.function import cross_entropy_loss, multi_references_acc
from llmog.utils import logging_decoded_samples


def run_k_shot_classification(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metric_name: List[str],
    reduction: str = "sum",
    is_parallel: bool = False,
    logging_samples: bool = False,
) -> Dict[str, float]:
    metric: evaluate.EvaluationModule = evaluate.load(*metric_name)
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for ds in tqdm(dataset):
            all_losses = []
            targets.append(ds.pop("target"))
            for ei, di, la in zip(ds["input_ids"], ds["decoder_input_ids"], ds["labels"]):
                ei, di, la = torch.tensor(ei), torch.tensor(di), torch.tensor(la)
                if not is_parallel:
                    ei, di, la = ei.to("cuda"), di.to("cuda"), la.to("cuda")
                if logging_samples:
                    logging_decoded_samples(tokenizer, ei, di, la)
                output = model(
                    input_ids=ei,
                    decoder_input_ids=di,
                    labels=la,
                    return_dict=True,
                )
                if is_parallel:
                    logits = output["logits"].cpu()
                    loss = cross_entropy_loss(logits, la, reduction)
                else:
                    loss = cross_entropy_loss(output["logits"], la, reduction)
                all_losses.append(loss.item())
            predictions.append(np.argmin(all_losses))
            torch.cuda.empty_cache()
        if isinstance(targets[0], list):  # multi-reference case
            result = multi_references_acc(predictions=predictions, references=targets)
        else:
            result = metric.compute(predictions=predictions, references=targets)
    return result


def run_k_shot_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metric_name: List[str],
    generation_kwargs: Dict,
    is_parallel: bool = False,
    logging_samples: bool = False,
    use_sentinel: bool = True,
    first_sentinel_token: str = "<extra_id_0>",
) -> Dict[str, float]:
    metric: evaluate.EvaluationModule = evaluate.load(*metric_name)
    predictions, targets = [], []
    stopping_sequences = generation_kwargs.pop("stopping_sequences")

    model.eval()
    with torch.no_grad():
        for ds in tqdm(dataset):
            target: str = ds["target"]
            targets.append(target)
            input_ids, decoder_input_ids = torch.tensor([ds["input_ids"]]), torch.tensor([ds["decoder_input_ids"]])
            if not is_parallel:
                input_ids, decoder_input_ids = input_ids.to("cuda"), decoder_input_ids.to("cuda")
            if logging_samples:
                logging_decoded_samples(tokenizer, input_ids, decoder_input_ids)
            gen: torch.Tensor = model.generate(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                eos_token_id=tokenizer.eos_token_id,
                **generation_kwargs,
            )
            gen_text: str = tokenizer.decode(gen.tolist()[0][ds["answer_start_idx"] :], skip_special_tokens=False)
            if use_sentinel:
                gen_text = gen_text.replace(first_sentinel_token, "").strip()
            if stopping_sequences:
                for seq in stopping_sequences:
                    gen_text = gen_text.split(seq)[0]
            if logging_samples:
                print(f"TARGET TEXT : {target}")
                print(f"GENERATED TEXT : {gen_text}")
            predictions.append(gen_text)
            torch.cuda.empty_cache()
    result = metric.compute(predictions=predictions, references=targets)
    return result


def run_fid_k_shot_classification(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metric_name: List[str],
    reduction: str = "sum",
    is_parallel: bool = False,
    logging_samples: bool = False,
) -> Dict[str, float]:
    metric: evaluate.EvaluationModule = evaluate.load(*metric_name)
    predictions, targets = [], []
    pad_token_id = tokenizer.pad_token_id
    model.eval()
    with torch.no_grad():
        for ds in tqdm(dataset):
            all_losses = []
            targets.append(ds.pop("target"))
            for ei, di, la in zip(ds["input_ids"], ds["decoder_input_ids"], ds["labels"]):
                ei = _torch_collate_batch(ei, tokenizer=tokenizer).unsqueeze(0)
                ea = torch.where(ei == pad_token_id, 0, torch.ones_like(ei))
                di, la = torch.tensor(di), torch.tensor(la)
                if not is_parallel:
                    ei, ea, di, la = ei.to("cuda"), ea.to("cuda"), di.to("cuda"), la.to("cuda")
                if logging_samples:
                    logging_decoded_samples(tokenizer, ei, di, la)
                output = model(
                    input_ids=ei,
                    attention_mask=ea,
                    decoder_input_ids=di,
                    labels=la,
                    return_dict=True,
                )
                if is_parallel:
                    logits = output["logits"].cpu()
                    loss = cross_entropy_loss(logits, la, reduction)
                else:
                    loss = cross_entropy_loss(output["logits"], la, reduction)
                all_losses.append(loss.item())
            predictions.append(np.argmin(all_losses))
            torch.cuda.empty_cache()
        if isinstance(targets[0], list):  # multi-reference case
            result = multi_references_acc(predictions=predictions, references=targets)
        else:
            result = metric.compute(predictions=predictions, references=targets)
    return result


def run_fid_k_shot_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metric_name: List[str],
    generation_kwargs: Dict,
    is_parallel: bool = False,
    logging_samples: bool = False,
    use_sentinel: bool = True,
    first_sentinel_token: str = "<extra_id_0>",
) -> Dict[str, float]:
    metric: evaluate.EvaluationModule = evaluate.load(*metric_name)
    predictions, targets = [], []
    pad_token_id = tokenizer.pad_token_id
    stopping_sequences = generation_kwargs.pop("stopping_sequences")

    model.eval()
    with torch.no_grad():
        for ds in tqdm(dataset):
            target: str = ds["target"]
            targets.append(target)
            input_ids = _torch_collate_batch(ds["input_ids"], tokenizer=tokenizer).unsqueeze(0)
            attention_mask = torch.where(input_ids == pad_token_id, 0, torch.ones_like(input_ids))
            decoder_input_ids = torch.tensor([ds["decoder_input_ids"]])
            if not is_parallel:
                input_ids, attention_mask, decoder_input_ids = (
                    input_ids.to("cuda"),
                    attention_mask.to("cuda"),
                    decoder_input_ids.to("cuda"),
                )
            if logging_samples:
                logging_decoded_samples(tokenizer, input_ids, decoder_input_ids)
            gen: torch.Tensor = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                eos_token_id=tokenizer.eos_token_id,
                **generation_kwargs,
            )
            gen_text: str = tokenizer.decode(gen.tolist()[0][ds["answer_start_idx"] :], skip_special_tokens=False)
            if use_sentinel:
                gen_text = gen_text.replace(first_sentinel_token, "").strip()
            if stopping_sequences:
                for seq in stopping_sequences:
                    gen_text = gen_text.split(seq)[0]
            if logging_samples:
                print(f"TARGET TEXT : {target}")
                print(f"GENERATED TEXT : {gen_text}")
            predictions.append(gen_text)
    result = metric.compute(predictions=predictions, references=targets)
    return result


def run_rag_k_shot_classification(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metric_name: List[str],
    reduction: str = "sum",
    is_parallel: bool = False,
    logging_samples: bool = False,
) -> Dict[str, float]:
    metric: evaluate.EvaluationModule = evaluate.load(*metric_name)
    predictions, targets = [], []
    pad_token_id = tokenizer.pad_token_id
    model.eval()
    with torch.no_grad():
        for ds in tqdm(dataset):
            all_losses = []
            targets.append(ds.pop("target"))
            for ei, di, la in zip(ds["input_ids"], ds["decoder_input_ids"], ds["labels"]):
                add_eos_loss = tokenizer.eos_token_id in la[0]
                ei = _torch_collate_batch(ei, tokenizer=tokenizer)
                ea = torch.where(ei == pad_token_id, 0, torch.ones_like(ei))
                di, la = torch.tensor(di), torch.tensor(la)
                ds = torch.zeros(1, ei.size(0)).float()
                if not is_parallel:
                    ei, ea, di, la, ds = ei.to("cuda"), ea.to("cuda"), di.to("cuda"), la.to("cuda"), ds.to("cuda")
                if logging_samples:
                    logging_decoded_samples(tokenizer, ei, di, la)
                loss = model(
                    input_ids=ei,
                    attention_mask=ea,
                    decoder_input_ids=di,
                    labels=la,
                    doc_scores=ds,
                    n_docs=ei.size(0),
                    return_dict=True,
                    add_eos_loss=add_eos_loss,
                    reduction=reduction,
                )["loss"]
                all_losses.append(loss.item())
            predictions.append(np.argmin(all_losses))
            torch.cuda.empty_cache()
        if isinstance(targets[0], list):  # multi-reference case
            result = multi_references_acc(predictions=predictions, references=targets)
        else:
            result = metric.compute(predictions=predictions, references=targets)
    return result


def run_rag_k_shot_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metric_name: List[str],
    generation_kwargs: Dict,
    is_parallel: bool = False,
    logging_samples: bool = False,
    use_sentinel: bool = True,
    first_sentinel_token: str = "<extra_id_0>",
) -> Dict[str, float]:
    metric: evaluate.EvaluationModule = evaluate.load(*metric_name)
    predictions, targets = [], []
    pad_token_id = tokenizer.pad_token_id
    stopping_sequences = generation_kwargs.pop("stopping_sequences")

    model.eval()
    with torch.no_grad():
        for ds in tqdm(dataset):
            target: str = ds["target"]
            targets.append(target)
            input_ids = _torch_collate_batch(ds["input_ids"], tokenizer=tokenizer)
            attention_mask = torch.where(input_ids == pad_token_id, 0, torch.ones_like(input_ids))
            decoder_input_ids = torch.tensor([ds["decoder_input_ids"]])
            if not is_parallel:
                input_ids, attention_mask, decoder_input_ids = (
                    input_ids.to("cuda"),
                    attention_mask.to("cuda"),
                    decoder_input_ids.to("cuda"),
                )
            if logging_samples:
                logging_decoded_samples(tokenizer, input_ids, decoder_input_ids)
            gen: torch.Tensor = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                eos_token_id=tokenizer.eos_token_id,
                **generation_kwargs,
            )
            gen_text: str = tokenizer.decode(gen.tolist()[0][ds["answer_start_idx"] :], skip_special_tokens=False)
            if use_sentinel:
                gen_text = gen_text.replace(first_sentinel_token, "").strip()
            if stopping_sequences:
                for seq in stopping_sequences:
                    gen_text = gen_text.split(seq)[0]
            if logging_samples:
                print(f"TARGET TEXT : {target}")
                print(f"GENERATED TEXT : {gen_text}")
            predictions.append(gen_text)
    result = metric.compute(predictions=predictions, references=targets)
    return result
