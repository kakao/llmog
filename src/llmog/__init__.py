import torch

from llmog.prompting import (
    fid_k_shot_ds_for_classification,
    fid_k_shot_ds_for_generation,
    k_shot_ds_for_classification,
    k_shot_ds_for_generation,
    rag_k_shot_ds_for_classification,
    rag_k_shot_ds_for_generation,
)
from llmog.run import (
    run_fid_k_shot_classification,
    run_fid_k_shot_generation,
    run_k_shot_classification,
    run_k_shot_generation,
    run_rag_k_shot_classification,
    run_rag_k_shot_generation,
)

METHOD_MAPPING = {
    "accuracy": {
        "k-shot": k_shot_ds_for_classification,
        "fid-k-shot": fid_k_shot_ds_for_classification,
        "rag-sequence-k-shot": rag_k_shot_ds_for_classification,
        "rag-token-k-shot": rag_k_shot_ds_for_classification,
    },
    "f1": {
        "k-shot": k_shot_ds_for_classification,
        "fid-k-shot": fid_k_shot_ds_for_classification,
        "rag-sequence-k-shot": rag_k_shot_ds_for_classification,
        "rag-token-k-shot": rag_k_shot_ds_for_classification,
    },
    "boolq": {
        "k-shot": k_shot_ds_for_classification,
        "fid-k-shot": fid_k_shot_ds_for_classification,
        "rag-sequence-k-shot": rag_k_shot_ds_for_classification,
        "rag-token-k-shot": rag_k_shot_ds_for_classification,
    },
    "copa": {
        "k-shot": k_shot_ds_for_classification,
        "fid-k-shot": fid_k_shot_ds_for_classification,
        "rag-sequence-k-shot": rag_k_shot_ds_for_classification,
        "rag-token-k-shot": rag_k_shot_ds_for_classification,
    },
    "cb": {
        "k-shot": k_shot_ds_for_classification,
        "fid-k-shot": fid_k_shot_ds_for_classification,
        "rag-sequence-k-shot": rag_k_shot_ds_for_classification,
        "rag-token-k-shot": rag_k_shot_ds_for_classification,
    },
    "multirc": {
        "k-shot": k_shot_ds_for_classification,
        "fid-k-shot": fid_k_shot_ds_for_classification,
        "rag-sequence-k-shot": rag_k_shot_ds_for_classification,
        "rag-token-k-shot": rag_k_shot_ds_for_classification,
    },
    "rte": {
        "k-shot": k_shot_ds_for_classification,
        "fid-k-shot": fid_k_shot_ds_for_classification,
        "rag-sequence-k-shot": rag_k_shot_ds_for_classification,
        "rag-token-k-shot": rag_k_shot_ds_for_classification,
    },
    "axg": {
        "k-shot": k_shot_ds_for_classification,
        "fid-k-shot": fid_k_shot_ds_for_classification,
        "rag-sequence-k-shot": rag_k_shot_ds_for_classification,
        "rag-token-k-shot": rag_k_shot_ds_for_classification,
    },
    "wsc.fixed": {
        "k-shot": k_shot_ds_for_classification,
        "fid-k-shot": fid_k_shot_ds_for_classification,
        "rag-sequence-k-shot": rag_k_shot_ds_for_classification,
        "rag-token-k-shot": rag_k_shot_ds_for_classification,
    },
    "record": {
        "k-shot": k_shot_ds_for_classification,
        "fid-k-shot": fid_k_shot_ds_for_classification,
        "rag-sequence-k-shot": rag_k_shot_ds_for_classification,
        "rag-token-k-shot": rag_k_shot_ds_for_classification,
    },
    "wic": {
        "k-shot": k_shot_ds_for_classification,
        "fid-k-shot": fid_k_shot_ds_for_classification,
        "rag-sequence-k-shot": rag_k_shot_ds_for_classification,
        "rag-token-k-shot": rag_k_shot_ds_for_classification,
    },
    "rouge": {
        "k-shot": k_shot_ds_for_generation,
        "fid-k-shot": fid_k_shot_ds_for_generation,
        "rag-token-k-shot": rag_k_shot_ds_for_generation,
    },
    "bleu": {
        "k-shot": k_shot_ds_for_generation,
        "fid-k-shot": fid_k_shot_ds_for_generation,
        "rag-token-k-shot": rag_k_shot_ds_for_generation,
    },
    "squad": {
        "k-shot": k_shot_ds_for_generation,
        "fid-k-shot": fid_k_shot_ds_for_generation,
        "rag-token-k-shot": rag_k_shot_ds_for_generation,
    },
}

RUN_MAPPER = {
    k_shot_ds_for_classification: run_k_shot_classification,
    k_shot_ds_for_generation: run_k_shot_generation,
    fid_k_shot_ds_for_classification: run_fid_k_shot_classification,
    fid_k_shot_ds_for_generation: run_fid_k_shot_generation,
    rag_k_shot_ds_for_classification: run_rag_k_shot_classification,
    rag_k_shot_ds_for_generation: run_rag_k_shot_generation,
}

METRIC_MAPPER = {
    "anli1": ["accuracy"],
    "anli2": ["accuracy"],
    "anli3": ["accuracy"],
    "hellaswag": ["accuracy"],
    "story_cloze-2016": ["accuracy"],
    "story_cloze-2018": ["accuracy"],
    "super_glue-boolq": ["super_glue", "boolq"],
    "super_glue-copa": ["super_glue", "copa"],
    "super_glue-cb": ["super_glue", "cb"],
    "super_glue-multirc": ["accuracy"],
    "super_glue-rte": ["super_glue", "rte"],
    "super_glue-axg": ["super_glue", "axg"],
    "super_glue-wsc.fixed": ["super_glue", "wsc.fixed"],
    "super_glue-record": ["super_glue", "record"],
    "super_glue-wic": ["super_glue", "wic"],
    "winogrande-winogrande_xs": ["accuracy"],
    "winogrande-winogrande_s": ["accuracy"],
    "winogrande-winogrande_m": ["accuracy"],
    "winogrande-winogrande_l": ["accuracy"],
    "winogrande-winogrande_xl": ["accuracy"],
    "winogrande-winogrande_debiased": ["accuracy"],
    "xsum": ["rouge"],
    "enriched_web_nlg-en": ["rouge"],
    "e2e_nlg_cleaned": ["rouge"],
}


TORCH_DTYPE_MAPPING = {
    "torch.float32": torch.float32,
    "float32": torch.float32,
    "torch.float16": torch.float16,
    "float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
