import argparse
import inspect
import os
import random
import re
from copy import deepcopy
from typing import Any, Dict, List

from omegaconf import OmegaConf
from promptsource.templates import Template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llmog import METHOD_MAPPING, METRIC_MAPPER, RUN_MAPPER, TORCH_DTYPE_MAPPING
from llmog.models.fid_model import FiDModel
from llmog.models.rag_model import RagSequenceModel, RagTokenModel
from llmog.prompting import get_prompt_template
from llmog.utils import fix_seed, get_train_valid_dataset, write_results


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument_group(title="plm")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_revision", type=str, default=None, help="The revision for model checkpoint")
    parser.add_argument("--torch_dtype", type=str, default="torch.float32", help="The dtype for the model")
    parser.add_argument("--first_sentinel_token", type=str, default="<extra_id_0>")
    parser.add_argument("--denoiser_prefix", type=str, default=None, choices=["[NLU]", "[NLG]", "[S2S]"])
    parser.add_argument("--model_cache_dir", type=str, default=None)

    parser.add_argument_group(title="env")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_gpus", type=int, default=1, help="Num gpus in your node for model parallel")
    parser.add_argument("--logging_samples", action="store_true", help="Print I/O text or not")
    parser.add_argument("--output_file", type=str, help="Path to save output results")

    parser.add_argument_group(title="task")
    parser.add_argument(
        "--type", type=str, choices=["k-shot", "fid-k-shot", "rag-sequence-k-shot", "rag-token-k-shot"]
    )
    parser.add_argument("--num_k", type=int, default=10)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument(
        "--subtask_name", type=str, default=None, help="For the bundle-type benchmark like superGLUE or KLUE"
    )
    parser.add_argument(
        "--train_path", type=str, default=None, help="Train data path for tasks not uploaded to huggingface datasets"
    )
    parser.add_argument(
        "--valid_path", type=str, default=None, help="Valid data path for tasks not uploaded to huggingface datasets"
    )
    parser.add_argument(
        "--fix_demon_samples", action="store_true", help="Fix few-shot demonstrations for all test iterations."
    )
    parser.add_argument("--num_valid_samples", type=int, help="Num test samples for evaluation")
    parser.add_argument("--num_valid_ratio", type=int, help="Test samples ratio for evaluation")
    parser.add_argument("--use_sentinel", action="store_true", help="Use sentinel token just as in pretrain.")
    parser.add_argument("--test_data_to_decoder", action="store_true", help="Put test input to decoder or not")
    parser.add_argument(
        "--add_eos_loss", action="store_true", help="If true, loss for eos token be included to total loss comparison."
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="sum",
        choices=["sum", "mean"],
        help="Loss reduction method for classification tasks",
    )
    parser.add_argument("--generation_hp_path", type=str, help="Yaml path for generation eval hyperparameters")
    args = parser.parse_args()
    return args


def get_generation_kwargs(generation_config_path: str) -> Dict[str, Any]:
    nested_args = OmegaConf.load(generation_config_path)
    generation_kwargs = nested_args.generation
    return generation_kwargs


def main():
    args = get_args()

    fix_seed(args.seed)
    model_kwargs = {
        "torch_dtype": TORCH_DTYPE_MAPPING[args.torch_dtype],
        "revision": args.model_revision,
        "low_cpu_mem_usage": True,
        "cache_dir": args.model_cache_dir,
    }

    template: Template = get_prompt_template(args.dataset_name, args.subtask_name)
    print(template.get_name())
    train_dataset, valid_dataset = get_train_valid_dataset(
        args.dataset_name, args.subtask_name, args.train_path, args.valid_path
    )

    # Sample valid(test) dataset if given sampling args
    if args.num_valid_ratio and not args.num_valid_samples:
        args.num_valid_samples = int(len(valid_dataset) * args.num_valid_ratio)
    if args.num_valid_ratio or args.num_valid_samples:
        if args.num_valid_samples >= len(valid_dataset):
            args.num_valid_samples = len(valid_dataset)
            print("Number of valid set entered is greater than total set, so all the valid set will be used")
        else:
            valid_random_indices = random.sample(range(len(valid_dataset)), args.num_valid_samples)
            valid_dataset = valid_dataset.select(valid_random_indices)

    if args.fix_demon_samples:
        train_random_indices = random.sample(range(len(train_dataset)), args.num_k)
    else:
        num_demon_examples = min(args.num_k * len(valid_dataset), len(train_dataset))
        train_random_indices = random.sample(range(len(train_dataset)), num_demon_examples)
    train_dataset = train_dataset.select(train_random_indices)

    print(f"Num Train Samples for k-shot: {args.num_k}")
    print(f"Num Test Samples: {len(valid_dataset)}")

    # check tokenizer include sentinel token
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.use_sentinel:
        assert args.first_sentinel_token in tokenizer.get_vocab(), "Put proper extra token"

    # load model for proper k-shot type
    if args.type == "fid-k-shot":
        # We recommend saving model after 'load_t5' using 'model.save_pretrained(save_path)'
        # and then loading the model using 'FiDT5.from_pretrained(save_path, **model_kwargs)'
        # from next time because 'load_t5' function is too slow
        model_base = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, **model_kwargs)
        model = FiDModel(model_base.config)
        model.load_t5(model_base.state_dict())
    elif args.type.startswith("rag"):
        rag_mapper = {"rag-token-k-shot": RagTokenModel, "rag-sequence-k-shot": RagSequenceModel}

        model = rag_mapper[args.type].from_pretrained(args.model_path, **model_kwargs)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, **model_kwargs)
    if args.num_gpus > 1:
        from parallelformers import parallelize

        parallelize(model, num_gpus=args.num_gpus, fp16=False, verbose="simple")
    else:
        model.to("cuda")

    # set decoder start token id
    decoder_start_id = (
        model.config.decoder_start_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    first_sentinel_id = tokenizer.get_vocab()[args.first_sentinel_token]

    # create proper dataset for k-shot method and task type
    task_key = "-".join([args.dataset_name, args.subtask_name]) if args.subtask_name else args.dataset_name
    metric_name: List[str] = METRIC_MAPPER[task_key]
    mapped_ds = METHOD_MAPPING[metric_name[-1]][args.type](
        tokenizer=tokenizer,
        decoder_start_id=decoder_start_id,
        first_sentinel_id=first_sentinel_id,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        template=template,
        num_k=args.num_k,
        denoiser_prefix=args.denoiser_prefix,
        use_sentinel=args.use_sentinel,
        test_data_to_decoder=args.test_data_to_decoder,
        add_eos_loss=args.add_eos_loss,
        num_proc=os.cpu_count() // 2,
    )

    logging_message = deepcopy(args.type).upper()
    logging_message = re.sub("K-SHOT", f"{args.num_k}-SHOT", logging_message)
    run_fn_ = RUN_MAPPER[METHOD_MAPPING[metric_name[-1]][args.type]]
    if "generation_kwargs" not in inspect.signature(run_fn_).parameters:
        generation_kwargs = None
        print(f"{logging_message} classification")
        result = run_fn_(
            model,
            tokenizer,
            mapped_ds,
            metric_name,
            args.reduction,
            args.num_gpus != 1,
            args.logging_samples,
        )
    else:
        print(f"{logging_message} generation")
        generation_kwargs = get_generation_kwargs(args.generation_hp_path)
        print(generation_kwargs)
        result = run_fn_(
            model,
            tokenizer,
            mapped_ds,
            metric_name,
            generation_kwargs,
            args.num_gpus != 1,
            args.logging_samples,
            args.use_sentinel,
            args.first_sentinel_token,
        )

    print(f"Results with [{template.get_name()}]")
    print(args)
    print(result)
    if args.output_file:
        write_results(
            results=result,
            template_name=template.get_name(),
            args=args,
            generation_kwargs=generation_kwargs,
        )


if __name__ == "__main__":
    main()
