import argparse

import torch
from accelerate.hooks import remove_hook_from_submodules

from main import perplexity_eval
from src.datautils import get_loaders
from src.modelutils import get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    # Model params
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the teacher model",
    )
    parser.add_argument(
        "--quant_model",
        type=str,
        required=True,
        help="path to quantized model",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--eval_model_seqlen",
        type=int,
        default=None,
        help="Model seqlen on validation. By default is equal to model_seqlen.",
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["wikitext2", "c4"],
        help="Datasets to run evaluation on",
    )
    # Misc params
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        choices=[None, "auto"],
        help="accelerate device map",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )
    args = parser.parse_args()
    # get device
    assert torch.cuda.is_available()
    device = "cuda"
    args.devices = [device]  # needed for perplexity eval

    # create original model
    orig_model = get_model(args.base_model, None, args.dtype, args.device_map, trust_remote_code=args.trust_remote_code)
    if not args.device_map:
        orig_model = orig_model.to(device)
    del orig_model
    # ???

    torch.cuda.empty_cache()
    quant_model = get_model(
        args.base_model, args.quant_model, args.dtype, args.device_map, trust_remote_code=args.trust_remote_code
    )
    if not args.device_map:
        quant_model = quant_model.to(device)

    # offload model to cpu
    quant_model = quant_model.cpu()
    if args.device_map:
        remove_hook_from_submodules(quant_model)
    torch.cuda.empty_cache()

    print("\n============ Evaluating perplexity... ============")
    torch.cuda.reset_peak_memory_stats()
    for dataset in args.eval_datasets:
        testloader = get_loaders(
            dataset,
            seed=args.seed,
            model_path=args.base_model,
            seqlen=args.eval_model_seqlen or args.model_seqlen,
            eval_mode=True,
            use_fast_tokenizer=args.use_fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
        args.dataset_name = dataset
        perplexity_eval(quant_model, testloader, args)
        # make sure that the cache is released
        torch.cuda.empty_cache()

    print(f"eval: {torch.cuda.max_memory_allocated()=:,}")
