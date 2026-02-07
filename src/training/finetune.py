import argparse
import os
import inspect
from typing import Optional

import torch
from datasets import load_from_disk
from tokenizers import Tokenizer
from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration

from src.training.dataset import T5TranslationDataset


def _find_hf_checkpoints(search_root: str = "models") -> list[str]:
    """
    Best-effort discovery of local Hugging Face checkpoints.
    We consider a directory a checkpoint if it contains `config.json`.
    """
    candidates: list[str] = []
    if not os.path.isdir(search_root):
        return candidates

    for dirpath, _, filenames in os.walk(search_root):
        if "config.json" in filenames:
            candidates.append(dirpath)
    candidates.sort()
    return candidates


def _mlflow_env_setup(mlflow_tracking_uri: Optional[str], mlflow_experiment: Optional[str]):
    if mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    if mlflow_experiment:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment


def _training_args_mlflow_kwargs(run_name: Optional[str]):
    params = inspect.signature(TrainingArguments.__init__).parameters
    kw = {}
    if "report_to" in params:
        kw["report_to"] = ["mlflow"]
    if run_name and "run_name" in params:
        kw["run_name"] = run_name
    return kw


def _maybe_log_mlflow_artifacts(output_dir: str):
    try:
        import mlflow  # type: ignore

        if mlflow.active_run() is None:
            return
        final_dir = os.path.join(output_dir, "final")
        if os.path.isdir(final_dir):
            mlflow.log_artifacts(final_dir, artifact_path="model")
    except Exception as e:
        print(f"MLflow artifact logging skipped: {e}")


def _maybe_log_mlflow_metric(name: str, value: float):
    try:
        import mlflow  # type: ignore

        if mlflow.active_run() is None:
            return
        mlflow.log_metric(name, value)
    except Exception:
        return


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compute_bleu(
    model: T5ForConditionalGeneration,
    tokenizer: Tokenizer,
    dataset,
    src_lang: str,
    tgt_lang: str,
    prefix: str,
    max_input_length: int,
    max_new_tokens: int,
    max_samples: int,
):
    """
    Compute SacreBLEU on a (small) eval dataset by running `model.generate`.
    """
    try:
        import evaluate  # type: ignore
    except Exception as e:
        print(f"Skipping BLEU (evaluate not available): {e}")
        return None

    metric = evaluate.load("sacrebleu")
    n = min(len(dataset), max_samples)
    if n <= 0:
        return None

    pad_id = tokenizer.token_to_id("[PAD]")
    eos_id = tokenizer.token_to_id("[SEP]")
    if pad_id is None or eos_id is None:
        print("Skipping BLEU (tokenizer missing [PAD]/[SEP]).")
        return None

    dev = _device()
    model = model.to(dev)
    model.eval()

    preds: list[str] = []
    refs: list[list[str]] = []

    with torch.no_grad():
        for i in range(n):
            row = dataset[i]
            tr = row.get("translation") or {}
            if not isinstance(tr, dict):
                tr = {}

            src_text = tr.get(src_lang, "") or ""
            tgt_text = tr.get(tgt_lang, "") or ""
            src_text = f"{prefix}{src_text}" if prefix else src_text

            enc = tokenizer.encode(src_text)
            ids = enc.ids
            if len(ids) > max_input_length - 1:
                ids = ids[: max_input_length - 1]
            ids = ids + [eos_id]

            input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
            attention_mask = torch.ones_like(input_ids, device=dev)

            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
            pred_text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
            preds.append(pred_text)
            refs.append([tgt_text])

    score = metric.compute(predictions=preds, references=refs)
    return score


def finetune(
    parallel_path: str = "data/cleaned_banglanmt_parallel_train",
    eval_parallel_path: Optional[str] = "data/cleaned_banglanmt_parallel_test",
    src_lang: str = "en",
    tgt_lang: str = "bn",
    init_model_dir: str = "models/checkpoints/final",
    output_dir: str = "models/finetuned_en_bn",
    tokenizer_path: str = "models/tokenizer/tokenizer.json",
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 3e-4,
    max_input_length: int = 128,
    max_target_length: int = 128,
    val_size: float = 0.01,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    prefix: str = "translate English to Bengali: ",
    compute_bleu: bool = True,
    metric_max_samples: int = 200,
    metric_max_new_tokens: int = 128,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    run_name: Optional[str] = None,
):
    _mlflow_env_setup(mlflow_tracking_uri, mlflow_experiment)
    if not os.path.isdir(init_model_dir):
        found = _find_hf_checkpoints("models")
        if found:
            # If `final/` wasn't produced (e.g. interrupted run), fall back to latest checkpoint.
            latest = found[-1]
            print(
                f"WARNING: init_model_dir not found: {init_model_dir}\n"
                f"Falling back to latest detected checkpoint: {latest}"
            )
            init_model_dir = latest
        else:
            raise FileNotFoundError(
                f"init_model_dir not found: {init_model_dir}. "
                "Run pre-training first (src.training.train) or pass --init_model_dir.\n\n"
                "No local Hugging Face checkpoints were found under `models/`.\n"
                "Create one by running pre-training first, e.g.:\n"
                "  venv/bin/python -m src.training.train --output_dir models/checkpoints\n"
                "Then fine-tune with:\n"
                "  --init_model_dir models/checkpoints/final"
            )

    print("Loading tokenizer...")
    tok = Tokenizer.from_file(tokenizer_path)
    pad_token_id = tok.token_to_id("[PAD]")
    eos_token_id = tok.token_to_id("[SEP]")
    if pad_token_id is None:
        raise ValueError("Tokenizer is missing required special token [PAD].")
    if eos_token_id is None:
        raise ValueError("Tokenizer is missing required special token [SEP] (used as EOS).")

    print(f"Loading initial model from {init_model_dir} ...")
    model = T5ForConditionalGeneration.from_pretrained(init_model_dir)
    model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = eos_token_id
    model.config.decoder_start_token_id = pad_token_id

    print(f"Loading cleaned train parallel dataset from {parallel_path} ...")
    ds_train_full = load_from_disk(parallel_path)
    print(f"Train examples (full): {len(ds_train_full)}")

    ds_eval = None
    if eval_parallel_path:
        if os.path.isdir(eval_parallel_path):
            print(f"Loading cleaned test/val parallel dataset from {eval_parallel_path} ...")
            ds_eval = load_from_disk(eval_parallel_path)
            print(f"Eval examples (full): {len(ds_eval)}")
        else:
            print(f"WARNING: eval_parallel_path not found: {eval_parallel_path}. Falling back to train split.")

    if ds_eval is None:
        if val_size <= 0 or val_size >= 1:
            raise ValueError("--val_size must be in (0, 1).")
        print(f"Creating train/val split from train (val_size={val_size}, seed={seed}) ...")
        split = ds_train_full.train_test_split(test_size=val_size, seed=seed, shuffle=True)
        ds_train = split["train"]
        ds_val = split["test"]
    else:
        ds_train = ds_train_full
        ds_val = ds_eval

    if max_train_samples is not None:
        ds_train = ds_train.select(range(min(max_train_samples, len(ds_train))))
    if max_eval_samples is not None:
        ds_val = ds_val.select(range(min(max_eval_samples, len(ds_val))))

    print(f"Train examples: {len(ds_train)} | Val examples: {len(ds_val)}")

    train_dataset = T5TranslationDataset(
        tokenizer_path=tokenizer_path,
        parallel_path=parallel_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        prefix=prefix,
        dataset=ds_train,
    )
    eval_dataset = T5TranslationDataset(
        tokenizer_path=tokenizer_path,
        parallel_path=parallel_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        prefix=prefix,
        dataset=ds_val,
    )

    # transformers renamed `evaluation_strategy` -> `eval_strategy` in newer versions.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_kw = {}
    if "evaluation_strategy" in ta_params:
        eval_kw["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_params:
        eval_kw["eval_strategy"] = "steps"

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        **eval_kw,
        **_training_args_mlflow_kwargs(run_name),
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        prediction_loss_only=True,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Fine-tuning complete. Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final"))
    _maybe_log_mlflow_artifacts(output_dir)

    if compute_bleu:
        print(f"Computing BLEU on up to {metric_max_samples} eval samples...")
        bleu = _compute_bleu(
            model=model,
            tokenizer=tok,
            dataset=ds_val,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            prefix=prefix,
            max_input_length=max_input_length,
            max_new_tokens=metric_max_new_tokens,
            max_samples=metric_max_samples,
        )
        if bleu and isinstance(bleu, dict) and "score" in bleu:
            print(f"BLEU: {bleu['score']}")
            try:
                _maybe_log_mlflow_metric("bleu", float(bleu["score"]))
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune on cleaned parallel translation data.")
    parser.add_argument("--parallel_path", default="data/cleaned_banglanmt_parallel_train")
    parser.add_argument("--eval_parallel_path", default="data/cleaned_banglanmt_parallel_test")
    parser.add_argument("--src_lang", default="en")
    parser.add_argument("--tgt_lang", default="bn")
    parser.add_argument("--init_model_dir", default="models/checkpoints/final")
    parser.add_argument("--output_dir", default="models/finetuned_en_bn")
    parser.add_argument("--tokenizer_path", default="models/tokenizer/tokenizer.json")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--prefix", default="translate English to Bengali: ")

    # Metrics (optional)
    parser.add_argument("--no_compute_bleu", action="store_true", help="Disable BLEU computation")
    parser.add_argument("--metric_max_samples", type=int, default=200, help="Max eval samples used for BLEU")
    parser.add_argument("--metric_max_new_tokens", type=int, default=128, help="Generation max_new_tokens for BLEU")

    # MLflow (optional)
    parser.add_argument("--mlflow_tracking_uri", default=None, help="e.g. http://localhost:5001")
    parser.add_argument("--mlflow_experiment", default=None, help="MLflow experiment name")
    parser.add_argument("--run_name", default=None, help="Run name shown in tracking UI")

    a = parser.parse_args()
    finetune(
        parallel_path=a.parallel_path,
        eval_parallel_path=a.eval_parallel_path,
        src_lang=a.src_lang,
        tgt_lang=a.tgt_lang,
        init_model_dir=a.init_model_dir,
        output_dir=a.output_dir,
        tokenizer_path=a.tokenizer_path,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        max_input_length=a.max_input_length,
        max_target_length=a.max_target_length,
        val_size=a.val_size,
        seed=a.seed,
        max_train_samples=a.max_train_samples,
        max_eval_samples=a.max_eval_samples,
        prefix=a.prefix,
        compute_bleu=(not a.no_compute_bleu),
        metric_max_samples=a.metric_max_samples,
        metric_max_new_tokens=a.metric_max_new_tokens,
        mlflow_tracking_uri=a.mlflow_tracking_uri,
        mlflow_experiment=a.mlflow_experiment,
        run_name=a.run_name,
    )

