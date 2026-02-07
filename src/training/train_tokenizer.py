import argparse
import os
from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

def _mlflow_env_setup(mlflow_tracking_uri, mlflow_experiment):
    if mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    if mlflow_experiment:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment


def _mlflow_log_params_and_artifacts(params: dict, tokenizer_dir: str, tokenizer_filename: str = "tokenizer.json"):
    try:
        import mlflow  # type: ignore

        if mlflow.active_run() is None:
            return

        for k, v in params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass

        path = os.path.join(tokenizer_dir, tokenizer_filename)
        if os.path.isfile(path):
            # If the tracking server is configured with a local `file:` artifact store that isn't
            # writable/accessible from this client (common with remote servers or containers),
            # artifact logging will fail. In that case, keep params but skip artifacts.
            artifact_uri = None
            try:
                artifact_uri = mlflow.get_artifact_uri()
            except Exception:
                artifact_uri = None

            if artifact_uri and artifact_uri.startswith("file:"):
                print(
                    "MLflow artifact logging skipped (artifact store is local file://).\n"
                    "Fix by starting MLflow server with artifact proxying, e.g.:\n"
                    "  mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts"
                )
                return

            try:
                mlflow.log_artifact(path, artifact_path="tokenizer")
            except OSError as e:
                # Read-only / permission errors
                if getattr(e, "errno", None) in (30, 13):
                    print(
                        "MLflow artifact logging skipped (artifact store not writable).\n"
                        "Fix by starting MLflow server with artifact proxying, e.g.:\n"
                        "  mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts"
                    )
                else:
                    print(f"MLflow artifact logging skipped: {e}")
            except Exception as e:
                print(f"MLflow artifact logging skipped: {e}")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")


def train_tokenizer(
    en_path,
    bn_path,
    output_dir,
    vocab_size=32000,
    mlflow_tracking_uri=None,
    mlflow_experiment=None,
    run_name=None,
):
    _mlflow_env_setup(mlflow_tracking_uri, mlflow_experiment)
    print(f"Loading datasets...")
    ds_en = load_from_disk(en_path)
    ds_bn = load_from_disk(bn_path)
    
    print(f"English samples: {len(ds_en)}")
    print(f"Bengali samples: {len(ds_bn)}")
    
    # Iterator for training
    def batch_iterator():
        batch_size = 1000
        for i in range(0, len(ds_en), batch_size):
            yield ds_en[i : i + batch_size]["text"]
        for i in range(0, len(ds_bn), batch_size):
            yield ds_bn[i : i + batch_size]["text"]

    # Initialize Tokenizer (BPE)
    print("Initializing BPE Tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # Pre-normalization (create clean splits)
    # Using specific pre-tokenizers isn't strictly necessary if data is well cleaned, 
    # but generic whitespace splitting is good for BPE.
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    print("Training tokenizer (this may take a moment)...")
    try:
        import mlflow  # type: ignore
    except Exception:
        mlflow = None

    def _train_and_save_and_log():
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

        # Post-processor (optional, but good for BERT/RoBERTa style)
        # For T5/Seq2Seq, standard might be simpler. Let's stick to simple BPE for now.

        # Decoder
        tokenizer.decoder = decoders.ByteLevel()  # or BPEDecoder

        # Save
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "tokenizer.json")
        print(f"Saving tokenizer to {save_path}...")
        tokenizer.save(save_path)
        print("Tokenizer saved.")

        # MLflow: log key params + tokenizer artifact
        _mlflow_log_params_and_artifacts(
            params={
                "en_path": en_path,
                "bn_path": bn_path,
                "vocab_size": vocab_size,
                "en_samples": len(ds_en),
                "bn_samples": len(ds_bn),
                "output_dir": output_dir,
            },
            tokenizer_dir=output_dir,
            tokenizer_filename="tokenizer.json",
        )

    if mlflow is None:
        _train_and_save_and_log()
    else:
        with (mlflow.start_run(run_name=run_name) if run_name else mlflow.start_run()):
            _train_and_save_and_log()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_path", default="data/cleaned_wikitext_train", help="Path to English dataset")
    parser.add_argument("--bn_path", default="data/cleaned_wikipedia_bn_train", help="Path to Bengali dataset")
    parser.add_argument("--output_dir", default="models/tokenizer", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")

    # MLflow (optional)
    parser.add_argument("--mlflow_tracking_uri", default=None, help="e.g. http://localhost:5001")
    parser.add_argument("--mlflow_experiment", default=None, help="MLflow experiment name")
    parser.add_argument("--run_name", default=None, help="Run name shown in tracking UI")
    
    args = parser.parse_args()
    
    train_tokenizer(
        args.en_path,
        args.bn_path,
        args.output_dir,
        args.vocab_size,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        run_name=args.run_name,
    )
