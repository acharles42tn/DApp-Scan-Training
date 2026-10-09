"""Command-line entry point: ``python -m scvd_dapp <command>``.

Commands
--------
* ``build    --raw <DAppSCAN checkout>``      build data/dappscan_v1.parquet (CPU, ~1 min)
* ``balance  --in <table> --out <table>``      v3: keep every labelled file, thin unlabelled ones per fold
* ``train    --config configs/<model>.yaml``   fine-tune + evaluate one model (GPU)
* ``baseline [--parquet ...]``                 file-length-only baseline (CPU, seconds)
* ``moe      --config configs/moe.yaml``       gate over finished single-model runs (CPU, minutes)
* ``collect  [--outputs outputs]``             one results table over every finished run

Environment (caches, log silencing) is configured before any heavy import.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .env import configure_environment


def _add_train(sub):
    p = sub.add_parser("train", help="fine-tune + evaluate one model from a YAML config")
    p.add_argument("--config", required=True)
    p.add_argument("--parquet-path", dest="parquet_path")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--run-name", dest="run_name")
    p.add_argument("--max-samples", dest="max_samples", type=int, help="cap FILES per split (smoke tests)")
    p.add_argument("--epochs", dest="num_epochs", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--learning-rate", dest="learning_rate", type=float)
    p.add_argument("--max-length", dest="max_length", type=int)
    p.add_argument("--input-mode", dest="input_mode", choices=["window", "truncate"])
    p.add_argument("--test-fold", dest="test_fold", type=int)
    p.add_argument("--val-fold", dest="val_fold", type=int)
    p.add_argument("--seed", dest="seed", type=int)
    p.add_argument("--resume", dest="resume_from_checkpoint", action="store_true", default=None)
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=None,
                   help="tokenize + window every split, write windows_report.json, exit (no model)")


def _add_baseline(sub):
    p = sub.add_parser("baseline", help="file-length-only baseline (sanity floor)")
    p.add_argument("--parquet-path", dest="parquet_path", default=None)
    p.add_argument("--output-dir", dest="output_dir", default="./outputs/length_baseline")
    p.add_argument("--test-fold", type=int, default=0)
    p.add_argument("--val-fold", type=int, default=1)


def _add_collect(sub):
    p = sub.add_parser("collect", help="summarize every outputs/*/test_results.json into one table")
    p.add_argument("--outputs", default="./outputs")


def _add_v2(sub):
    p = sub.add_parser("tfidf", help="v2: TF-IDF + logistic regression on whole files (CPU, ~2 min)")
    p.add_argument("--parquet-path", dest="parquet_path", default=None)
    p.add_argument("--output-dir", dest="output_dir", required=True)
    p.add_argument("--test-fold", type=int, default=0)
    p.add_argument("--val-fold", type=int, default=1)
    p = sub.add_parser("blend", help="v2: blend two finished runs of the same fold (CPU, seconds)")
    p.add_argument("--runs", nargs=2, required=True, metavar=("RUN_A", "RUN_B"))
    p.add_argument("--output-dir", dest="output_dir", required=True)
    p.add_argument("--parquet-path", dest="parquet_path", default=None)
    p = sub.add_parser("cvsummary", help="v2: mean +- sd over folds of every <model>_f<k> run")
    p.add_argument("--outputs", default="./outputs/v2")


def _add_moe(sub):
    p = sub.add_parser("moe", help="fit the MoE gate over finished single-model runs (CPU)")
    p.add_argument("--config", default="configs/moe.yaml")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="scvd_dapp", description="DAppSCAN smart-contract weakness detection")
    sub = parser.add_subparsers(dest="command", required=True)
    from . import balance, build_dataset

    build_dataset.add_cli(sub)
    balance.add_cli(sub)
    _add_train(sub)
    _add_baseline(sub)
    _add_moe(sub)
    _add_collect(sub)
    _add_v2(sub)
    args = parser.parse_args(argv)

    if args.command == "build":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                            stream=sys.stdout)
        build_dataset.run_cli(args)

    elif args.command == "balance":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                            stream=sys.stdout)
        balance.run_cli(args)

    elif args.command == "train":
        from .callbacks import setup_logging
        from .config import TrainConfig

        overrides = {k: v for k, v in vars(args).items() if k not in ("command", "config") and v is not None}
        if "run_name" in overrides and "output_dir" not in overrides:
            overrides["output_dir"] = f"./outputs/{overrides['run_name']}"
        config = TrainConfig.from_yaml(args.config, overrides)
        configure_environment(config.cache_dir)
        logger = setup_logging(config.log_dir, config.run_name)
        from .env import describe_runtime

        logger.info("Runtime: %s", describe_runtime())
        from .train import run

        run(config)

    elif args.command == "baseline":
        from .callbacks import setup_logging
        from .config import _default_parquet
        from .baselines import run_length_baseline

        setup_logging("./logs", "length_baseline")
        run_length_baseline(args.parquet_path or _default_parquet(), args.output_dir, args.test_fold, args.val_fold)

    elif args.command == "moe":
        from .callbacks import setup_logging
        from .moe import MoEConfig, run_moe

        mcfg = MoEConfig.from_yaml(args.config)
        configure_environment()
        setup_logging(mcfg.log_dir, mcfg.run_name)
        run_moe(mcfg)

    elif args.command == "tfidf":
        from pathlib import Path

        from .callbacks import setup_logging
        from .config import _default_parquet
        from .tfidf import run_tfidf

        setup_logging("./logs", f"tfidf_{Path(args.output_dir).name}")
        run_tfidf(args.parquet_path or _default_parquet(), args.output_dir, args.test_fold, args.val_fold)

    elif args.command == "blend":
        from pathlib import Path

        from .blend import run_blend
        from .callbacks import setup_logging

        setup_logging("./logs", f"blend_{Path(args.output_dir).name}")
        run_blend(args.runs[0], args.runs[1], args.output_dir, args.parquet_path)

    elif args.command == "cvsummary":
        from .collect import collect_cv

        print(collect_cv(args.outputs))

    elif args.command == "collect":
        from .collect import collect

        print(collect(args.outputs))


if __name__ == "__main__":
    main()
