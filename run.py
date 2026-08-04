"""Single-entry CLI runner for the CS-JT-VAE pipeline.

Usage:
    python run.py <stage> [--config PATH] [-- extra args ...]
    python run.py list
    python run.py all          # generation + verification (needs trained weights)
    python run.py train-all    # full pipeline from the raw dataset (GPU)

Each stage runs one or more scripts under ``scripts/`` with their configs from
``configs/``. Trained weights and the processed dataset are hosted on Zenodo
(see README); ``all`` assumes they are already in place, while ``train-all``
rebuilds them from scratch. Every step auto-logs to ``logs/<stage>/``.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOGS = REPO / "logs"

# Each stage is an ordered list of (script, config-or-None) steps.
STAGES: dict[str, list[tuple[str, str | None]]] = {
    "preprocess":        [("scripts/preprocess_moonshot.py", None)],
    "augment":           [("scripts/augment_dataset.py",     None)],
    "build-vocab":       [("scripts/build_vocab.py",         "configs/jtvae_v3_aug.yaml")],
    "train-jtvae":       [("scripts/train_jtvae.py",         "configs/jtvae_v3_aug.yaml")],
    "train-scorer":      [("scripts/train_scorer.py",        "configs/joint_full_v3_aug.yaml")],
    "train-bioactivity": [("scripts/train_bioactivity.py",   "configs/bioactivity_v3_aug.yaml")],
    "generate":          [("scripts/generate.py",            "configs/nsga_v3_6obj.yaml")],
    "admet":             [("scripts/run_admet.py",           None)],
    "docking":           [("scripts/run_docking.py",         None)],
    "figures":           [("scripts/fig_ames_by_warhead.py",    None),
                          ("scripts/fig_safety_and_docking.py", None),
                          ("scripts/fig_headline.py",           None)],
    "stats":             [("scripts/stats_generation.py",  None),
                          ("scripts/stats_yield.py",       None),
                          ("scripts/stats_nirmatrelvir.py", None),
                          ("scripts/stats_docking.py",     None)],
    "moo-catalogue":     [("scripts/bench_moo.py",   "configs/moo_catalogue_v3.yaml"),
                          ("scripts/analyze_moo.py", None)],
}

TRAIN_PIPELINE = ["preprocess", "augment", "build-vocab",
                  "train-jtvae", "train-scorer", "train-bioactivity"]
INFER_PIPELINE = ["generate", "admet", "docking", "figures", "stats"]


def _run_step(stage: str, script_rel: str, config: str | None,
              extra: list[str]) -> int:
    script = REPO / script_rel
    if not script.exists():
        print(f"[run] script not found: {script}", file=sys.stderr)
        return 1
    cmd: list[str] = [sys.executable, str(script)]
    if config is not None:
        cfg_path = REPO / config
        if not cfg_path.exists():
            print(f"[run] config not found: {cfg_path}", file=sys.stderr)
            return 1
        cmd += ["--config", str(cfg_path)]
    cmd += extra

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = LOGS / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{ts}_{script.stem}.log"
    print(f"[run] >>> {stage}: {script_rel}")
    print("[run]    " + " ".join(cmd))
    print(f"[run]    logging -> {log_path.relative_to(REPO)}")
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# stage: {stage}\n# cmd: {' '.join(cmd)}\n"
                   f"# start (UTC): {ts}\n\n")
        logf.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
            logf.flush()
        rc = proc.wait()
        end = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        logf.write(f"\n# end (UTC): {end}  rc={rc}\n")
    if rc != 0:
        print(f"[run] !!! {stage}/{script.stem} failed (rc={rc})")
    return rc


def _run_stage(stage: str, config_override: str | None, extra: list[str]) -> int:
    steps = STAGES[stage]
    for script_rel, config in steps:
        # A --config override applies to the (single) config-taking step.
        cfg = config_override if (config_override and config is not None) else config
        step_extra = extra if len(steps) == 1 else []
        rc = _run_step(stage, script_rel, cfg, step_extra)
        if rc != 0:
            return rc
    return 0


def _print_stage_table() -> None:
    print(f"{'STAGE':<20} SCRIPTS")
    print("-" * 78)
    for stage, steps in STAGES.items():
        scripts = ", ".join(Path(s).name for s, _ in steps)
        print(f"{stage:<20} {scripts}")
    print()
    print("Pipelines:")
    print(f"  all        {' -> '.join(INFER_PIPELINE)}")
    print(f"  train-all  {' -> '.join(TRAIN_PIPELINE + INFER_PIPELINE)}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage",
                        help="stage name, 'list', 'all', or 'train-all'")
    parser.add_argument("--config", default=None,
                        help="override the config path for the stage's config step")
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.stage == "list":
        _print_stage_table()
        return 0
    if args.stage in {"all", "train-all"}:
        sequence = (INFER_PIPELINE if args.stage == "all"
                    else TRAIN_PIPELINE + INFER_PIPELINE)
        for s in sequence:
            rc = _run_stage(s, None, [])
            if rc != 0:
                return rc
        return 0
    if args.stage not in STAGES:
        print(f"[run] unknown stage: {args.stage}", file=sys.stderr)
        _print_stage_table()
        return 2
    return _run_stage(args.stage, args.config, extra)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
