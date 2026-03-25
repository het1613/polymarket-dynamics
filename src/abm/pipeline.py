"""Stage-wise RQ4 ABM pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.abm.config import (
    ABM_OUTPUT_DIR,
    CalibrationConfig,
    SimulationParams,
    STAGE1_SEARCH_SPACE,
    STAGE2_SEARCH_SPACE,
    STAGE3_SEARCH_SPACE,
    ensure_output_dirs,
)
from src.abm.empirical import load_empirical_reference, write_empirical_reference
from src.abm.evaluation import (
    latin_hypercube_draws,
    needs_stage3,
    save_metric_tables,
    save_params,
    save_score_table,
    stage1_score,
    stage2_score,
    stage3_score,
    stage_match_flags,
    compute_panel_metrics,
)
from src.abm.plots import (
    plot_acf_comparison,
    plot_activity_heatmap,
    plot_price_paths,
    plot_quantile_regression,
    plot_reversal_comparison,
    plot_tail_quantiles,
    plot_trade_rate_timeseries,
)
from src.abm.simulator import simulate_panel


StageScorer = Callable[[dict[str, Any], dict[str, Any]], tuple[float, dict[str, float]]]


def _stage_dir(stage: str) -> Path:
    return ABM_OUTPUT_DIR / stage


def _evaluate_candidate(
    params: SimulationParams,
    empirical_metrics: dict[str, Any],
    skeletons,
    scorer: StageScorer,
    seeds: tuple[int, ...],
    sample_n_for_quantile: int,
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    scores = []
    for seed in seeds:
        panel = simulate_panel(skeletons, params, seed=seed)
        metrics = compute_panel_metrics(
            panel,
            sample_n_for_quantile=sample_n_for_quantile,
        )
        score, parts = scorer(metrics, empirical_metrics)
        rows.append({"seed": seed, "score": score, **parts})
        scores.append(score)
    return float(sum(scores) / max(len(scores), 1)), rows


def _run_search(
    *,
    base_params: SimulationParams,
    stage: str,
    search_space: dict[str, tuple[float, float]],
    n_draws: int,
    random_seed: int,
    empirical_metrics: dict[str, Any],
    skeletons,
    scorer: StageScorer,
    calibration_seeds: tuple[int, ...],
    sample_n_for_quantile: int,
) -> tuple[SimulationParams, list[dict[str, Any]]]:
    draws = latin_hypercube_draws(search_space, n_draws, random_seed)
    results: list[dict[str, Any]] = []
    best_score = float("inf")
    best_params = base_params.with_updates(stage=stage)

    print(f"[{stage}] calibration search with {len(draws)} candidates", flush=True)
    for idx, draw in enumerate(draws):
        params = base_params.with_updates(stage=stage, **draw)
        avg_score, per_seed_rows = _evaluate_candidate(
            params,
            empirical_metrics,
            skeletons,
            scorer,
            calibration_seeds,
            sample_n_for_quantile,
        )
        mean_parts = {
            key: float(pd.DataFrame(per_seed_rows)[key].mean())
            for key in per_seed_rows[0]
            if key not in {"seed", "score"}
        }
        result_row = {
            "candidate_id": idx,
            "stage": stage,
            "avg_score": avg_score,
            **draw,
            **mean_parts,
        }
        results.append(result_row)
        print(f"[{stage}] candidate {idx + 1}/{len(draws)} avg_score={avg_score:.4f}", flush=True)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
            print(f"[{stage}] new best score={best_score:.4f}", flush=True)

    return best_params, results


def _finalise_stage(
    *,
    stage: str,
    params: SimulationParams,
    empirical_metrics: dict[str, Any],
    skeletons,
    scorer: StageScorer,
    final_seeds: tuple[int, ...],
) -> dict[str, Any]:
    stage_dir = _stage_dir(stage)
    stage_dir.mkdir(parents=True, exist_ok=True)

    per_seed_records = []
    best_panel = None
    best_metrics = None
    best_score = float("inf")

    print(f"[{stage}] final evaluation on {len(final_seeds)} seeds", flush=True)
    for seed in final_seeds:
        panel = simulate_panel(skeletons, params, seed=seed)
        metrics = compute_panel_metrics(panel, sample_n_for_quantile=None)
        score, parts = scorer(metrics, empirical_metrics)
        per_seed_records.append({"seed": seed, "score": score, **parts})
        print(f"[{stage}] seed={seed} score={score:.4f}", flush=True)
        if score < best_score:
            best_score = score
            best_panel = panel
            best_metrics = metrics

    assert best_panel is not None
    assert best_metrics is not None

    best_panel.to_parquet(stage_dir / "features.parquet", index=False)
    pd.DataFrame(per_seed_records).to_csv(stage_dir / "final_seed_scores.csv", index=False)
    save_metric_tables(stage_dir, best_metrics)
    save_params(stage_dir, params.to_dict())
    return {
        "panel": best_panel,
        "metrics": best_metrics,
        "final_seed_scores": pd.DataFrame(per_seed_records),
        "stage_dir": stage_dir,
    }


def _write_stage_summary(
    *,
    stage: str,
    stage_dir: Path,
    params: SimulationParams,
    empirical_metrics: dict[str, Any],
    simulated_metrics: dict[str, Any],
    next_stage_justified: bool,
    next_stage_reason: str,
) -> None:
    if stage == "stage1":
        score, parts = stage1_score(simulated_metrics, empirical_metrics)
        mechanisms = [
            "latent anchor with Gaussian drift plus occasional jumps",
            "three trader classes only: informed, noise, herding",
            "base participation with deterministic deadline ramp",
            "single reduced-form nonlinear impact rule with static effective depth",
            "hard cash, inventory, position, and per-step order caps",
            "explicit no-move threshold",
        ]
    elif stage == "stage2":
        score, parts = stage2_score(simulated_metrics, empirical_metrics)
        mechanisms = [
            "all stage-1 mechanisms",
            "participation responds to recent volatility and recent activity",
            "effective depth varies with activity, deadline pressure, and their interaction",
        ]
    else:
        score, parts = stage3_score(simulated_metrics, empirical_metrics)
        mechanisms = [
            "all stage-2 mechanisms",
            "thin-market transient dislocation state that decays over time",
            "dislocation shocks only activate in low-activity, large-imbalance states",
        ]

    flags = stage_match_flags(stage, simulated_metrics, empirical_metrics)
    matched = [name.replace("_", " ") for name, ok in flags.items() if ok]
    failed = [name.replace("_", " ") for name, ok in flags.items() if not ok]

    lines = [
        f"# {stage.upper()} Summary",
        "",
        "## Mechanisms Present",
        *[f"- {item}" for item in mechanisms],
        "",
        "## Empirical Targets Matched",
    ]
    lines.extend([f"- {item}" for item in matched] if matched else ["- None clearly."])
    lines.extend(
        [
            "",
            "## Targets Still Missing or Weak",
        ]
    )
    lines.extend([f"- {item}" for item in failed] if failed else ["- None material."])
    lines.extend(
        [
            "",
            "## Calibration Snapshot",
            f"- total score: {score:.4f}",
        ]
    )
    for key, value in parts.items():
        if key == "total_score":
            continue
        lines.append(f"- {key.replace('_', ' ')}: {value:.4f}")
    lines.extend(
        [
            "",
            "## Next Stage Decision",
            f"- justified: {'yes' if next_stage_justified else 'no'}",
            f"- reason: {next_stage_reason}",
        ]
    )

    (stage_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_pipeline_summary(
    output_dir: Path,
    *,
    stage1_done: dict[str, Any],
    stage2_done: dict[str, Any],
    stage3_done: dict[str, Any] | None,
) -> None:
    stage1_score = float(stage1_done["final_seed_scores"]["score"].min())
    stage2_score = float(stage2_done["final_seed_scores"]["score"].min())
    stage3_score = (
        None if stage3_done is None else float(stage3_done["final_seed_scores"]["score"].min())
    )
    minimal_answer = "stage2"
    if stage3_score is not None and stage3_score < stage2_score:
        minimal_answer = "stage3"

    summary = {
        "stage1_score": stage1_score,
        "stage2_score": stage2_score,
        "stage3_ran": stage3_done is not None,
        "stage3_score": stage3_score,
        "minimal_mechanism_answer": minimal_answer,
    }
    with (output_dir / "pipeline_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def _load_stage_params(stage: str) -> SimulationParams:
    stage_dir = _stage_dir(stage)
    with (stage_dir / "best_params.json").open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return SimulationParams(**payload)


def _load_stage_stub(stage: str) -> dict[str, Any] | None:
    stage_dir = _stage_dir(stage)
    final_seed_scores = stage_dir / "final_seed_scores.csv"
    if not final_seed_scores.exists():
        return None
    return {
        "stage_dir": stage_dir,
        "final_seed_scores": pd.read_csv(final_seed_scores),
    }


def _write_stage_plots(
    *,
    stage: str,
    panel: pd.DataFrame,
    simulated_metrics: dict[str, Any],
    empirical_metrics: dict[str, Any],
) -> None:
    plot_tail_quantiles(
        empirical_metrics["tail_quantiles"],
        simulated_metrics["tail_quantiles"],
        stage,
    )
    plot_acf_comparison(
        empirical_metrics["acf_abs"].loc[1:],
        simulated_metrics["acf_abs"].loc[1:],
        stage,
    )
    plot_price_paths(panel, stage)
    plot_trade_rate_timeseries(panel, stage)
    plot_activity_heatmap(
        empirical_metrics["interaction"],
        simulated_metrics["interaction"],
        stage,
    )
    plot_quantile_regression(
        empirical_metrics["quantile_regression_raw"],
        simulated_metrics["quantile_regression_raw"],
        stage,
    )
    plot_reversal_comparison(
        empirical_metrics["reversal_activity"],
        simulated_metrics["reversal_activity"],
        stage,
        "reversal_activity",
    )
    plot_reversal_comparison(
        empirical_metrics["reversal_deadline"],
        simulated_metrics["reversal_deadline"],
        stage,
        "reversal_deadline",
    )


def run_rq4_pipeline(
    *,
    config: CalibrationConfig | None = None,
    start_stage: str = "stage1",
    stop_after: str | None = None,
) -> dict[str, Any]:
    """Run the staged RQ4 ABM pipeline and write staged artifacts."""
    if config is None:
        config = CalibrationConfig()

    ensure_output_dirs()
    reference = load_empirical_reference()
    empirical_panel = reference["panel"]
    empirical_metrics = reference["metrics"]
    skeletons = reference["skeletons"]
    write_empirical_reference(reference)

    stage1_done = None
    stage2_done = None
    stage3_done = None

    if start_stage == "stage1":
        stage1_params, stage1_search = _run_search(
            base_params=SimulationParams(stage="stage1"),
            stage="stage1",
            search_space=STAGE1_SEARCH_SPACE,
            n_draws=config.stage1_draws,
            random_seed=config.random_seed,
            empirical_metrics=empirical_metrics,
            skeletons=skeletons,
            scorer=stage1_score,
            calibration_seeds=config.calibration_seeds,
            sample_n_for_quantile=config.calibration_sample_n,
        )
        save_score_table(_stage_dir("stage1"), stage1_search)
        stage1_done = _finalise_stage(
            stage="stage1",
            params=stage1_params,
            empirical_metrics=empirical_metrics,
            skeletons=skeletons,
            scorer=stage1_score,
            final_seeds=config.final_seeds,
        )
        _write_stage_plots(
            stage="stage1",
            panel=stage1_done["panel"],
            simulated_metrics=stage1_done["metrics"],
            empirical_metrics=empirical_metrics,
        )
        _write_stage_summary(
            stage="stage1",
            stage_dir=stage1_done["stage_dir"],
            params=stage1_params,
            empirical_metrics=empirical_metrics,
            simulated_metrics=stage1_done["metrics"],
            next_stage_justified=True,
            next_stage_reason="Stage 2 is justified because the activity x deadline sign pattern is not present by construction in stage 1.",
        )
    else:
        stage1_params = _load_stage_params("stage1")
        stage1_done = _load_stage_stub("stage1")

    if stop_after == "stage1":
        return {
            "empirical_panel": empirical_panel,
            "empirical_metrics": empirical_metrics,
            "stage1": stage1_done,
            "stage2": None,
            "stage3": None,
        }

    if start_stage in {"stage1", "stage2"}:
        stage2_params, stage2_search = _run_search(
            base_params=stage1_params,
            stage="stage2",
            search_space=STAGE2_SEARCH_SPACE,
            n_draws=config.stage2_draws,
            random_seed=config.random_seed + 1,
            empirical_metrics=empirical_metrics,
            skeletons=skeletons,
            scorer=stage2_score,
            calibration_seeds=config.calibration_seeds,
            sample_n_for_quantile=config.calibration_sample_n,
        )
        save_score_table(_stage_dir("stage2"), stage2_search)
        stage2_done = _finalise_stage(
            stage="stage2",
            params=stage2_params,
            empirical_metrics=empirical_metrics,
            skeletons=skeletons,
            scorer=stage2_score,
            final_seeds=config.final_seeds,
        )
        _write_stage_plots(
            stage="stage2",
            panel=stage2_done["panel"],
            simulated_metrics=stage2_done["metrics"],
            empirical_metrics=empirical_metrics,
        )

        stage3_required = needs_stage3(stage2_done["metrics"])
        _write_stage_summary(
            stage="stage2",
            stage_dir=stage2_done["stage_dir"],
            params=stage2_params,
            empirical_metrics=empirical_metrics,
            simulated_metrics=stage2_done["metrics"],
            next_stage_justified=stage3_required,
            next_stage_reason=(
                "Stage 3 is justified because low-activity reversal remains weaker than the empirical asymmetry."
                if stage3_required
                else "Stage 3 is not required because stage 2 already reproduces the low-activity reversal asymmetry directionally."
            ),
        )
    else:
        stage2_params = _load_stage_params("stage2")
        stage2_done = _load_stage_stub("stage2")
        stage3_required = True

    if stop_after == "stage2":
        return {
            "empirical_panel": empirical_panel,
            "empirical_metrics": empirical_metrics,
            "stage1": stage1_done,
            "stage2": stage2_done,
            "stage3": None,
        }

    if stage3_required:
        stage3_params, stage3_search = _run_search(
            base_params=stage2_params,
            stage="stage3",
            search_space=STAGE3_SEARCH_SPACE,
            n_draws=config.stage3_draws,
            random_seed=config.random_seed + 2,
            empirical_metrics=empirical_metrics,
            skeletons=skeletons,
            scorer=stage3_score,
            calibration_seeds=config.calibration_seeds,
            sample_n_for_quantile=config.calibration_sample_n,
        )
        save_score_table(_stage_dir("stage3"), stage3_search)
        stage3_done = _finalise_stage(
            stage="stage3",
            params=stage3_params,
            empirical_metrics=empirical_metrics,
            skeletons=skeletons,
            scorer=stage3_score,
            final_seeds=config.final_seeds,
        )
        _write_stage_plots(
            stage="stage3",
            panel=stage3_done["panel"],
            simulated_metrics=stage3_done["metrics"],
            empirical_metrics=empirical_metrics,
        )
        _write_stage_summary(
            stage="stage3",
            stage_dir=stage3_done["stage_dir"],
            params=stage3_params,
            empirical_metrics=empirical_metrics,
            simulated_metrics=stage3_done["metrics"],
            next_stage_justified=False,
            next_stage_reason="No further mechanism stage is planned in the minimal RQ4 build.",
        )

    _write_pipeline_summary(
        ABM_OUTPUT_DIR,
        stage1_done=stage1_done,
        stage2_done=stage2_done,
        stage3_done=stage3_done,
    )

    return {
        "empirical_panel": empirical_panel,
        "empirical_metrics": empirical_metrics,
        "stage1": stage1_done,
        "stage2": stage2_done,
        "stage3": stage3_done,
    }


def main() -> None:
    defaults = CalibrationConfig()
    parser = argparse.ArgumentParser(description="Run the staged RQ4 ABM pipeline.")
    parser.add_argument("--stage1-draws", type=int, default=defaults.stage1_draws)
    parser.add_argument("--stage2-draws", type=int, default=defaults.stage2_draws)
    parser.add_argument("--stage3-draws", type=int, default=defaults.stage3_draws)
    parser.add_argument("--calibration-sample", type=int, default=defaults.calibration_sample_n)
    parser.add_argument("--n-calibration-seeds", type=int, default=len(defaults.calibration_seeds))
    parser.add_argument("--n-final-seeds", type=int, default=len(defaults.final_seeds))
    parser.add_argument(
        "--start-stage",
        choices=["stage1", "stage2", "stage3"],
        default="stage1",
    )
    parser.add_argument(
        "--stop-after",
        choices=["stage1", "stage2", "stage3"],
        default=None,
    )
    args = parser.parse_args()

    calibration_seed_pool = (101, 202, 303, 404)
    final_seed_pool = (401, 402, 403, 404)

    cfg = CalibrationConfig(
        stage1_draws=args.stage1_draws,
        stage2_draws=args.stage2_draws,
        stage3_draws=args.stage3_draws,
        calibration_sample_n=args.calibration_sample,
        calibration_seeds=calibration_seed_pool[: max(1, args.n_calibration_seeds)],
        final_seeds=final_seed_pool[: max(1, args.n_final_seeds)],
    )
    run_rq4_pipeline(
        config=cfg,
        start_stage=args.start_stage,
        stop_after=args.stop_after,
    )


if __name__ == "__main__":
    main()
