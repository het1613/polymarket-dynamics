"""Generate the additive RQ4 notebook that reads staged ABM artifacts."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from src.abm.config import ABM_OUTPUT_DIR


def build_notebook(output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = Path("notebooks/05_rq4_abm.ipynb")

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Notebook 05 — RQ4: Staged Politics-First ABM\n\n"
            "This notebook reuses the repo's empirical feature panel and the additive "
            "`src.abm` package to ask whether a minimal heterogeneous-trader ABM can "
            "reproduce the politics-panel signatures and RQ3 conditionals."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n\n"
            "import pandas as pd\n"
            "from IPython.display import Markdown, Image, display\n\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n\n"
            "from src.abm.pipeline import run_rq4_pipeline\n"
            "from src.abm.config import ABM_OUTPUT_DIR, ABM_PLOT_PREFIX\n\n"
            "if not (ABM_OUTPUT_DIR / 'pipeline_summary.json').exists():\n"
            "    run_rq4_pipeline()\n\n"
            "ABM_OUTPUT_DIR"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "def load_stage(stage):\n"
            "    stage_dir = ABM_OUTPUT_DIR / stage\n"
            "    if not stage_dir.exists():\n"
            "        return None\n"
            "    return {\n"
            "        'dir': stage_dir,\n"
            "        'summary': (stage_dir / 'summary.md').read_text(encoding='utf-8') if (stage_dir / 'summary.md').exists() else '',\n"
            "        'features': pd.read_parquet(stage_dir / 'features.parquet') if (stage_dir / 'features.parquet').exists() else None,\n"
            "        'tail_quantiles': pd.read_csv(stage_dir / 'tail_quantiles.csv', index_col=0) if (stage_dir / 'tail_quantiles.csv').exists() else None,\n"
            "        'deadline_summary': pd.read_csv(stage_dir / 'deadline_summary.csv', index_col=0) if (stage_dir / 'deadline_summary.csv').exists() else None,\n"
            "        'interaction': pd.read_csv(stage_dir / 'interaction.csv', index_col=[0, 1]) if (stage_dir / 'interaction.csv').exists() else None,\n"
            "        'quantile_regression_raw': pd.read_csv(stage_dir / 'quantile_regression_raw.csv', index_col=0) if (stage_dir / 'quantile_regression_raw.csv').exists() else None,\n"
            "        'final_seed_scores': pd.read_csv(stage_dir / 'final_seed_scores.csv') if (stage_dir / 'final_seed_scores.csv').exists() else None,\n"
            "    }\n\n"
            "empirical_dir = ABM_OUTPUT_DIR / 'empirical'\n"
            "emp_tail = pd.read_csv(empirical_dir / 'tail_quantiles.csv', index_col=0)\n"
            "emp_deadline = pd.read_csv(empirical_dir / 'deadline_summary.csv', index_col=0)\n"
            "emp_interaction = pd.read_csv(empirical_dir / 'interaction.csv', index_col=[0, 1])\n"
            "emp_qr = pd.read_csv(empirical_dir / 'quantile_regression_raw.csv', index_col=0)\n\n"
            "stage1 = load_stage('stage1')\n"
            "stage2 = load_stage('stage2')\n"
            "stage3 = load_stage('stage3')\n"
            "pipeline_summary = pd.read_json(ABM_OUTPUT_DIR / 'pipeline_summary.json', typ='series')\n"
            "pipeline_summary"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## Empirical Reference")
    )
    cells.append(
        nbf.v4.new_code_cell(
            "display(emp_tail)\n"
            "display(emp_deadline)\n"
            "display(emp_interaction)"
        )
    )

    for stage in ("stage1", "stage2", "stage3"):
        cells.append(nbf.v4.new_markdown_cell(f"## {stage.upper()}"))
        cells.append(
            nbf.v4.new_code_cell(
                f"bundle = {stage}\n"
                "if bundle is None:\n"
                f"    display(Markdown('Stage {stage[-1]} was not run.'))\n"
                "else:\n"
                "    display(Markdown(bundle['summary']))\n"
                "    if bundle['tail_quantiles'] is not None:\n"
                "        display(bundle['tail_quantiles'])\n"
                "    if bundle['deadline_summary'] is not None:\n"
                "        display(bundle['deadline_summary'])\n"
                "    if bundle['interaction'] is not None:\n"
                "        display(bundle['interaction'])\n"
                "    if bundle['quantile_regression_raw'] is not None:\n"
                "        display(bundle['quantile_regression_raw'])\n"
                "    if bundle['final_seed_scores'] is not None:\n"
                "        display(bundle['final_seed_scores'])\n"
            )
        )
        for suffix in (
            "tail_quantiles",
            "acf",
            "price_paths",
            "trade_rate",
            "activity_heatmap",
            "quantile_regression",
            "reversal_activity",
            "reversal_deadline",
        ):
            cells.append(
                nbf.v4.new_code_cell(
                    f"img = Path.cwd().parent / 'plots' / f'{{ABM_PLOT_PREFIX}}_{stage}_{suffix}.png'\n"
                    "if img.exists():\n"
                    "    display(Image(filename=str(img)))"
                )
            )

    cells.append(
        nbf.v4.new_markdown_cell("## Minimal Mechanism Answer")
    )
    cells.append(
        nbf.v4.new_code_cell(
            "pipeline_summary"
        )
    )

    nb["cells"] = cells
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    return output_path


if __name__ == "__main__":
    build_notebook()
