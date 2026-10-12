"""
CLI entry point – run a single query through the pipeline.

Usage:
    python3 -m core.cli "Does aspirin reduce the risk of heart attack?"

    # Use Anthropic instead:
    SATTVIC_LLM_PROVIDER=anthropic python3 -m core.cli "..."

    # Use local Ollama:
    SATTVIC_LLM_PROVIDER=ollama python3 -m core.cli "..."
"""
from __future__ import annotations

# Load .env first, before any other imports that read env vars
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)   # .env always takes precedence
except ImportError:
    pass  # python-dotenv not installed; env vars must be set manually

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.pipeline import Pipeline, PipelineConfig
from core.tracer import tracer

console = Console()


def _render_result(result: "BuddhiResult", run_id: str) -> None:  # noqa: F821
    from pramana.schemas import AdhyasaLabel
    claim   = result.claim
    verdict = result.verdict

    colour_map = {
        "accept":          "green",
        "abstain":         "yellow",
        "flag_and_accept": "orange3",
        "reject":          "red",
    }
    colour = colour_map.get(verdict.value, "white")

    # Argument table
    tbl = Table(show_header=False, box=None, padding=(0, 1))
    tbl.add_column(style="dim", width=18)
    tbl.add_column()
    tbl.add_row("Thesis (Claim)",   claim.pratijna)
    tbl.add_row("Ground (Hetu)",     claim.hetu)
    tbl.add_row("Rule (Udāharaṇa)",  claim.udaharana)
    tbl.add_row("Case Application",  claim.upanaya)
    tbl.add_row("Conclusion",        f"[bold]{claim.nigamana}[/bold]")
    tbl.add_row("",                  "")
    tbl.add_row("Evidence Modality", f"[cyan]{claim.pramana.value}[/cyan]")
    tbl.add_row("Confidence",        f"{claim.confidence:.0%}")
    tbl.add_row("Verdict",           f"[{colour}]{verdict.value.upper()}[/{colour}]")

    if claim.adhyasa_flags:
        flags_str = ", ".join(f.value for f in claim.adhyasa_flags)
        tbl.add_row("Fallacies Flagged ⚠", f"[red]{flags_str}[/red]")

    if result.reasons:
        tbl.add_row("Reasons", "; ".join(result.reasons))

    console.print(Panel(tbl, title=f"[bold]Epistemic Engine · run {run_id[:8]}[/bold]", border_style=colour))


def main() -> None:
    if len(sys.argv) < 2:
        console.print("[red]Usage:[/red]  python -m core.cli \"<your query>\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    console.rule("[bold cyan]Epistemic Engine[/bold cyan]")
    console.print(f"Query: [italic]{query}[/italic]\n")

    pipeline = Pipeline(PipelineConfig())

    # Run through the pipeline
    result = pipeline.run(query)
    
    # Extract run_id from tracer
    run_id = tracer._current_run_id if hasattr(tracer, '_current_run_id') and tracer._current_run_id else "unknown"

    _render_result(result, run_id)

    console.print(f"\n[dim]Trace written to traces/ (run_id: {run_id})[/dim]")


if __name__ == "__main__":
    main()
