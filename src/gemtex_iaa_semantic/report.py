"""Run directories, Markdown tables and the exclusion report."""
from __future__ import annotations

import itertools
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)


def run_directory(base: Path, command: str) -> Path:
    """A new directory ``<timestamp>_<command>``; if the second is taken, the next free one (runs sort by time)."""
    now = datetime.now()
    for seconds in itertools.count():
        path = Path(base) / f"{now + timedelta(seconds=seconds):%Y%m%d-%H%M%S}_{command}"
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            continue


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.3f}", max_rows: int = 40) -> str:
    """Small dependency-free Markdown table."""
    if df is None or df.empty:
        return "_(empty)_\n"
    df = df.head(max_rows)
    index = df.index.name is not None or not isinstance(df.index, pd.RangeIndex)
    frame = df.reset_index() if index else df

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return "" if math.isnan(v) else floatfmt.format(v)
        return str(v).replace("|", "\\|")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, sep, *rows]) + "\n"


def write_exclusion_report(out_dir: Path, rules: Iterable[str], settings: dict, tables: dict,
                           removed_annotators: Iterable[str]) -> Path:
    """``exclusion_report.md``: rules, settings, counts and the full lists of excluded and low-change annotation
    documents (``tables`` as returned by ``pipeline.selection_tables``)."""
    removed = list(removed_annotators)
    excluded = tables["excluded_annotation_documents"].drop(columns="user")
    low = tables["low_change_annotation_documents"].drop(columns="user")
    too_few = tables.get("too_few_annotators_documents", pd.DataFrame())
    lines = ["# Exclusion report\n", f"_generated {datetime.now():%Y-%m-%d %H:%M}_\n",
             "Annotation documents (document x annotator) left out before any analysis. All figures of the run are "
             "computed without them.\n",
             "## Rules\n", *[f"* {rule}" for rule in rules], "",
             "## Settings\n", *[f"* {k}: {v}" for k, v in settings.items()], "",
             "## Summary\n", markdown_table(tables["exclusion_summary"]),
             f"Annotators without any document left (not analysed): {', '.join(removed) if removed else 'none'}\n",
             "### Per annotator\n", markdown_table(tables["excluded_by_annotator"]),
             f"## Excluded annotation documents ({len(excluded)})\n",
             markdown_table(excluded, max_rows=max(1, len(excluded))),
             f"## Low-change annotation documents ({len(low)})\n",
             "F1 against the pre-annotation of at least 1 - PREANNO_LOW_CHANGE. They are analysed unless "
             "`excluded_as` names a reason (EXCLUDE_LOW_CHANGE, or excluded as a copy).\n",
             markdown_table(low, max_rows=max(1, len(low))),
             f"## Documents with too few annotators left ({len(too_few)})\n",
             "Fewer than MIN_ANNOTATORS annotators after the exclusions above; dropped with their remaining "
             "annotation documents (`annotators`: how many were left).\n",
             markdown_table(too_few, max_rows=max(1, len(too_few)))]
    path = Path(out_dir) / "exclusion_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Exclusion report written to %s", path)
    return path
