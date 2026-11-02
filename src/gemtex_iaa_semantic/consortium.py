"""Agreement reports of the GeMTeX sites (``site-report``) and their combination (``pool``).

Every site runs ``site-report`` on its own INCEpTION exports, one or several. Only documents with at least
``MIN_ANNOTATORS`` annotators enter (``io.loader``), after the exclusions of ``selection``. The run directory has two
folders:

* ``share/`` is what the site sends back. Documents and annotators appear only as codes (``D0001``, ``A01``). It
  holds no document text; no file, project, account or display names; no concepts; and no paths. Besides the report
  it has, per document and annotator pair, the counts every figure is computed from, so that ``pool`` combines
  sites exactly.
* ``local/`` stays at the site: the same tables with the real names, the exclusion report and the log.

The codes are kept in ``DIR_OUTPUT/pseudonyms.csv`` (local) and stay the same across runs. They are numbered, not
hashed: a hash of a file name could be recovered by guessing the name.

Figures, pooled over documents and annotator pairs (sum of numerators / sum of denominators), with a 95 % percentile
bootstrap CI over documents:

* span F1 strict (same span) and lenient (overlapping spans matched one to one); stacked spans count once;
* end-to-end F1: same span and concept key;
* Cohen's κ on 1:1 exact spans, chance agreement per document and annotator pair;
* nominal Krippendorff's α on clusters of overlapping mentions (at most one per annotator), chance agreement per
  document, pooled like κ;
* the share of EXACT regions whose mentions both annotators kept from the pre-annotation (provenance
  ``inherited``); empty without a pre-annotation.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gemtex_iaa_semantic import align as A
from gemtex_iaa_semantic import measures as M
from gemtex_iaa_semantic import pipeline as P
from gemtex_iaa_semantic import preannotation as PA
from gemtex_iaa_semantic import selection as SEL
from gemtex_iaa_semantic.config import Config
from gemtex_iaa_semantic.io import loader as L
from gemtex_iaa_semantic.report import markdown_table, write_exclusion_report
from gemtex_iaa_semantic.snomed import SnomedStore

log = logging.getLogger(__name__)

COUNTS = ["n_a", "n_b", "strict", "lenient", "e2e_a", "e2e_b", "e2e", "kappa_n", "kappa_agree", "kappa_expected",
          "alpha_n", "alpha_agree", "alpha_expected", "exact", "exact_inherited"]
PAIR_COLUMNS = ["doc", "annotator_a", "annotator_b", *COUNTS]
SHARE_FILES = ["report.md", "summary.json", "figures.csv", "annotators.csv", "annotator_pairs.csv",
               "pair_counts.csv"]


@dataclass(frozen=True)
class Figure:
    name: str
    column: str                 # column in the annotator and pair tables
    num: str
    den: str
    factor: float = 1.0
    n: Optional[str] = None     # the count reported as n (default: den)


FIGURES = [
    Figure("span F1 strict", "f1_strict", "strict", "spans", 2.0),
    Figure("span F1 lenient (overlap)", "f1_lenient", "lenient", "spans", 2.0),
    Figure("end-to-end F1 (span + concept)", "f1_e2e", "e2e", "e2e_mentions", 2.0),
    Figure("Cohen's kappa on 1:1 exact spans (pooled)", "kappa", "kappa_num", "kappa_den", n="kappa_n"),
    Figure("Krippendorff's alpha (nominal), chance per document (pooled)", "alpha", "alpha_num", "alpha_den",
           n="alpha_n"),
    Figure("share of EXACT regions inherited from the pre-annotation", "exact_inherited", "exact_inherited",
           "exact"),
]
DEFINITIONS = [
    "**span F1 strict / lenient**: mentions of two annotators with the same span / with overlapping spans (matched "
    "one to one); n = spans of both annotators, stacked spans counted once.",
    "**end-to-end F1**: mentions with the same span and the same concept (and literal); n = mentions of both.",
    "**Cohen's kappa**: concept agreement on spans each annotator marked exactly once and identically, corrected "
    "for chance per document and annotator pair, pooled; n = such spans.",
    "**Krippendorff's alpha**: nominal α on the concepts of clusters of overlapping mentions (all annotators of a "
    "document) in which no annotator has more than one mention; chance agreement per document, pooled over "
    "documents like κ; n = pairable values (one per annotator in a cluster of at least two).",
    "**share inherited**: of the regions where both annotators agree exactly, the share where both kept the "
    "pre-annotation (INITIAL_CAS) unchanged; n = exactly agreeing regions. High values mean that the agreement "
    "above largely repeats the pre-annotation.",
    "Values are pooled over documents and annotator pairs; the 95 % CI is a percentile bootstrap over documents.",
]


# ---------------- counts ----------------

def pair_counts(ctx) -> pd.DataFrame:
    """Per document and annotator pair (real names, columns doc, user_a, user_b): the counts of all figures.

    ``ctx`` is a ``pipeline.Context``; ``exact_inherited`` is NaN without a pre-annotation."""
    rows: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    pre = ctx.preannotation
    sides = PA.region_pre_side(ctx.regions, pre) if len(pre) else [None] * len(ctx.regions)
    for region, side in zip(ctx.regions, sides):
        row = rows.setdefault((region.doc, region.user_a, region.user_b), dict.fromkeys(COUNTS, 0))
        for key, value in M.span_counts(region).items():
            row[key] += value
        if M.is_exact(region):
            row["exact"] += 1
            row["exact_inherited"] += side == "both"
    one = M.one_to_one_pairs(ctx.regions)
    for k in M.kappa_terms(one[one["region_kind"] == A.EXACT]).itertuples(index=False):
        row = rows[(k.doc, k.user_a, k.user_b)]
        row["kappa_n"], row["kappa_agree"], row["kappa_expected"] = k.n, k.agree, k.expected
    for k in M.alpha_terms(M.alpha_units(ctx.clustered)).itertuples(index=False):
        row = rows[(k.doc, k.user_a, k.user_b)]
        row["alpha_n"], row["alpha_agree"], row["alpha_expected"] = k.n, k.agree, k.expected
    out = pd.DataFrame([{"doc": d, "user_a": a, "user_b": b, **v} for (d, a, b), v in rows.items()],
                       columns=["doc", "user_a", "user_b", *COUNTS])
    if not len(pre):
        out["exact_inherited"] = np.nan
    return out


def _derive(counts: pd.DataFrame) -> pd.DataFrame:
    return counts.assign(spans=counts["n_a"] + counts["n_b"], e2e_mentions=counts["e2e_a"] + counts["e2e_b"],
                         kappa_num=counts["kappa_agree"] - counts["kappa_expected"],
                         kappa_den=counts["kappa_n"] - counts["kappa_expected"],
                         alpha_num=counts["alpha_agree"] - counts["alpha_expected"],
                         alpha_den=counts["alpha_n"] - counts["alpha_expected"])


def figures(counts: pd.DataFrame, n_boot: int = 1000, seed: int = 13, group: str = "doc") -> pd.DataFrame:
    """All figures pooled over ``counts`` (PAIR_COLUMNS) with their bootstrap CI over ``group``."""
    df = _derive(counts)
    rows = []
    for f in FIGURES:
        sub = df.dropna(subset=[f.num, f.den])
        sub = sub[sub[f.n or f.den] > 0]
        point, lo, hi = M.bootstrap_ratio(sub, f.num, f.den, num_factor=f.factor, group=group, n_boot=n_boot, seed=seed)
        rows.append({"figure": f.name, "value": point, "ci_low": lo, "ci_high": hi,
                     "n": int(round(sub[f.n or f.den].sum())), "documents": int(sub[group].nunique()),
                     "pairs": len(sub)})
    return pd.DataFrame(rows, columns=["figure", "value", "ci_low", "ci_high", "n", "documents", "pairs"])


def _values(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for f in FIGURES:
        sub = df.dropna(subset=[f.num, f.den])
        den = sub[f.den].sum()
        out[f.column] = float(f.factor * sub[f.num].sum() / den) if den else float("nan")
    return out


def by_annotator(counts: pd.DataFrame) -> pd.DataFrame:
    """Per annotator: the figures over all document-level pairs the annotator is part of."""
    df = _derive(counts)
    both = pd.concat([df.assign(annotator=df["annotator_a"]), df.assign(annotator=df["annotator_b"])])
    rows = [{"annotator": a, "documents": g["doc"].nunique(), "pairs": len(g), **_values(g)}
            for a, g in both.groupby("annotator", sort=True)]
    return pd.DataFrame(rows, columns=["annotator", "documents", "pairs", *[f.column for f in FIGURES]])


def by_pair(counts: pd.DataFrame) -> pd.DataFrame:
    """Per annotator pair: the figures over their common documents."""
    df = _derive(counts)
    rows = [{"annotator_a": a, "annotator_b": b, "documents": g["doc"].nunique(), **_values(g)}
            for (a, b), g in df.groupby(["annotator_a", "annotator_b"], sort=True)]
    return pd.DataFrame(rows, columns=["annotator_a", "annotator_b", "documents", *[f.column for f in FIGURES]])


# ---------------- pseudonyms ----------------

class Pseudonyms:
    """Codes for documents (``D0001``) and annotators (``A01``), kept in a local CSV file (kind, name, code) so
    that they stay the same across runs. New names get the next free numbers, in sorted order."""
    PREFIX = {"document": "D", "annotator": "A"}
    WIDTH = {"document": 4, "annotator": 2}

    def __init__(self, path: Path):
        self.path = Path(path)
        self.codes: Dict[str, Dict[str, str]] = {kind: {} for kind in self.PREFIX}
        if self.path.exists():
            for row in pd.read_csv(self.path, dtype=str, keep_default_na=False).itertuples(index=False):
                self.codes[row.kind][row.name] = row.code

    def assign(self, kind: str, names: Iterable[str]) -> Dict[str, str]:
        codes = self.codes[kind]
        last = max((int(c[1:]) for c in codes.values()), default=0)
        for name in sorted(set(map(str, names)) - set(codes)):
            last += 1
            codes[name] = f"{self.PREFIX[kind]}{last:0{self.WIDTH[kind]}d}"
        return codes

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [(kind, name, code) for kind, codes in self.codes.items() for name, code in sorted(codes.items())]
        pd.DataFrame(rows, columns=["kind", "name", "code"]).to_csv(self.path, index=False)


def pseudonymise(counts: pd.DataFrame, docs: Dict[str, str], users: Dict[str, str]) -> pd.DataFrame:
    """``pair_counts`` with codes (PAIR_COLUMNS); each pair is ordered by code, with its per-side counts."""
    a, b = counts["user_a"].map(users), counts["user_b"].map(users)
    swap = (a > b).to_numpy(bool)
    out = counts.assign(doc=counts["doc"].map(docs), annotator_a=np.where(swap, b, a),
                        annotator_b=np.where(swap, a, b))
    for x, y in (("n_a", "n_b"), ("e2e_a", "e2e_b")):
        out[x], out[y] = np.where(swap, counts[y], counts[x]), np.where(swap, counts[x], counts[y])
    return out[PAIR_COLUMNS].sort_values(["doc", "annotator_a", "annotator_b"]).reset_index(drop=True)


# ---------------- site report ----------------

def tool_version() -> str:
    """Package version, plus the git commit if run from a checkout (``+modified`` if the code differs from it)."""
    try:
        text = version("gemtex-iaa-semantic")
    except PackageNotFoundError:
        text = "unknown"
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True,
                                timeout=10).stdout.strip()
        if commit:
            changed = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here, capture_output=True,
                                     text=True, timeout=10).stdout.strip()
            text += f" ({commit}{'+modified' if changed else ''})"
    except (OSError, subprocess.SubprocessError):
        pass
    return text


def data_counts(data: L.ProjectData, selection: Optional[SEL.Selection], counts: pd.DataFrame) -> Dict[str, int]:
    """Text-free counts of what was read (``data``: as loaded), excluded and analysed (``counts``: pair_counts)."""
    meta = data.meta
    status = meta.get("document_status", {})
    ex = selection.excluded if selection is not None else pd.DataFrame(columns=["reason"])
    too_few = len(selection.too_few) if selection is not None else 0
    low = selection.low_change if selection is not None else pd.DataFrame(columns=["excluded_as"])
    return {
        "exports": int(meta.get("exports", 1)),
        "documents_in_exports": int(meta.get("documents_in_exports", 0)),
        "documents_min_annotators_finished": int(status.get(L.ANALYSABLE, 0)),
        "documents_pending": int(status.get(L.PENDING, 0)),
        "documents_too_few_assigned": int(status.get(L.TOO_FEW, 0)),
        "documents_dropped_too_few_annotators": int(meta.get("documents_dropped_too_few", 0)) + too_few,
        "excluded_untouched": int((ex["reason"] == SEL.UNTOUCHED).sum()),
        "excluded_copy": int((ex["reason"] == SEL.COPY).sum()),
        "excluded_low_change": int((ex["reason"] == SEL.LOW_CHANGE).sum()),
        "low_change_analysed": int((low["excluded_as"] == "").sum()),
        "documents_analysed": int(len(counts["doc"].unique())) if len(counts) else 0,
        "annotators_analysed": int(len(set(counts["user_a"]) | set(counts["user_b"]))) if len(counts) else 0,
        "annotator_pairs": int(len(counts[["user_a", "user_b"]].drop_duplicates())) if len(counts) else 0,
        "document_annotator_pairs": int(len(counts)),
    }


DATA_LABELS = {
    "exports": "INCEpTION exports",
    "documents_in_exports": "documents in the exports",
    "documents_min_annotators_finished": "documents finished by at least MIN_ANNOTATORS annotators (read)",
    "documents_pending": "documents assigned to at least MIN_ANNOTATORS, finished by fewer (not read)",
    "documents_too_few_assigned": "documents assigned to fewer than MIN_ANNOTATORS annotators (not read)",
    "documents_dropped_too_few_annotators": "documents dropped: fewer than MIN_ANNOTATORS annotators left",
    "excluded_untouched": "annotation documents excluded: untouched pre-annotation",
    "excluded_copy": "annotation documents excluded: copy of another annotation document",
    "excluded_low_change": "annotation documents excluded: low change",
    "low_change_analysed": "low-change annotation documents (analysed)",
    "documents_analysed": "documents analysed",
    "annotators_analysed": "annotators analysed",
    "annotator_pairs": "annotator pairs",
    "document_annotator_pairs": "document x annotator pairs",
}


CHECKS = ["untouched", "copies", "low_change_excluded", "low_change_analysed"]


def annotator_checks(selection: Optional[SEL.Selection]) -> pd.DataFrame:
    """Per user (real names) with an excluded or low-change annotation document: how many, per reason."""
    if selection is None:
        return pd.DataFrame(columns=["user", *CHECKS])
    ex, low = selection.excluded, selection.low_change
    rows = [{"user": u,
             "untouched": int(((ex["user"] == u) & (ex["reason"] == SEL.UNTOUCHED)).sum()),
             "copies": int(((ex["user"] == u) & (ex["reason"] == SEL.COPY)).sum()),
             "low_change_excluded": int(((ex["user"] == u) & (ex["reason"] == SEL.LOW_CHANGE)).sum()),
             "low_change_analysed": int(((low["user"] == u) & (low["excluded_as"] == "")).sum())}
            for u in sorted(set(ex["user"]) | set(low["user"]))]
    return pd.DataFrame(rows, columns=["user", *CHECKS])


def _settings(cfg: Config, release: str, preannotation: bool) -> Dict[str, object]:
    return {"STATES": list(cfg.STATES), "MIN_ANNOTATORS": cfg.MIN_ANNOTATORS,
            "SKIP_ANNOTATORS (number)": len(cfg.SKIP_ANNOTATORS), "EXCLUDE_UNTOUCHED": cfg.EXCLUDE_UNTOUCHED,
            "EXCLUDE_COPIES": cfg.EXCLUDE_COPIES, "EXCLUDE_LOW_CHANGE": cfg.EXCLUDE_LOW_CHANGE,
            "PREANNO_LOW_CHANGE": cfg.PREANNO_LOW_CHANGE, "BOOTSTRAP": cfg.BOOTSTRAP, "SEED": cfg.SEED,
            "SNOMED CT release": release, "pre-annotation": preannotation}


def site_report(cfg: Config, out_dir: Path, use_cache: bool = True) -> Path:
    """Compute the site's figures and write ``out_dir/share`` and ``out_dir/local``; returns the shared report."""
    share, local = Path(out_dir) / "share", Path(out_dir) / "local"
    share.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    data = L.get_project(cfg, use_cache=use_cache)
    store = SnomedStore(cfg.HDF5)
    ctx = selection = None
    counts = pd.DataFrame(columns=["doc", "user_a", "user_b", *COUNTS])
    if len(data.docs):
        ctx = P.prepare(cfg, store=store, data=data, use_cache=use_cache)
        selection, counts = ctx.selection, pair_counts(ctx)
        write_exclusion_report(local, SEL.RULES, selection.settings, P.selection_tables(ctx),
                               selection.removed_annotators)
    else:
        log.warning("no document has at least %d annotators in %s", cfg.MIN_ANNOTATORS, ", ".join(cfg.STATES))
    preannotation = bool(ctx is not None and len(ctx.preannotation))

    pseudonyms = Pseudonyms(Path(cfg.DIR_OUTPUT) / "pseudonyms.csv")
    checks = annotator_checks(selection)
    docs = pseudonyms.assign("document", counts["doc"])
    users = pseudonyms.assign("annotator", set(counts["user_a"]) | set(counts["user_b"]) | set(checks["user"]))
    pseudonyms.save()

    coded = pseudonymise(counts, docs, users)
    overall = figures(coded, cfg.BOOTSTRAP, cfg.SEED)
    checks = checks.assign(annotator=checks["user"].map(users).astype(object)).drop(columns="user")
    annotators = by_annotator(coded).astype({"annotator": object}).merge(checks, on="annotator", how="outer")
    counted = ["documents", "pairs", *CHECKS]                  # 0 for annotators whose documents were all excluded
    annotators[counted] = annotators[counted].fillna(0).astype(int)
    annotators = annotators.sort_values("annotator").reset_index(drop=True)
    pairs = by_pair(coded)
    summary = {"site": cfg.SITE, "tool": tool_version(), "generated": f"{datetime.now():%Y-%m-%d %H:%M}",
               "settings": _settings(cfg, store.release_date, preannotation),
               "data": data_counts(data, selection, counts),
               "figures": json.loads(overall.to_json(orient="records"))}

    (share / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    overall.to_csv(share / "figures.csv", index=False)
    annotators.to_csv(share / "annotators.csv", index=False)
    pairs.to_csv(share / "annotator_pairs.csv", index=False)
    coded.to_csv(share / "pair_counts.csv", index=False)
    report = share / "report.md"
    report.write_text(_site_markdown(summary, overall, annotators, pairs), encoding="utf-8")

    # the same tables with the real names, for the site only
    ui_names = data.annotators.set_index("user")["ui_name"]
    names = pd.DataFrame({"user": list(users)}, index=pd.Index(list(users.values()), name="annotator"))
    names["ui_name"] = names["user"].map(ui_names)
    annotators.join(names, on="annotator").to_csv(local / "annotators.csv", index=False)
    for side in ("a", "b"):
        pairs = pairs.join(names.add_suffix(f"_{side}"), on=f"annotator_{side}")
    pairs.to_csv(local / "annotator_pairs.csv", index=False)
    coded.assign(document=coded["doc"].map({code: name for name, code in docs.items()})).to_csv(
        local / "pair_counts.csv", index=False)
    (local / "README.txt").write_text(
        "This folder contains real document and annotator names. Keep it at your site and do not send it.\n"
        "Send only the folder ../share. The codes (D..., A...) are listed in "
        f"{Path(cfg.DIR_OUTPUT) / 'pseudonyms.csv'}.\n", encoding="utf-8")
    log.info("site report written to %s", report)
    return report


def _site_markdown(summary: Dict, overall: pd.DataFrame, annotators: pd.DataFrame, pairs: pd.DataFrame) -> str:
    data = pd.DataFrame([(DATA_LABELS.get(k, k), v) for k, v in summary["data"].items()], columns=["item", "count"])
    lines = [f"# Agreement report: {summary['site']}\n",
             f"_generated {summary['generated']} with gemtex-iaa-semantic {summary['tool']}_\n",
             "Text-free and pseudonymised: documents (D…) and annotators (A…) appear only as codes, which only the "
             "site can resolve.\n",
             "## Data\n", markdown_table(data, max_rows=len(data)),
             "## Agreement\n", markdown_table(overall),
             *[f"* {d}" for d in DEFINITIONS], "",
             "## Annotators\n",
             "Figures over all documents and partners of the annotator. Annotation documents `untouched` "
             "(pre-annotation left as it was), `copies` and `low_change_excluded` were excluded; "
             "`low_change_analysed` changed less than PREANNO_LOW_CHANGE of the pre-annotation but were analysed.\n",
             markdown_table(annotators, max_rows=max(1, len(annotators))),
             "## Annotator pairs\n", markdown_table(pairs, max_rows=max(1, len(pairs))),
             "## Settings\n", *[f"* {k}: {v}" for k, v in summary["settings"].items()], ""]
    return "\n".join(lines) + "\n"


# ---------------- pooling ----------------

def read_share(directory: Path) -> Tuple[Dict, pd.DataFrame]:
    """``summary.json`` and ``pair_counts.csv`` of a site's ``share`` folder (or of its run directory)."""
    directory = Path(directory)
    if (directory / "share").is_dir():
        directory = directory / "share"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    counts = pd.read_csv(directory / "pair_counts.csv", dtype={"doc": str, "annotator_a": str, "annotator_b": str})
    return summary, counts


def pool(directories: Sequence[Path], n_boot: int = 1000, seed: int = 13) -> Dict[str, object]:
    """Combine the share folders of several sites. Documents and annotators are site-specific (``<site>:<code>``);
    the figures are pooled over all documents of all sites, with a bootstrap CI over documents. Counts missing from
    a share (made with an older version) leave the site out of the figures that need them."""
    summaries, frames, missing = [], [], []
    for directory in directories:
        summary, counts = read_share(directory)
        site = summary["site"]
        if any(s["site"] == site for s in summaries):
            raise ValueError(f"site {site!r} given twice ({directory})")
        summaries.append(summary)
        if set(COUNTS) - set(counts.columns):
            missing.append(f"{site} ({', '.join(c for c in COUNTS if c not in counts.columns)})")
            counts = counts.reindex(columns=PAIR_COLUMNS)
        frames.append(counts.assign(site=site, doc=site + ":" + counts["doc"],
                                    annotator_a=site + ":" + counts["annotator_a"],
                                    annotator_b=site + ":" + counts["annotator_b"]))
    counts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["site", *PAIR_COLUMNS])
    rows = [figures(counts, n_boot, seed).assign(site="ALL")]
    rows += [figures(g, n_boot, seed).assign(site=site) for site, g in counts.groupby("site", sort=True)]
    pooled = pd.concat(rows, ignore_index=True)
    pooled = pooled[["site", *[c for c in pooled.columns if c != "site"]]]
    annotators = by_annotator(counts)
    annotators.insert(0, "site", annotators["annotator"].str.split(":").str[0])
    sites = pd.DataFrame([{"site": s["site"], "tool": s["tool"], "generated": s["generated"], **s["data"]}
                          for s in summaries]).sort_values("site").reset_index(drop=True)
    warnings = []
    for key in ("tool",):
        values = {s[key] for s in summaries}
        if len(values) > 1:
            warnings.append(f"different {key}: {', '.join(sorted(values))}")
    if missing:
        warnings.append(f"counts missing (older version; the site is left out of those figures): "
                        f"{'; '.join(missing)}")
    for key in sorted({k for s in summaries for k in s["settings"]} - {"BOOTSTRAP", "SEED", "pre-annotation"}):
        values = {json.dumps(s["settings"].get(key)) for s in summaries}
        if len(values) > 1:
            warnings.append(f"different setting {key}: "
                            + ", ".join(f"{s['site']}={json.dumps(s['settings'].get(key))}" for s in summaries))
    without = [s["site"] for s in summaries if not s["settings"].get("pre-annotation")]
    if without:
        warnings.append(f"no pre-annotation (not in the inherited share): {', '.join(without)}")
    return {"figures": pooled, "annotators": annotators, "sites": sites, "pair_counts": counts,
            "warnings": warnings}


def _cell(value: float, low: float, high: float) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.3f} [{low:.3f}, {high:.3f}]" if pd.notna(low) else f"{value:.3f}"


def write_pool(result: Dict[str, object], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled, sites = result["figures"], result["sites"]
    pooled.to_csv(out_dir / "pooled_figures.csv", index=False)
    result["annotators"].to_csv(out_dir / "pooled_annotators.csv", index=False)
    sites.to_csv(out_dir / "sites.csv", index=False)
    result["pair_counts"].to_csv(out_dir / "pooled_pair_counts.csv", index=False)
    wide = pooled.assign(cell=[_cell(v, lo, hi) for v, lo, hi in zip(pooled["value"], pooled["ci_low"],
                                                                     pooled["ci_high"])])
    wide = wide.pivot(index="figure", columns="site", values="cell").reindex([f.name for f in FIGURES])
    wide = wide[["ALL", *[c for c in wide.columns if c != "ALL"]]]
    data = sites.drop(columns=["tool", "generated"]).set_index("site").T
    data.insert(0, "ALL", data.sum(axis=1))
    data.index = [DATA_LABELS.get(i, i) for i in data.index]
    lines = [f"# Pooled agreement report ({len(sites)} site{'s' if len(sites) != 1 else ''})\n",
             f"_generated {datetime.now():%Y-%m-%d %H:%M}_\n",
             "## Warnings\n", *([f"* {w}" for w in result["warnings"]] or ["none"]), "",
             "## Agreement (value [95 % CI])\n", markdown_table(wide), *[f"* {d}" for d in DEFINITIONS], "",
             "## Data\n", markdown_table(data.rename_axis("item"), max_rows=len(data)),
             "## Sites\n", markdown_table(sites[["site", "tool", "generated"]], max_rows=max(1, len(sites)))]
    path = out_dir / "pooled_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
