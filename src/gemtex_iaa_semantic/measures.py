"""Agreement measures as sufficient statistics (counts per document and annotator pair), and the document
bootstrap of their pooled ratios."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from gemtex_iaa_semantic.align import BOUNDARY, EXACT, Region, match_one_to_one

# ---------------- spans ----------------


def span_counts(region: Region) -> Dict[str, int]:
    """Sufficient statistics of strict, lenient and end-to-end F1 for one region (stacked spans count once in the
    span F1s)."""
    sa, sb = region.spans_a, region.spans_b
    strict = len(set(sa) & set(sb))
    lenient = len(match_one_to_one(sa, sb))
    keys_a = Counter(zip(region.a["begin"], region.a["end"], region.a["key"]))
    keys_b = Counter(zip(region.b["begin"], region.b["end"], region.b["key"]))
    return {"n_a": len(sa), "n_b": len(sb), "strict": strict, "lenient": lenient,
            "e2e_a": sum(keys_a.values()), "e2e_b": sum(keys_b.values()),
            "e2e": sum((keys_a & keys_b).values())}


def is_exact(region: Region) -> bool:
    """EXACT: both annotators have mentions in the region, none unresolved, with the same spans and concept keys."""
    a, b = region.a, region.b
    if a.empty or b.empty or a["unresolved"].any() or b["unresolved"].any():
        return False
    return set(zip(a["begin"], a["end"], a["key"])) == set(zip(b["begin"], b["end"], b["key"]))


# ---------------- concepts ----------------

def one_to_one_pairs(regions: Iterable[Region]) -> pd.DataFrame:
    """Regions where each annotator has exactly one mention (exact or boundary alignment)."""
    rows: List[Dict] = []
    for r in regions:
        if r.kind not in (EXACT, BOUNDARY) or len(r.a) != 1 or len(r.b) != 1:
            continue
        ma, mb = r.a.iloc[0], r.b.iloc[0]
        rows.append({"doc": r.doc, "user_a": r.user_a, "user_b": r.user_b, "region_kind": r.kind,
                     "mention_a": int(ma.mention_id), "mention_b": int(mb.mention_id),
                     "key_a": ma.key, "key_b": mb.key, "concept_a": int(ma.concept), "concept_b": int(mb.concept),
                     "status_a": ma.status, "status_b": mb.status, "same_key": ma.key == mb.key})
    return pd.DataFrame(rows, columns=["doc", "user_a", "user_b", "region_kind", "mention_a", "mention_b", "key_a",
                                       "key_b", "concept_a", "concept_b", "status_a", "status_b", "same_key"])


def kappa_terms(pairs: pd.DataFrame) -> pd.DataFrame:
    """Pooled Cohen's κ (De Vries et al. 2008) over 1:1 pairs, as sufficient statistics per (doc, user_a, user_b).

    With n items, ``agree`` matches and chance agreement n·p_e from the pair's marginals in that document,
    ``num`` = agree − n·p_e and ``den`` = n − n·p_e, so sum(num) / sum(den) is the pooled κ and one row alone is the
    pair's Cohen's κ in that document.
    """
    rows = []
    for (doc, ua, ub), g in pairs.groupby(["doc", "user_a", "user_b"], sort=True):
        n = len(g)
        ca, cb = g["key_a"].value_counts(), g["key_b"].value_counts()
        expected = float((ca * cb.reindex(ca.index, fill_value=0)).sum()) / n
        agree = int(g["same_key"].sum())
        rows.append({"doc": doc, "user_a": ua, "user_b": ub, "n": n, "agree": agree, "expected": expected,
                     "num": agree - expected, "den": n - expected})
    return pd.DataFrame(rows, columns=["doc", "user_a", "user_b", "n", "agree", "expected", "num", "den"])


def alpha_units(clustered: pd.DataFrame) -> pd.DataFrame:
    """Clusters where every annotator present has exactly one mention; one row per (cluster, annotator)."""
    counts = clustered.groupby(["cluster", "user"]).size()
    multi = set(counts[counts > 1].index.get_level_values(0))
    single = clustered[~clustered["cluster"].isin(multi)]
    return single[["cluster", "doc", "user", "key", "concept", "status"]].reset_index(drop=True)


def alpha_terms(unit_rows: pd.DataFrame, value: str = "key") -> pd.DataFrame:
    """Nominal Krippendorff's α over ``alpha_units`` with chance agreement per document, as sufficient statistics
    per (doc, user_a, user_b) like ``kappa_terms``.

    In a unit with m values, every pair of annotators adds 2 / (m − 1) to ``n`` (a unit adds m, its pairable values)
    and, if they agree, to ``agree`` (the diagonal of the coincidence matrix). ``expected`` is the document's chance
    agreement (sum n_c² − n) / (n − 1), shared among its pairs in proportion to their ``n``. Summed over a document,
    (agree − expected) / (n − expected) is its α; pooled over documents, sum(num) / sum(den).
    """
    rows = []
    for doc, d in unit_rows.groupby("doc", sort=True):
        pairs: Dict = {}
        pairable = []
        for _, u in d.groupby("cluster", sort=True):
            if len(u) < 2:
                continue
            pairable.append(u[value])
            weight = 2.0 / (len(u) - 1)
            for (ua, va), (ub, vb) in combinations(sorted(zip(u["user"], u[value])), 2):
                n, agree = pairs.get((ua, ub), (0.0, 0.0))
                pairs[(ua, ub)] = (n + weight, agree + weight * (va == vb))
        if not pairable:
            continue
        values = pd.concat(pairable).value_counts()
        n_doc = int(values.sum())
        chance = float((values ** 2).sum() - n_doc) / (n_doc - 1)
        for (ua, ub), (n, agree) in sorted(pairs.items()):
            expected = n * chance / n_doc
            rows.append({"doc": doc, "user_a": ua, "user_b": ub, "n": n, "agree": agree, "expected": expected,
                         "num": agree - expected, "den": n - expected})
    return pd.DataFrame(rows, columns=["doc", "user_a", "user_b", "n", "agree", "expected", "num", "den"])


# ---------------- pooling ----------------

def ratio(df: pd.DataFrame, num: str, den: str, num_factor: float = 1.0) -> float:
    d = df[den].sum()
    return float(num_factor * df[num].sum() / d) if d else float("nan")


def bootstrap_ratio(df: pd.DataFrame, num: str, den: str, *, num_factor: float = 1.0, group: str = "doc",
                    n_boot: int = 1000, seed: int = 13, level: float = 0.95) -> Tuple[float, float, float]:
    """Point estimate and percentile CI of sum(num)/sum(den), resampling documents with replacement."""
    point = ratio(df, num, den, num_factor)
    if df.empty or n_boot <= 0:
        return point, float("nan"), float("nan")
    per_group = df.groupby(group)[[num, den]].sum()
    nums, dens = per_group[num].to_numpy(float), per_group[den].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_group), size=(n_boot, len(per_group)))
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = num_factor * nums[draws].sum(axis=1) / dens[draws].sum(axis=1)
    stats = stats[np.isfinite(stats)]
    if not len(stats):
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return point, float(lo), float(hi)
