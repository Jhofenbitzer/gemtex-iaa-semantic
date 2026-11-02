"""Alignment of mentions: pairwise regions (per annotator pair) and multi-annotator clusters.

A *region* is a maximal stretch of text covered by transitively overlapping mentions of the two annotators of a
pair, so that annotations with different segmentations (precoordinated span vs. split spans) end up in the same
unit. A *cluster* is the same over the mentions of all annotators of a document.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

EXACT, BOUNDARY, ONE_TO_MANY, MANY_TO_MANY, OMISSION = "exact", "boundary", "one_to_many", "many_to_many", "omission"


def merge_intervals(spans: Sequence[Tuple[int, int, object]]) -> List[Tuple[int, int, List[object]]]:
    """Group items with overlapping [begin, end) spans (touching spans do not overlap)."""
    groups: List[Tuple[int, int, List[object]]] = []
    for begin, end, item in sorted(spans, key=lambda s: (s[0], s[1])):
        if groups and begin < groups[-1][1]:
            b, e, items = groups[-1]
            groups[-1] = (b, max(e, end), items + [item])
        else:
            groups.append((begin, end, [item]))
    return groups


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def jaccard(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = overlap(a, b)
    return inter / (max(a[1], b[1]) - min(a[0], b[0])) if inter else 0.0


def match_one_to_one(a_spans: Sequence[Tuple[int, int]], b_spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Maximum-weight 1:1 matching of overlapping spans (weight = character Jaccard)."""
    if not a_spans or not b_spans:
        return []
    weights = np.array([[jaccard(a, b) for b in b_spans] for a in a_spans])
    rows, cols = linear_sum_assignment(-weights)
    return [(r, c) for r, c in zip(rows, cols) if weights[r, c] > 0]


@dataclass
class Region:
    doc: str
    user_a: str
    user_b: str
    begin: int
    end: int
    a: pd.DataFrame = field(repr=False)      # mentions of annotator a in the region
    b: pd.DataFrame = field(repr=False)

    @property
    def spans_a(self) -> List[Tuple[int, int]]:
        return sorted(set(zip(self.a["begin"], self.a["end"])))

    @property
    def spans_b(self) -> List[Tuple[int, int]]:
        return sorted(set(zip(self.b["begin"], self.b["end"])))

    @property
    def kind(self) -> str:
        sa, sb = self.spans_a, self.spans_b
        if not sa or not sb:
            return OMISSION
        if sa == sb:
            return EXACT
        if len(sa) == 1 and len(sb) == 1:
            return BOUNDARY
        if len(sa) == 1 or len(sb) == 1:
            return ONE_TO_MANY
        return MANY_TO_MANY


def pair_regions(doc: str, a: pd.DataFrame, b: pd.DataFrame, user_a: str, user_b: str) -> List[Region]:
    items = [(r.begin, r.end, ("a", i)) for i, r in enumerate(a.itertuples(index=False))]
    items += [(r.begin, r.end, ("b", i)) for i, r in enumerate(b.itertuples(index=False))]
    regions = []
    for begin, end, members in merge_intervals(items):
        ia = [i for side, i in members if side == "a"]
        ib = [i for side, i in members if side == "b"]
        regions.append(Region(doc, user_a, user_b, begin, end, a.iloc[ia], b.iloc[ib]))
    return regions


def annotator_pairs(mentions: pd.DataFrame, docs: pd.DataFrame = None) -> Iterator[Tuple[str, str, str]]:
    """All (doc, user_a, user_b) with user_a < user_b among the annotators of each document.

    Annotators of a document are those with loaded annotations; if ``docs`` has a ``users`` column listing
    annotators without any mention (empty but finished documents), they are included too.
    """
    users_per_doc: Dict[str, set] = mentions.groupby("doc")["user"].apply(set).to_dict()
    if docs is not None and "users" in docs:
        for doc, users in zip(docs["doc"], docs["users"]):
            users_per_doc.setdefault(doc, set()).update(users)
    for doc in sorted(users_per_doc):
        for ua, ub in combinations(sorted(users_per_doc[doc]), 2):
            yield doc, ua, ub


def all_pair_regions(mentions: pd.DataFrame, docs: pd.DataFrame = None) -> Iterator[Region]:
    by_doc_user = {k: g for k, g in mentions.groupby(["doc", "user"])}
    empty = mentions.iloc[0:0]
    for doc, ua, ub in annotator_pairs(mentions, docs):
        yield from pair_regions(doc, by_doc_user.get((doc, ua), empty), by_doc_user.get((doc, ub), empty), ua, ub)


def clusters(mentions: pd.DataFrame) -> pd.DataFrame:
    """Multi-annotator clusters: overlap components over all annotators' mentions of a document.

    Returns the mentions with an additional ``cluster`` column (unique over the corpus).
    """
    out = []
    next_id = 0
    for _, group in mentions.groupby("doc", sort=True):
        items = [(r.begin, r.end, r.mention_id) for r in group.itertuples(index=False)]
        for _, _, members in merge_intervals(items):
            out.extend((m, next_id) for m in members)
            next_id += 1
    ids = pd.DataFrame(out, columns=["mention_id", "cluster"])
    return mentions.merge(ids, on="mention_id", how="left")
