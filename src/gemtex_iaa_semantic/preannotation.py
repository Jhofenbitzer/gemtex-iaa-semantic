"""Pre-annotation (INITIAL_CAS): what the annotators did with it.

Every annotator started from the same automatic pre-annotation, so agreement on concepts that both annotators left
as they were is inherited rather than independent. Mentions are compared by span and concept key (concept +
normalised literal). Only documents with a pre-annotated concept are compared.

* Decision of an annotator on a pre-annotated mention: ``kept`` if the annotator has a mention with the same span
  and key (matched one to one), else ``relabelled`` if an unmatched mention has the same span, ``reshaped`` if one
  overlaps it, and ``deleted`` otherwise. Unmatched mentions of the annotator are ``changed`` if they overlap a
  pre-annotated mention and ``added`` otherwise.
* Per pairwise region, :func:`region_pre_side` says whose mentions equal the pre-annotated mentions overlapping it:
  ``both`` (the region is *inherited*), ``a``, ``b`` or ``neither``, or ``""`` without a pre-annotated mention.

The export has no editing history, so a kept mention may have been checked or never looked at.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from gemtex_iaa_semantic.align import Region

KEPT, RELABELLED, RESHAPED, DELETED = "kept", "relabelled", "reshaped", "deleted"
DECISIONS = [KEPT, RELABELLED, RESHAPED, DELETED]
CHANGED, ADDED = "changed", "added"
COUNT_COLUMNS = ["n_pre", "n_annotator", *DECISIONS, CHANGED, ADDED, "kept_span"]


def match(pre: pd.DataFrame, ann: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Decisions on the pre-annotated mentions ``pre`` and classes (kept, changed, added) of the annotator's
    mentions ``ann``, both of one document (columns begin, end, key)."""
    pb, pe = pre["begin"].to_numpy(np.int64), pre["end"].to_numpy(np.int64)
    ab, ae = ann["begin"].to_numpy(np.int64), ann["end"].to_numpy(np.int64)
    free: Dict[tuple, List[int]] = {}
    for j, triple in enumerate(zip(ab.tolist(), ae.tolist(), ann["key"].tolist())):
        free.setdefault(triple, []).append(j)
    kept = np.zeros(len(pre), dtype=bool)
    matched = np.zeros(len(ann), dtype=bool)
    for i, triple in enumerate(zip(pb.tolist(), pe.tolist(), pre["key"].tolist())):
        if free.get(triple):
            kept[i] = matched[free[triple].pop(0)] = True
    overlap = (pb[:, None] < ae[None, :]) & (ab[None, :] < pe[:, None])
    same_span = (pb[:, None] == ab[None, :]) & (pe[:, None] == ae[None, :])
    unmatched = ~matched[None, :]
    decisions = np.select([kept, (same_span & unmatched).any(axis=1), (overlap & unmatched).any(axis=1)],
                          [KEPT, RELABELLED, RESHAPED], DELETED)
    classes = np.select([matched, overlap.any(axis=0)], [KEPT, CHANGED], ADDED)
    return decisions, classes


def compare(pre: pd.DataFrame, mentions: pd.DataFrame, doc_users: Mapping[str, Sequence[str]],
            low_change: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every annotation document of a pre-annotated document with its pre-annotation.

    Returns the decisions (doc, user, mention_id of the pre-annotated mention, decision) and the annotation
    documents (doc, user, COUNT_COLUMNS, f1 and f1_span against the pre-annotation, ``untouched`` if identical to
    it, ``low_change`` if f1 >= 1 - ``low_change``). ``doc_users`` lists the annotators of each document, including
    those with an empty document (who deleted everything).
    """
    pre_by_doc = {doc: g for doc, g in pre.groupby("doc")}
    ann_by = {key: g for key, g in mentions.groupby(["doc", "user"])}
    no_mentions = mentions.iloc[0:0]
    decisions, documents = [], []
    for doc in sorted(set(doc_users) & set(pre_by_doc)):
        p = pre_by_doc[doc]
        pre_spans = Counter(zip(p["begin"], p["end"]))
        for user in sorted(doc_users[doc]):
            a = ann_by.get((doc, user), no_mentions)
            dec, cls = match(p, a)
            decisions.append(pd.DataFrame({"doc": doc, "user": user, "mention_id": p["mention_id"].to_numpy(),
                                           "decision": dec}))
            n_dec, n_cls = Counter(dec.tolist()), Counter(cls.tolist())
            documents.append({"doc": doc, "user": user, "n_pre": len(p), "n_annotator": len(a),
                              **{d: n_dec[d] for d in DECISIONS}, CHANGED: n_cls[CHANGED], ADDED: n_cls[ADDED],
                              "kept_span": sum((pre_spans & Counter(zip(a["begin"], a["end"]))).values())})
    dec = (pd.concat(decisions, ignore_index=True) if decisions
           else pd.DataFrame(columns=["doc", "user", "mention_id", "decision"]))
    docs = pd.DataFrame(documents, columns=["doc", "user", *COUNT_COLUMNS]).astype({c: "int64" for c in COUNT_COLUMNS})
    den = (docs["n_pre"] + docs["n_annotator"]).replace(0, np.nan).astype(float)
    docs["f1"] = 2 * docs[KEPT] / den
    docs["f1_span"] = 2 * docs["kept_span"] / den
    docs["untouched"] = ((docs[KEPT] == docs["n_pre"]) & (docs[KEPT] == docs["n_annotator"])).astype(bool)
    docs["low_change"] = (docs["f1"] >= 1 - low_change).astype(bool)
    return dec, docs


def region_pre_side(regions: Iterable[Region], pre: pd.DataFrame) -> List[str]:
    """Per region, in the order of ``regions``: whose mentions equal the pre-annotated mentions overlapping it
    (``both``, ``a``, ``b`` or ``neither``), or ``""`` if no pre-annotated mention overlaps it."""
    by_doc = {doc: (g["begin"].to_numpy(np.int64), g["end"].to_numpy(np.int64), g["key"].to_numpy(object))
              for doc, g in pre.groupby("doc")}
    out = []
    for r in regions:
        pb, pe, pk = by_doc.get(r.doc, (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, object)))
        inside = np.flatnonzero((pb < r.end) & (pe > r.begin))
        if not len(inside):
            out.append("")
            continue
        expected = set(zip(pb[inside].tolist(), pe[inside].tolist(), pk[inside].tolist()))
        a_pre = set(zip(r.a["begin"].tolist(), r.a["end"].tolist(), r.a["key"].tolist())) == expected
        b_pre = set(zip(r.b["begin"].tolist(), r.b["end"].tolist(), r.b["key"].tolist())) == expected
        out.append("both" if a_pre and b_pre else "a" if a_pre else "b" if b_pre else "neither")
    return out
