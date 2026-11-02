"""Which annotation documents enter the analysis.

Annotation documents are excluded before any measure is computed, and every exclusion is listed in the run's
``exclusion_report.md``. Documents are compared by span and concept key, as everywhere else; in that order:

* ``untouched``: identical to the non-empty automatic pre-annotation of its document (nothing was annotated);
* ``low_change``: F1 against the pre-annotation >= 1 - ``low_change`` (only if ``exclude_low_change``; such
  documents are always listed);
* ``copy``: identical to another non-empty annotation document of the same document. Of each group of identical
  documents one is kept: the one of the account with the smallest share of copied documents (among its documents
  left after the steps above; ties: by display name). The creation times in the export are import times and cannot
  tell the original. Relations are not part of the comparison; whether they are identical too is reported.

Documents left with fewer than ``MIN_ANNOTATORS`` annotators are dropped (:func:`require_annotators`), and
annotators without any document left drop out of the analysis.
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Sequence, Tuple

import pandas as pd

from gemtex_iaa_semantic import preannotation as PA
from gemtex_iaa_semantic.io.loader import ProjectData, without_documents

UNTOUCHED, LOW_CHANGE, COPY = "untouched", "low_change", "copy"
REASONS = [UNTOUCHED, LOW_CHANGE, COPY]
EXCLUDED_COLUMNS = ["doc", "user", "ui_name", "site", "reason", "copy_of", "n_mentions", "n_relations",
                    "f1_preannotation", "relations_identical"]
LOW_CHANGE_COLUMNS = ["doc", "user", "ui_name", "site", "n_pre", "n_annotator", "f1_preannotation", "excluded_as"]
TOO_FEW_COLUMNS = ["doc", "annotators", "mentions"]

Pair = Tuple[str, str]       # (doc, user)

RULES = [
    "Annotation documents are compared by the span and concept key (concept + normalised literal) of their "
    "concepts; relations and No_Human spans are not compared.",
    "`untouched`: identical to the non-empty automatic pre-annotation (INITIAL_CAS) of its document "
    "(EXCLUDE_UNTOUCHED).",
    "`low_change`: F1 against the pre-annotation >= 1 - PREANNO_LOW_CHANGE; always listed, excluded only with "
    "EXCLUDE_LOW_CHANGE.",
    "`copy`: identical to another non-empty annotation document of the same document (EXCLUDE_COPIES), among the "
    "documents left after the rules above. Of each group of identical documents the one of the account with the "
    "smallest share of copied documents is kept (ties: by display name); `copy_of` names it. The creation times in "
    "the export are import times and cannot tell the original. `relations_identical` says whether the relations "
    "are identical too.",
    "Documents with fewer than MIN_ANNOTATORS annotators (in STATES, over all exports) are not read. Documents "
    "left with fewer annotators after the rules above are dropped (`too_few`), with their remaining annotation "
    "documents.",
    "Annotators without any document left are not analysed.",
]


@dataclass
class Selection:
    excluded: pd.DataFrame                  # EXCLUDED_COLUMNS, one row per excluded annotation document
    low_change: pd.DataFrame                # LOW_CHANGE_COLUMNS, every low-change document (excluded_as: "" = kept)
    loaded: Dict[str, int] = field(default_factory=dict)     # annotation documents, mentions, annotators loaded
    removed_annotators: List[str] = field(default_factory=list)
    settings: Dict[str, object] = field(default_factory=dict)
    # TOO_FEW_COLUMNS, documents left with fewer than MIN_ANNOTATORS annotators (require_annotators)
    too_few: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TOO_FEW_COLUMNS))

    @property
    def pairs(self) -> FrozenSet[Pair]:
        return frozenset(zip(self.excluded["doc"], self.excluded["user"]))

    def describe(self) -> str:
        """E.g. ``7 untouched, 38 copy`` (``none`` if nothing is excluded)."""
        counts = self.excluded["reason"].value_counts()
        return ", ".join(f"{counts[r]} {r}" for r in REASONS if r in counts) or "none"


def _signatures(mentions: pd.DataFrame) -> Dict[Pair, tuple]:
    return {key: tuple(sorted(zip(g["begin"].tolist(), g["end"].tolist(), g["key"].tolist())))
            for key, g in mentions.groupby(["doc", "user"])}


def _relation_signatures(relations: pd.DataFrame, mentions: pd.DataFrame) -> Dict[Pair, tuple]:
    """Relations of each annotation document by the span and key of their endpoints (mention ids differ)."""
    if relations.empty:
        return {}
    triple = dict(zip(mentions["mention_id"], zip(mentions["begin"], mentions["end"], mentions["key"])))
    out: Dict[Pair, list] = {}
    for r in relations.itertuples(index=False):
        kind = "" if pd.isna(r.kind) else str(r.kind)
        out.setdefault((r.doc, r.user), []).append((triple.get(r.governor), triple.get(r.dependent), kind))
    return {key: tuple(sorted(v, key=repr)) for key, v in out.items()}


def copy_groups(signatures: Mapping[Pair, tuple], candidates: Sequence[Pair]) -> List[List[Pair]]:
    """Groups (two or more) of non-empty candidate annotation documents of the same document with identical
    mentions."""
    groups: Dict[Tuple[str, tuple], List[Pair]] = {}
    for doc, user in candidates:
        signature = signatures.get((doc, user), ())
        if signature:
            groups.setdefault((doc, signature), []).append((doc, user))
    return [sorted(g) for g in groups.values() if len(g) > 1]


def select_documents(mentions: pd.DataFrame, pre: pd.DataFrame, relations: pd.DataFrame,
                     doc_users: Mapping[str, Sequence[str]], names: pd.DataFrame, *, exclude_untouched: bool = True,
                     exclude_low_change: bool = False, exclude_copies: bool = True,
                     low_change: float = 0.05) -> Selection:
    """Decide the exclusions (see the module docstring).

    ``mentions`` and ``pre`` carry concept keys (enriched); ``relations`` are the loaded ones (mention ids);
    ``doc_users`` lists the loaded annotators of every document (also those with an empty document); ``names`` maps
    user -> ui_name, site.
    """
    pairs = [(doc, user) for doc in sorted(doc_users) for user in sorted(doc_users[doc])]
    n_mentions = Counter(zip(mentions["doc"], mentions["user"]))
    n_relations = Counter(zip(relations["doc"], relations["user"])) if len(relations) else Counter()
    compared = PA.compare(pre, mentions, doc_users, low_change)[1]       # empty without a pre-annotation
    f1 = dict(zip(zip(compared["doc"], compared["user"]), compared["f1"]))

    reason: Dict[Pair, str] = {}
    if exclude_untouched:
        reason.update({(d, u): UNTOUCHED for d, u in compared.loc[compared["untouched"], ["doc", "user"]].values})
    if exclude_low_change:
        reason.update({(d, u): LOW_CHANGE for d, u in compared.loc[compared["low_change"], ["doc", "user"]].values
                       if (d, u) not in reason})
    copy_of: Dict[Pair, Pair] = {}
    if exclude_copies:
        candidates = [p for p in pairs if p not in reason]
        groups = copy_groups(_signatures(mentions), candidates)
        copied = Counter(user for g in groups for _, user in g)
        documents = Counter(user for _, user in candidates)
        ui = names["ui_name"].to_dict()

        def rank(pair: Pair):
            user = pair[1]
            return copied[user] / documents[user], str(ui.get(user, user))
        for group in groups:
            kept = min(group, key=rank)
            for pair in group:
                if pair != kept:
                    reason[pair], copy_of[pair] = COPY, kept

    rel = _relation_signatures(relations, mentions)
    rows = []
    for (doc, user), why in reason.items():
        source = copy_of.get((doc, user))
        rows.append({"doc": doc, "user": user, "ui_name": names["ui_name"].get(user, user),
                     "site": names["site"].get(user, "unknown"), "reason": why,
                     "copy_of": names["ui_name"].get(source[1], source[1]) if source else "",
                     "n_mentions": n_mentions[(doc, user)], "n_relations": n_relations[(doc, user)],
                     "f1_preannotation": f1.get((doc, user), float("nan")),
                     "relations_identical": rel.get((doc, user), ()) == rel.get(source, ()) if source else None})
    excluded = pd.DataFrame(rows, columns=EXCLUDED_COLUMNS)
    excluded["_order"] = excluded["reason"].map(REASONS.index)
    excluded = excluded.sort_values(["_order", "doc", "ui_name"]).drop(columns="_order").reset_index(drop=True)

    low = compared[compared["low_change"] & ~compared["untouched"]]
    low_table = pd.DataFrame({"doc": low["doc"], "user": low["user"],
                              "ui_name": low["user"].map(names["ui_name"]), "site": low["user"].map(names["site"]),
                              "n_pre": low["n_pre"], "n_annotator": low["n_annotator"], "f1_preannotation": low["f1"],
                              "excluded_as": [reason.get((d, u), "") for d, u in zip(low["doc"], low["user"])]},
                             columns=LOW_CHANGE_COLUMNS).sort_values(["doc", "ui_name"]).reset_index(drop=True)

    remaining = Counter(user for doc, user in pairs if (doc, user) not in reason)
    removed = sorted(str(names["ui_name"].get(u, u)) for u in {u for _, u in pairs} if not remaining[u])
    return Selection(excluded, low_table,
                     loaded={"annotation documents": len(pairs), "mentions": len(mentions),
                             "annotators": len({u for _, u in pairs})},
                     removed_annotators=removed,
                     settings={"EXCLUDE_UNTOUCHED": exclude_untouched, "EXCLUDE_COPIES": exclude_copies,
                               "EXCLUDE_LOW_CHANGE": exclude_low_change, "PREANNO_LOW_CHANGE": low_change,
                               "pre-annotation available": bool(len(pre))})


def _without(df: pd.DataFrame, pairs: FrozenSet[Pair]) -> pd.DataFrame:
    if df.empty or not pairs:
        return df
    return df[~pd.MultiIndex.from_arrays([df["doc"], df["user"]]).isin(list(pairs))].reset_index(drop=True)


def apply(data: ProjectData, pairs: FrozenSet[Pair]) -> ProjectData:
    """``data`` without the annotation documents ``pairs``; annotators without any document left are dropped."""
    if not pairs:
        return data
    docs = data.docs.copy()
    docs["users"] = [sorted(u for u in users if (doc, u) not in pairs)
                     for doc, users in zip(docs["doc"], docs["users"])]
    docs["n_annotators"] = pd.Series([len(u) for u in docs["users"]], index=docs.index, dtype="int64")
    per_user = Counter(u for users in docs["users"] for u in users)
    annotators = data.annotators.assign(n_docs=data.annotators["user"].map(per_user).fillna(0).astype(int))
    return dataclasses.replace(data, docs=docs, annotators=annotators[annotators["n_docs"] > 0].reset_index(drop=True),
                               mentions=_without(data.mentions, pairs), relations=_without(data.relations, pairs),
                               no_human=_without(data.no_human, pairs))


def require_annotators(data: ProjectData, minimum: int, selection: Selection) -> ProjectData:
    """``data`` (after the exclusions) without the documents left with fewer than ``minimum`` annotators; they are
    recorded in ``selection.too_few``, and annotators without any document left in ``removed_annotators``."""
    few = data.docs[data.docs["n_annotators"] < minimum]
    mentions = data.mentions.groupby("doc").size()
    selection.too_few = pd.DataFrame({"doc": few["doc"], "annotators": few["n_annotators"],
                                      "mentions": few["doc"].map(mentions).fillna(0).astype(int)},
                                     columns=TOO_FEW_COLUMNS).reset_index(drop=True)
    selection.settings["MIN_ANNOTATORS"] = minimum
    out = without_documents(data, few["doc"])
    ui = data.annotators.set_index("user")["ui_name"]
    lost = set(data.annotators["user"]) - set(out.annotators["user"])
    selection.removed_annotators = sorted(set(selection.removed_annotators) | {str(ui.get(u, u)) for u in lost})
    return out


def summary(selection: Selection, data: ProjectData) -> pd.DataFrame:
    """Annotation documents and mentions loaded, excluded per reason, dropped with their document (too few
    annotators left) and analysed (``data``: after all of it)."""
    ex = selection.excluded
    rows = [("loaded", selection.loaded["annotation documents"], selection.loaded["mentions"])]
    rows += [(f"excluded: {r}", int((ex["reason"] == r).sum()), int(ex.loc[ex["reason"] == r, "n_mentions"].sum()))
             for r in REASONS]
    rows.append(("dropped: too few annotators left", int(selection.too_few["annotators"].sum()),
                 int(selection.too_few["mentions"].sum())))
    rows.append(("analysed", int(data.docs["n_annotators"].sum()), len(data.mentions)))
    return pd.DataFrame(rows, columns=["step", "annotation documents", "mentions"]).set_index("step")


def by_annotator(selection: Selection) -> pd.DataFrame:
    """Excluded annotation documents per annotator and reason."""
    ex = selection.excluded
    if ex.empty:
        return pd.DataFrame(columns=REASONS)
    return pd.crosstab(ex["ui_name"], ex["reason"]).reindex(columns=REASONS, fill_value=0)
