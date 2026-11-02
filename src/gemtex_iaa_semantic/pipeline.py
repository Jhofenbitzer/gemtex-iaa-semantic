"""From the loaded exports to the aligned, selected mentions every figure is computed from."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from gemtex_iaa_semantic import align as A
from gemtex_iaa_semantic import selection as SEL
from gemtex_iaa_semantic.config import Config
from gemtex_iaa_semantic.io.loader import ProjectData, get_project
from gemtex_iaa_semantic.snomed import SnomedStore, enrich_mentions

log = logging.getLogger(__name__)


@dataclass
class Context:
    cfg: Config
    store: SnomedStore
    data: ProjectData                  # after the exclusions
    mentions: pd.DataFrame             # with concept keys (snomed.enrich_mentions)
    clustered: pd.DataFrame            # mentions with their multi-annotator cluster
    regions: List[A.Region]            # pairwise regions of all annotator pairs of every document
    preannotation: pd.DataFrame        # INITIAL_CAS concepts with keys; empty without PREANNOTATION
    selection: SEL.Selection           # excluded annotation documents


def prepare(cfg: Config, store: Optional[SnomedStore] = None, data: Optional[ProjectData] = None,
            use_cache: bool = True) -> Context:
    store = store or SnomedStore(cfg.HDF5)
    data = data or get_project(cfg, use_cache=use_cache)
    mentions = enrich_mentions(data.mentions, store)
    pre = data.preannotation
    if len(pre) and (cfg.PREANNOTATION or cfg.EXCLUDE_UNTOUCHED or cfg.EXCLUDE_LOW_CHANGE):
        pre = enrich_mentions(pre, store)
    selection = SEL.select_documents(
        mentions, pre if "key" in pre else pre.iloc[0:0], data.relations,
        dict(zip(data.docs["doc"], data.docs["users"].apply(list))),
        data.annotators.set_index("user")[["ui_name", "site"]], exclude_untouched=cfg.EXCLUDE_UNTOUCHED,
        exclude_low_change=cfg.EXCLUDE_LOW_CHANGE, exclude_copies=cfg.EXCLUDE_COPIES, low_change=cfg.PREANNO_LOW_CHANGE)
    log.info("excluded annotation documents: %s", selection.describe())
    data = SEL.require_annotators(SEL.apply(data, selection.pairs), cfg.MIN_ANNOTATORS, selection)
    if len(selection.too_few):
        log.info("documents left with fewer than %d annotators (dropped): %d", cfg.MIN_ANNOTATORS,
                 len(selection.too_few))
    mentions = mentions[mentions["mention_id"].isin(data.mentions["mention_id"])].reset_index(drop=True)
    docs = data.docs.copy()
    docs["users"] = docs["users"].apply(list)
    return Context(cfg, store, data, mentions, A.clusters(mentions), list(A.all_pair_regions(mentions, docs)),
                   pre if cfg.PREANNOTATION and "key" in pre else pre.iloc[0:0], selection)


def selection_tables(ctx: Context) -> Dict[str, pd.DataFrame]:
    """Tables of the excluded annotation documents (for ``report.write_exclusion_report``)."""
    sel = ctx.selection
    return {"exclusion_summary": SEL.summary(sel, ctx.data), "excluded_by_annotator": SEL.by_annotator(sel),
            "excluded_annotation_documents": sel.excluded, "low_change_annotation_documents": sel.low_change,
            "too_few_annotators_documents": sel.too_few}
