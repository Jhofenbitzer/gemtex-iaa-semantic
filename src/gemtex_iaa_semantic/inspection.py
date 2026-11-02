"""Text-free diagnostics of the exports (`inspect`): counts and format histograms that can be shared safely."""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

from gemtex_iaa_semantic.config import Config
from gemtex_iaa_semantic.io.loader import ProjectData, id_format
from gemtex_iaa_semantic.snomed import INACTIVE, MALFORMED, UNKNOWN, SnomedStore, enrich_mentions


def _counter_lines(title: str, counter, top: Optional[int] = None) -> List[str]:
    items = counter.most_common(top) if isinstance(counter, Counter) else list(counter.items())
    lines = [f"{title}:"]
    lines += [f"  {k}: {v}" for k, v in items]
    return lines


def inspect_project(data: ProjectData, cfg: Config, store: Optional[SnomedStore] = None) -> str:
    m = data.mentions
    lines = [f"project: {data.meta.get('project_name')} (INCEpTION {data.meta.get('inception_version')})",
             f"exports: {data.meta.get('exports', 1)}; documents in the exports: "
             f"{data.meta.get('documents_in_exports')}",
             f"included states: {data.meta.get('states')}"]
    lines += _counter_lines(f"documents by annotators in these states (read: analysable, i.e. at least "
                            f"{cfg.MIN_ANNOTATORS}; pending: assigned to enough, finished by fewer)",
                            Counter(data.meta.get("document_status", {})))
    lines.append(f"documents dropped after reading (fewer than {cfg.MIN_ANNOTATORS} annotators, e.g. different "
                 f"texts in several exports): {data.meta.get('documents_dropped_too_few', 0)}")
    lines += _counter_lines("annotation document states (exportedproject.json)",
                            Counter(data.meta.get("state_counts", {})))
    lines += _counter_lines("skipped CASes", Counter(data.meta.get("skipped", {})))
    lines.append("layers (exportedproject.json):")
    for layer in data.meta.get("layers", []):
        lines.append(f"  {layer['name']} [{layer['type']}, overlap={layer.get('overlap')}] "
                     f"features={layer['features']}")
    custom = [t for t in data.meta.get("types", []) if t.startswith(("gemtex", "custom", "webanno"))]
    lines.append(f"custom types in TypeSystem: {custom}")
    lines.append(f"configured layers: concept={cfg.CONCEPT_LAYER}>{cfg.ID_FEATURE}/{cfg.LITERAL_FEATURE}, "
                 f"relation={cfg.RELATION_LAYER}>{cfg.RELATION_FEATURE}, no_human={cfg.NO_HUMAN_LAYER}")
    lines.append(f"documents: {len(data.docs)}; "
                 f"with >= 2 included annotators: {(data.docs['n_annotators'] >= 2).sum()}")
    lines += _counter_lines("included annotators per document (n annotators: n documents)",
                            dict(sorted(Counter(data.docs["n_annotators"]).items())))
    lines += _counter_lines("annotators per site", Counter(data.annotators["site"]))
    lines.append(f"annotators: {len(data.annotators)}; site 'unknown' (no SITE, uiName not matching SITE_PATTERN): "
                 f"{(data.annotators['site'] == 'unknown').sum()}")
    lines.append(f"documents per annotator: min {data.annotators['n_docs'].min()}, "
                 f"median {data.annotators['n_docs'].median()}, max {data.annotators['n_docs'].max()}")
    lines.append(f"mentions: {len(m)}")
    lines += _counter_lines("id feature formats", Counter(m["id_raw"].map(id_format)))
    lines += _counter_lines("value kinds (id / literal)", Counter(m["value_kind"]))
    lines.append(f"relations: {len(data.relations)}; No_Human spans: {len(data.no_human)}")
    pre = data.preannotation
    lines.append(f"pre-annotation (INITIAL_CAS): {len(pre)} concepts in {pre['doc'].nunique()} documents, "
                 f"{data.meta.get('preannotation_relations', 0)} relations")
    stacked = m.groupby(["doc", "user", "begin", "end"]).size()
    lines.append(f"stacked spans (same annotator, same span, >1 concept): {(stacked > 1).sum()}")

    if store is not None:
        e = enrich_mentions(m, store)
        lines.append(f"SNOMED CT release in HDF5: {store.release_date} ({len(store)} concepts, "
                     f"{store.n_active} active)")
        lines += _counter_lines("resolution status", Counter(e["status"]))
        bad = e[e["status"].isin([UNKNOWN, INACTIVE, MALFORMED])]
        codes = Counter(f"{s}:{(c if c else '<unparsed>')}" for s, c in zip(bad["status"], bad["sctid"]))
        lines += _counter_lines("unresolved codes (status:sctid, top 30)", codes, top=30)
    return "\n".join(lines)
