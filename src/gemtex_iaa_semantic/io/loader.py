"""Load one or several zipped INCEpTION XMI project exports into tidy tables.

CASes are read with ``io.cas.annotation_info_from_xmi_zip`` (adapted from INCEpTALYTICS). Annotation states and
annotator display names (``<site>_<annotator>``) come from ``exportedproject.json``. The pre-annotation every
annotator started from (``INITIAL_CAS``) is not an annotator; its concepts go into a table of their own.

Only documents with at least ``MIN_ANNOTATORS`` annotators in ``STATES`` (over all exports) are read; the others
are counted (:func:`annotation_status`). Several exports become one dataset (:func:`combine`): a document name in
several exports is one document if its text is the same in all of them, and accounts are identified by username.
"""
from __future__ import annotations

import bisect
import dataclasses
import glob
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Optional, Sequence, Tuple
from zipfile import ZipFile

import pandas as pd
from cassis.typesystem import TypeNotFoundError

from gemtex_iaa_semantic.config import Config
from gemtex_iaa_semantic.io.cas import SENTENCE_TYPE_NAME, annotation_info_from_xmi_zip

log = logging.getLogger(__name__)

INITIAL_CAS = "INITIAL_CAS"
ALWAYS_SKIPPED = (INITIAL_CAS, "CURATION_USER")
PREANNOTATION_SITE = "preannotation"
CACHE_VERSION = 3           # 2: pre-annotation table; 3: several exports, MIN_ANNOTATORS, SITE
# annotation document states that count as assigned to an annotator (INCEpTION's IGNORE does not)
ASSIGNED_STATES = frozenset({"NEW", "IN_PROGRESS", "FINISHED"})
# document status by its number of annotators (annotation_status)
ANALYSABLE, PENDING, TOO_FEW = "analysable", "pending", "too_few"
STATUSES = [ANALYSABLE, PENDING, TOO_FEW]
NUMBER_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
DATE_LITERAL_RE = re.compile(r"^\d{4}(?:-(?:\d{2}|xx)(?:-(?:\d{2}|xx))?)?$", re.I)
IRI_RE = re.compile(r"^https?://snomed\.info/id/\d+$")

MENTION_COLUMNS = ["mention_id", "doc", "user", "ui_name", "site", "annotator", "xmi_id", "begin", "end", "text",
                   "sentence", "sent_begin", "sent_end", "id_raw", "literal", "value_kind"]
RAW_COLUMNS = ["mention_id", "doc", "user", "begin", "end", "text", "sentence", "sent_begin", "sent_end", "id_raw",
               "literal", "value_kind", "xmi_id"]
TABLES = ("docs", "annotators", "mentions", "relations", "no_human", "preannotation")


@dataclass
class ProjectData:
    docs: pd.DataFrame           # doc, text, users (included annotators), n_annotators
    annotators: pd.DataFrame     # user, ui_name, site, annotator, n_docs
    mentions: pd.DataFrame       # see MENTION_COLUMNS
    relations: pd.DataFrame      # doc, user, governor, dependent (mention ids), kind
    no_human: pd.DataFrame       # doc, user, begin, end
    meta: Dict = field(default_factory=dict)
    # concepts of INITIAL_CAS (MENTION_COLUMNS, user INITIAL_CAS); mention ids follow the annotators' ones
    preannotation: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=MENTION_COLUMNS))

    def doc_text(self, doc: str) -> str:
        return self.docs.set_index("doc").at[doc, "text"]


def value_kind(id_raw: Optional[str], literal: Optional[str]) -> str:
    """Classify an annotation value (guideline 3.3.3.1/3.3.3.2: numbers and dates live in ``literal``)."""
    has_id = bool(id_raw and str(id_raw).strip())
    lit = (literal or "").strip()
    if has_id:
        return "code_with_literal" if lit else "code"
    if not lit:
        return "empty"
    if NUMBER_RE.match(lit):
        return "number"
    if DATE_LITERAL_RE.match(lit):
        return "date"
    return "literal_other"


def id_format(id_raw: Optional[str]) -> str:
    if id_raw is None or not str(id_raw).strip():
        return "empty"
    text = str(id_raw).strip()
    if IRI_RE.match(text):
        return "snomed_iri"
    if text.isdigit():
        return "bare_sctid"
    return "other"


def project_paths(cfg: Config) -> List[Path]:
    """The export files of ``PATH_PROJECT``: files, every ``*.zip`` of a directory, or the matches of a glob."""
    paths: List[Path] = []
    for entry in cfg.PATH_PROJECT:
        if any(c in str(entry) for c in "*?["):
            matches = sorted(Path(p) for p in glob.glob(str(entry)))
        elif entry.is_dir():
            matches = sorted(entry.glob("*.zip"))
        else:
            matches = [entry]
        if not matches:
            raise FileNotFoundError(f"No project export matches PATH_PROJECT entry {entry}")
        paths += [m for m in matches if m.resolve() not in {p.resolve() for p in paths}]
    if not paths:
        raise FileNotFoundError("No project export configured (PATH_PROJECT in [default]).")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Project export not found: {', '.join(missing)}")
    return paths


def read_exported_project(project_path) -> Dict:
    with ZipFile(project_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith("exportedproject.json"))
        return json.loads(zf.read(name))


def _annotator_table(project: Dict, site_pattern: str, site: str = "") -> pd.DataFrame:
    """Accounts of the project; ``site``, if given, is the site of all of them (else from ``site_pattern``)."""
    pattern = re.compile(site_pattern)
    rows = []
    for user in project.get("project_users", []):
        ui_name = user.get("uiName") or user["username"]
        match = pattern.match(ui_name)
        rows.append({"user": user["username"], "ui_name": ui_name,
                     "site": site or (match.group("site") if match else "unknown"),
                     "annotator": match.group("annotator") if match else ui_name})
    return pd.DataFrame(rows, columns=["user", "ui_name", "site", "annotator"])


def _skipped_users(project: Dict, cfg: Config) -> set:
    """Usernames never analysed: INITIAL_CAS, CURATION_USER and SKIP_ANNOTATORS (by username or uiName)."""
    skip = set(ALWAYS_SKIPPED) | set(cfg.SKIP_ANNOTATORS)
    return skip | {u["username"] for u in project.get("project_users", []) if u.get("uiName") in skip}


def annotation_status(projects: Sequence[Dict], cfg: Config) -> pd.DataFrame:
    """Per document name over all exports: annotators with an annotation document in STATES (``included``) and
    assigned to it (``assigned``: NEW, IN_PROGRESS, FINISHED or STATES, not IGNORE), and the ``status``:
    ``analysable`` (included >= MIN_ANNOTATORS), ``pending`` (only assigned >= MIN_ANNOTATORS) or ``too_few``."""
    states = set(cfg.STATES)
    assigned_states = ASSIGNED_STATES | states
    included: Dict[str, set] = {}
    assigned: Dict[str, set] = {}
    for project in projects:
        skip = _skipped_users(project, cfg)
        for d in project.get("source_documents", []):
            assigned.setdefault(d["name"], set())
        for d in project.get("annotation_documents", []):
            if d["user"] in skip:
                continue
            if d.get("state") in assigned_states:
                assigned.setdefault(d["name"], set()).add(d["user"])
            if d.get("state") in states:
                included.setdefault(d["name"], set()).add(d["user"])
    docs = sorted(set(assigned) | set(included))
    out = pd.DataFrame({"doc": docs, "included": [len(included.get(d, ())) for d in docs],
                        "assigned": [len(assigned.get(d, ())) for d in docs]},
                       columns=["doc", "included", "assigned"])
    out["status"] = [ANALYSABLE if i >= cfg.MIN_ANNOTATORS else PENDING if a >= cfg.MIN_ANNOTATORS else TOO_FEW
                     for i, a in zip(out["included"], out["assigned"])]
    return out


def _safe_select(cas, type_name):
    try:
        return list(cas.select(type_name))
    except TypeNotFoundError:
        return []


def _concepts(cas, doc: str, text: str, user: str, cfg: Config, first_id: int) -> Tuple[List[tuple], Dict[int, int]]:
    """Concept annotations of one CAS as mention rows (ids from ``first_id``) and their xmi id -> mention id."""
    sentences = sorted((s.begin, s.end) for s in _safe_select(cas, SENTENCE_TYPE_NAME))
    starts = [s[0] for s in sentences]
    rows, by_xmi = [], {}
    for ann in _safe_select(cas, cfg.CONCEPT_LAYER):
        s_idx = bisect.bisect_right(starts, ann.begin) - 1 if sentences else -1
        sent = sentences[s_idx] if s_idx >= 0 else (0, len(text))
        mention_id = first_id + len(rows)
        by_xmi[ann.xmiID] = mention_id
        id_raw, literal = ann.get(cfg.ID_FEATURE), ann.get(cfg.LITERAL_FEATURE)
        rows.append((mention_id, doc, user, ann.begin, ann.end, ann.get_covered_text(), s_idx,
                     sent[0], sent[1], id_raw, literal, value_kind(id_raw, literal), ann.xmiID))
    return rows, by_xmi


def load_export(path: Path, project: Dict, cfg: Config, documents: Optional[Collection[str]] = None) -> ProjectData:
    """One export (``project``: its exportedproject.json); only ``documents`` are read if given."""
    states = set(cfg.STATES)
    included = {(d["name"], d["user"]) for d in project.get("annotation_documents", []) if d.get("state") in states}
    annotators = _annotator_table(project, cfg.SITE_PATTERN, cfg.SITE)
    ui_names = annotators.set_index("user")["ui_name"].to_dict()
    skip = set(ALWAYS_SKIPPED) | set(cfg.SKIP_ANNOTATORS)

    mentions, relations, no_human, initial, texts = [], [], [], [], {}
    typesystem_types = []
    skipped = Counter()
    loaded_pairs = set()
    initial_relations = 0
    for cas, doc, user in annotation_info_from_xmi_zip(str(path), documents):
        if not typesystem_types:
            typesystem_types = [t.name for t in cas.typesystem.get_types()]
        texts.setdefault(doc, cas.sofa_string or "")
        if user == INITIAL_CAS:
            initial += _concepts(cas, doc, texts[doc], user, cfg, len(initial))[0]
            initial_relations += len(_safe_select(cas, cfg.RELATION_LAYER))
            continue
        if user in skip or ui_names.get(user) in skip:
            skipped["skip_list"] += 1
            continue
        if (doc, user) not in included:
            skipped["state"] += 1
            continue
        loaded_pairs.add((doc, user))

        rows, by_xmi = _concepts(cas, doc, texts[doc], user, cfg, len(mentions))
        mentions += rows
        for rel in _safe_select(cas, cfg.RELATION_LAYER):
            gov, dep = rel.get("Governor"), rel.get("Dependent")
            if gov is None or dep is None or gov.xmiID not in by_xmi or dep.xmiID not in by_xmi:
                skipped["dangling_relation"] += 1
                continue
            relations.append((doc, user, by_xmi[gov.xmiID], by_xmi[dep.xmiID], rel.get(cfg.RELATION_FEATURE)))
        for nh in _safe_select(cas, cfg.NO_HUMAN_LAYER):
            no_human.append((doc, user, nh.begin, nh.end))

    m = pd.DataFrame(mentions, columns=RAW_COLUMNS)
    m = m.merge(annotators[["user", "ui_name", "site", "annotator"]], on="user", how="left")
    m["ui_name"] = m["ui_name"].fillna(m["user"])
    m["site"] = m["site"].fillna("unknown")
    m["annotator"] = m["annotator"].fillna(m["user"])
    m = m[MENTION_COLUMNS]
    pre = pd.DataFrame(initial, columns=RAW_COLUMNS).assign(ui_name=INITIAL_CAS, site=PREANNOTATION_SITE,
                                                            annotator=INITIAL_CAS)[MENTION_COLUMNS]
    pre["mention_id"] = pre["mention_id"].astype("int64") + len(m)

    docs = pd.DataFrame({"doc": sorted(texts), "text": [texts[d] for d in sorted(texts)]})
    users_per_doc = {}
    for d, u in loaded_pairs:
        users_per_doc.setdefault(d, []).append(u)
    docs["users"] = [sorted(users_per_doc.get(d, [])) for d in docs["doc"]]
    docs["n_annotators"] = pd.Series([len(u) for u in docs["users"]], index=docs.index, dtype="int64")
    per_user = Counter(u for _, u in loaded_pairs)
    annotators["n_docs"] = annotators["user"].map(per_user).fillna(0).astype(int)

    meta = {
        "project_name": project.get("name"), "inception_version": project.get("application_version"),
        "states": sorted(states), "skipped": dict(skipped), "types": typesystem_types,
        "layers": [{"name": L.get("name"), "type": L.get("type"), "overlap": L.get("overlap_mode"),
                    "features": [f.get("name") for f in L.get("features", [])]} for L in project.get("layers", [])],
        "state_counts": dict(Counter(d.get("state") for d in project.get("annotation_documents", []))),
        "preannotation_relations": initial_relations,
    }
    return ProjectData(docs=docs, annotators=annotators[annotators["n_docs"] > 0].reset_index(drop=True),
                       mentions=m.reset_index(drop=True),
                       relations=pd.DataFrame(relations, columns=["doc", "user", "governor", "dependent", "kind"]),
                       no_human=pd.DataFrame(no_human, columns=["doc", "user", "begin", "end"]), meta=meta,
                       preannotation=pre.reset_index(drop=True))


def without_documents(data: ProjectData, docs: Iterable[str]) -> ProjectData:
    """``data`` without the documents ``docs``; annotators without any document left are dropped."""
    docs = set(docs)
    if not docs:
        return data

    def keep(df: pd.DataFrame) -> pd.DataFrame:
        return df[~df["doc"].isin(docs)].reset_index(drop=True)
    kept = keep(data.docs)
    per_user = Counter(u for users in kept["users"] for u in users)
    annotators = data.annotators.assign(n_docs=data.annotators["user"].map(per_user).fillna(0).astype(int))
    return dataclasses.replace(data, docs=kept, annotators=annotators[annotators["n_docs"] > 0].reset_index(drop=True),
                               mentions=keep(data.mentions), relations=keep(data.relations),
                               no_human=keep(data.no_human), preannotation=keep(data.preannotation))


def _labels(paths: Sequence[Path]) -> List[str]:
    """Short unique names of the exports (file stems)."""
    stems = [p.stem for p in paths]
    return [s if stems.count(s) == 1 else f"{s}-{i + 1}" for i, s in enumerate(stems)]


def combine(parts: Sequence[ProjectData], labels: Sequence[str]) -> ProjectData:
    """One dataset from several exports (``labels``: their names).

    A document name found in several exports is one document if its text is the same in all of them; otherwise each
    text variant becomes the document ``<label>/<name>`` (label of the first export with it). Accounts are identified
    by username: an annotation document (document x user) already loaded from an earlier export is skipped
    (``skipped: duplicate``), and so is the pre-annotation of a document already loaded. Mention ids are renumbered.
    """
    if len(parts) == 1:
        return parts[0]
    variants: Dict[str, Dict[str, int]] = {}            # name -> text -> first export with it
    for i, part in enumerate(parts):
        for doc, text in zip(part.docs["doc"], part.docs["text"]):
            variants.setdefault(doc, {}).setdefault(text, i)
    for doc, texts in variants.items():
        if len(texts) > 1:
            log.warning("document %s has different texts in %d exports; they are analysed as different documents",
                        doc, len(texts))

    def name(doc: str, text: str) -> str:
        texts = variants[doc]
        return doc if len(texts) == 1 else f"{labels[texts[text]]}/{doc}"

    seen, pre_docs, skipped = set(), set(), Counter()
    mentions, relations, no_human, pre, annotators = [], [], [], [], []
    texts: Dict[str, str] = {}
    users: Dict[str, set] = {}
    for part in parts:
        rename = {doc: name(doc, text) for doc, text in zip(part.docs["doc"], part.docs["text"])}
        pairs = {(rename[doc], u) for doc, us in zip(part.docs["doc"], part.docs["users"]) for u in us}
        duplicate = pairs & seen
        skipped["duplicate"] += len(duplicate)
        seen |= pairs

        def own(df: pd.DataFrame) -> pd.DataFrame:
            df = df.assign(doc=df["doc"].map(rename))
            if not duplicate or df.empty:
                return df
            return df[~pd.MultiIndex.from_arrays([df["doc"], df["user"]]).isin(list(duplicate))]
        m = own(part.mentions)
        first = sum(len(x) for x in mentions)
        ids = dict(zip(m["mention_id"], range(first, first + len(m))))
        mentions.append(m.assign(mention_id=m["mention_id"].map(ids)))
        r = own(part.relations)
        r = r[r["governor"].isin(set(ids)) & r["dependent"].isin(set(ids))]
        relations.append(r.assign(governor=r["governor"].map(ids), dependent=r["dependent"].map(ids)))
        no_human.append(own(part.no_human))
        p = part.preannotation.assign(doc=part.preannotation["doc"].map(rename))
        pre.append(p[~p["doc"].isin(pre_docs)])
        pre_docs |= set(p["doc"])
        for doc, text, us in zip(part.docs["doc"], part.docs["text"], part.docs["users"]):
            texts.setdefault(rename[doc], text)
            users.setdefault(rename[doc], set()).update(us)
        annotators.append(part.annotators)

    m = pd.concat(mentions, ignore_index=True)
    p = pd.concat(pre, ignore_index=True)
    p["mention_id"] = range(len(m), len(m) + len(p))
    docs = pd.DataFrame({"doc": sorted(texts), "text": [texts[d] for d in sorted(texts)]})
    docs["users"] = [sorted(users[d]) for d in docs["doc"]]
    docs["n_annotators"] = pd.Series([len(u) for u in docs["users"]], index=docs.index, dtype="int64")
    per_user = Counter(u for us in docs["users"] for u in us)
    accounts = pd.concat(annotators, ignore_index=True).drop_duplicates("user")
    accounts = accounts.assign(n_docs=accounts["user"].map(per_user).fillna(0).astype(int))
    meta = {
        "project_name": "; ".join(str(part.meta.get("project_name")) for part in parts),
        "inception_version": "; ".join(sorted({str(part.meta.get("inception_version")) for part in parts})),
        "states": parts[0].meta.get("states"),
        "skipped": dict(sum((Counter(part.meta.get("skipped", {})) for part in parts), Counter()) + skipped),
        "types": sorted(set().union(*(part.meta.get("types", []) for part in parts))),
        "layers": parts[0].meta.get("layers", []),
        "state_counts": dict(sum((Counter(part.meta.get("state_counts", {})) for part in parts), Counter())),
        "preannotation_relations": sum(part.meta.get("preannotation_relations", 0) for part in parts),
    }
    return ProjectData(docs=docs, annotators=accounts[accounts["n_docs"] > 0].reset_index(drop=True),
                       mentions=m, relations=pd.concat(relations, ignore_index=True),
                       no_human=pd.concat(no_human, ignore_index=True), meta=meta, preannotation=p)


def load_project(cfg: Config) -> ProjectData:
    """Load the exports of PATH_PROJECT as one dataset (see the module docstring). Documents with fewer than
    MIN_ANNOTATORS annotators in STATES are not read, or dropped after combining the exports."""
    paths = project_paths(cfg)
    projects = [read_exported_project(p) for p in paths]
    status = annotation_status(projects, cfg)
    documents = set(status.loc[status["status"] == ANALYSABLE, "doc"]) if cfg.MIN_ANNOTATORS > 0 else None
    data = combine([load_export(path, project, cfg, documents) for path, project in zip(paths, projects)],
                   _labels(paths))
    too_few = data.docs.loc[data.docs["n_annotators"] < cfg.MIN_ANNOTATORS, "doc"]
    data = without_documents(data, too_few)
    data.meta.update({"exports": len(paths), "documents_in_exports": len(status),
                      "document_status": {s: int((status["status"] == s).sum()) for s in STATUSES},
                      "min_annotators": cfg.MIN_ANNOTATORS, "documents_dropped_too_few": len(too_few)})
    return data


# ---------------- cache ----------------

def cache_dir(cfg: Config) -> Path:
    """Named after the export, or after the number of exports and a hash of their paths."""
    paths = project_paths(cfg)
    if len(paths) == 1:
        return Path(cfg.DIR_CACHE) / paths[0].stem
    digest = hashlib.sha1("\n".join(sorted(str(p.resolve()) for p in paths)).encode()).hexdigest()[:10]
    return Path(cfg.DIR_CACHE) / f"{len(paths)}-exports-{digest}"


def save_cache(data: ProjectData, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        getattr(data, name).to_parquet(directory / f"{name}.parquet", index=False)
    (directory / "meta.json").write_text(json.dumps(data.meta, indent=1), encoding="utf-8")


def load_cached(directory: Path) -> Optional[ProjectData]:
    if not all((directory / f"{n}.parquet").exists() for n in TABLES) or not (directory / "meta.json").exists():
        return None
    tables = {n: pd.read_parquet(directory / f"{n}.parquet") for n in TABLES}
    return ProjectData(**tables, meta=json.loads((directory / "meta.json").read_text(encoding="utf-8")))


def _cache_key(cfg: Config) -> List:
    return [CACHE_VERSION, [[str(p.resolve()), p.stat().st_mtime] for p in project_paths(cfg)],
            sorted(cfg.STATES), sorted(cfg.SKIP_ANNOTATORS), cfg.CONCEPT_LAYER, cfg.ID_FEATURE, cfg.LITERAL_FEATURE,
            cfg.RELATION_LAYER, cfg.RELATION_FEATURE, cfg.NO_HUMAN_LAYER, cfg.SITE_PATTERN, cfg.SITE,
            cfg.MIN_ANNOTATORS]


def get_project(cfg: Config, use_cache: bool = True) -> ProjectData:
    """Load the exports, re-using the parquet cache if it was built from the same files and settings."""
    directory = cache_dir(cfg)
    key = _cache_key(cfg)
    if use_cache:
        cached = load_cached(directory)
        if cached is not None and cached.meta.get("cache_key") == key:
            log.info("Using cached tables from %s", directory)
            return cached
    data = load_project(cfg)
    data.meta["cache_key"] = key
    save_cache(data, directory)
    return data
