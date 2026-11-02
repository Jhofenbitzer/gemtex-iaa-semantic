"""Build small synthetic INCEpTION XMI project exports (same layout and layer types as the real export)."""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from cassis import Cas, TypeSystem

SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
IRI = "http://snomed.info/id/"


def typesystem() -> TypeSystem:
    ts = TypeSystem()
    ts.create_type(SENTENCE)
    concept = ts.create_type("gemtex.Concept")
    ts.create_feature(concept, "id", "uima.cas.String")
    ts.create_feature(concept, "literal", "uima.cas.String")
    relation = ts.create_type("gemtex.Relation")
    ts.create_feature(relation, "Governor", "gemtex.Concept")
    ts.create_feature(relation, "Dependent", "gemtex.Concept")
    ts.create_feature(relation, "kind", "uima.cas.String")
    ts.create_type("webanno.custom.No_Human")
    return ts


@dataclass
class M:
    """A mention spec: ``text`` (the n-th occurrence ``occ`` in the document) or an explicit ``span``."""
    text: Optional[str] = None
    code: Optional[str] = None
    literal: Optional[str] = None
    occ: int = 0
    span: Optional[Tuple[int, int]] = None

    def resolve(self, doc_text: str) -> Tuple[int, int]:
        if self.span:
            return self.span
        starts = [m.start() for m in re.finditer(re.escape(self.text), doc_text)]
        begin = starts[self.occ]
        return begin, begin + len(self.text)


@dataclass
class Ann:
    mentions: List[M] = field(default_factory=list)
    relations: List[Tuple[int, int]] = field(default_factory=list)       # indices into mentions
    no_human: List[Tuple[int, int]] = field(default_factory=list)
    state: str = "FINISHED"


def sentences(text: str) -> List[Tuple[int, int]]:
    spans, start = [], 0
    for m in re.finditer(r"[.!?](\s+|$)", text):
        spans.append((start, m.start() + 1))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def make_cas(ts: TypeSystem, text: str, ann: Ann) -> Cas:
    cas = Cas(typesystem=ts)
    cas.sofa_string = text
    Sentence, Concept = ts.get_type(SENTENCE), ts.get_type("gemtex.Concept")
    Relation, NoHuman = ts.get_type("gemtex.Relation"), ts.get_type("webanno.custom.No_Human")
    for b, e in sentences(text):
        cas.add(Sentence(begin=b, end=e))
    created = []
    for m in ann.mentions:
        b, e = m.resolve(text)
        code = None if m.code is None else (m.code if m.code.startswith("http") or not m.code.isdigit()
                                            else IRI + m.code)
        c = Concept(begin=b, end=e, id=code, literal=m.literal)
        cas.add(c)
        created.append(c)
    for gi, di in ann.relations:
        gov, dep = created[gi], created[di]
        cas.add(Relation(begin=dep.begin, end=dep.end, Governor=gov, Dependent=dep))
    for b, e in ann.no_human:
        cas.add(NoHuman(begin=b, end=e))
    return cas


def build_export(path: Path, docs: Dict[str, str], users: Sequence[Tuple[str, str]],
                 annotations: Dict[Tuple[str, str], Ann], name: str = "mock",
                 initial: Optional[Dict[str, Ann]] = None) -> Path:
    """Write a zipped export: exportedproject.json, source/, annotation/<doc>/<user>.zip (+ INITIAL_CAS, the
    pre-annotation from ``initial``, empty by default)."""
    ts = typesystem()
    ts_xml = ts.to_xml()
    project = {
        "name": name, "application_name": "INCEpTION", "application_version": "41.5",
        "layers": [
            {"name": "gemtex.Concept", "type": "span", "overlap_mode": "ANY_OVERLAP",
             "features": [{"name": "id"}, {"name": "literal"}]},
            {"name": "gemtex.Relation", "type": "relation", "overlap_mode": "ANY_OVERLAP",
             "features": [{"name": "kind"}, {"name": "Governor"}, {"name": "Dependent"}]},
            {"name": "webanno.custom.No_Human", "type": "span", "features": []},
        ],
        "project_users": [{"username": u, "uiName": ui} for u, ui in users],
        "source_documents": [{"name": d, "format": "xmi", "state": "ANNOTATION_FINISHED"} for d in docs],
        "annotation_documents": [{"name": d, "user": u, "state": a.state} for (d, u), a in annotations.items()],
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("exportedproject.json", json.dumps(project))
        for doc, text in docs.items():
            zf.writestr(f"source/{doc}", text)
            per_doc = {"INITIAL_CAS": (initial or {}).get(doc, Ann())}
            per_doc.update({u: a for (d, u), a in annotations.items() if d == doc})
            for user, ann in per_doc.items():
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as inner:
                    inner.writestr(f"{user}.xmi", make_cas(ts, text, ann).to_xmi())
                    inner.writestr("TypeSystem.xml", ts_xml)
                zf.writestr(f"annotation/{doc}/{user}.zip", buffer.getvalue())
    return path
