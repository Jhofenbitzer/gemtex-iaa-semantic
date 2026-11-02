"""SNOMED CT: the HDF5 file distributed in the GeMTeX consortium, and the comparable key of each mention.

Only the concept codes and whether they are active are read. An annotated ``id`` resolves to an active concept of
the release; everything else (empty, malformed, unknown, inactive) is unresolved.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd

SNOMED_IRI_PREFIX = "http://snomed.info/id/"
# a bare SCTID, optionally followed by "| term" (as in the guideline notation "386661006 | Fever (finding)")
SNOMED_ID_RE = re.compile(r"(\d{6,18})(?:\s*\|.*)?")

# resolution status of an annotated code
OK, EMPTY, MALFORMED, UNKNOWN, INACTIVE = "ok", "empty", "malformed", "unknown", "inactive"
# value kinds (io.loader.value_kind) whose value is the literal: numbers and dates (guideline 3.3.3.1/3.3.3.2)
LITERAL_KINDS = ("number", "date", "literal_other")


def parse_sctid(value) -> Optional[str]:
    """Extract the SCTID from an annotation value such as ``http://snomed.info/id/22298006``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    if text.startswith(SNOMED_IRI_PREFIX):
        text = text[len(SNOMED_IRI_PREFIX):]
    match = SNOMED_ID_RE.fullmatch(text)
    return match.group(1) if match else None


@dataclass(frozen=True)
class Resolution:
    status: str
    index: int = -1
    sctid: Optional[str] = None


class SnomedStore:
    """The concept codes of the HDF5 file and whether they are active; concepts are identified by row index."""

    def __init__(self, path):
        self.path = Path(path)
        with h5py.File(self.path, "r") as fi:
            grp = fi["concepts"]
            self.release_date = str(grp.attrs.get("release_date", ""))
            self.codes: List[str] = list(grp["codes"].asstr()[:])
            self.active = grp["active"][:].astype(bool)
        self.code_to_index = {code: i for i, code in enumerate(self.codes)}
        self.n_active = int(self.active.sum())

    def __len__(self) -> int:
        return len(self.codes)

    def resolve(self, value) -> Resolution:
        """Map an annotated ``id`` value to an active concept; everything else is unresolved."""
        if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
            return Resolution(EMPTY)
        sctid = parse_sctid(value)
        if sctid is None:
            return Resolution(MALFORMED)
        idx = self.code_to_index.get(sctid)
        if idx is None:
            return Resolution(UNKNOWN, sctid=sctid)
        if not self.active[idx]:
            return Resolution(INACTIVE, idx, sctid)
        return Resolution(OK, idx, sctid)


def normalise_literal(value_kind: str, literal) -> Optional[str]:
    lit = (literal or "").strip()
    if not lit:
        return None
    if value_kind == "number" or re.match(r"^[+-]?\d+(?:[.,]\d+)?$", lit):
        try:
            return repr(float(lit.replace(",", ".")))
        except ValueError:
            return lit
    return lit.lower()


def concept_key(row) -> str:
    """Comparable label of one mention: concept (+ literal), literal value, or the unresolved code."""
    lit = row.literal_norm
    if row.status == OK:
        return f"C{int(row.concept)}" + (f"={lit}" if lit else "")
    if row.value_kind in LITERAL_KINDS:
        return f"L:{lit}"
    return f"U:{row.sctid or row.id_raw}"


def enrich_mentions(mentions: pd.DataFrame, store: SnomedStore) -> pd.DataFrame:
    """Add the resolution (``status``, ``concept`` row index or -1, ``sctid``), the normalised literal, the concept
    ``key`` and ``unresolved``: neither an active concept nor a literal value (number or date)."""
    out = mentions.copy()
    resolved = [store.resolve(v) for v in out["id_raw"]]
    out["status"] = [r.status for r in resolved]
    out["concept"] = np.asarray([r.index if r.status == OK else -1 for r in resolved], dtype=np.int64)
    out["sctid"] = [r.sctid for r in resolved]
    out["literal_norm"] = [normalise_literal(k, lit) for k, lit in zip(out["value_kind"], out["literal"])]
    out["unresolved"] = ~((out["status"] == OK) | ((out["status"] == EMPTY) & out["value_kind"].isin(LITERAL_KINDS)))
    out["key"] = [concept_key(r) for r in out.itertuples(index=False)]
    return out
