"""Configuration of a site (INI file read with configparser, see ``config/example.cfg``)."""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


def _list(value: Optional[str]) -> List[str]:
    """Comma separated list; tolerates the ``['a', 'b']`` notation."""
    if value is None:
        return []
    value = value.strip().strip("][")
    return [v.strip().strip("'\"") for v in value.replace("\n", " ").split(",") if v.strip().strip("'\"")]


def _paths(value: Optional[str]) -> List[str]:
    """Comma or line separated paths (spaces inside a path are kept)."""
    return [v.strip() for v in (value or "").replace("\n", ",").split(",") if v.strip()]


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Config:
    PATH_PROJECT: Tuple[Path, ...]     # export files, directories (all *.zip in them) or glob patterns
    DIR_OUTPUT: Path
    DIR_CACHE: Path
    VERBOSE: int

    CONCEPT_LAYER: str
    ID_FEATURE: str
    LITERAL_FEATURE: str
    RELATION_LAYER: str
    RELATION_FEATURE: str
    NO_HUMAN_LAYER: str
    STATES: Tuple[str, ...]
    SKIP_ANNOTATORS: Tuple[str, ...]
    SITE_PATTERN: str
    MIN_ANNOTATORS: int

    HDF5: Path

    BOOTSTRAP: int
    SEED: int
    PREANNOTATION: bool
    PREANNO_LOW_CHANGE: float
    EXCLUDE_UNTOUCHED: bool
    EXCLUDE_COPIES: bool
    EXCLUDE_LOW_CHANGE: bool
    SITE: str = ""                     # if set, the site of every annotator (instead of SITE_PATTERN)


def load_config(config_file) -> Config:
    """Read ``config_file``; options it does not know (e.g. of older versions) are ignored."""
    cp = configparser.ConfigParser()
    if not cp.read(config_file, encoding="utf-8"):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    d, a, s, x = (cp[k] if cp.has_section(k) else {} for k in ("default", "annotation", "snomed", "analysis"))
    return Config(
        PATH_PROJECT=tuple(Path(p).expanduser() for p in _paths(d.get("PATH_PROJECT"))),
        DIR_OUTPUT=Path(d.get("DIR_OUTPUT", "data/results")),
        DIR_CACHE=Path(d.get("DIR_CACHE", "data/cache")),
        VERBOSE=int(d.get("VERBOSE", 0)),
        SITE=d.get("SITE", "").strip(),
        CONCEPT_LAYER=a.get("CONCEPT_LAYER", "gemtex.Concept"),
        ID_FEATURE=a.get("ID_FEATURE", "id"),
        LITERAL_FEATURE=a.get("LITERAL_FEATURE", "literal"),
        RELATION_LAYER=a.get("RELATION_LAYER", "gemtex.Relation"),
        RELATION_FEATURE=a.get("RELATION_FEATURE", "kind"),
        NO_HUMAN_LAYER=a.get("NO_HUMAN_LAYER", "webanno.custom.No_Human"),
        STATES=tuple(_list(a.get("STATES", "FINISHED"))),
        SKIP_ANNOTATORS=tuple(_list(a.get("SKIP_ANNOTATORS", "[]"))),
        SITE_PATTERN=a.get("SITE_PATTERN", r"^(?P<site>[^_]+)_(?P<annotator>.+)$"),
        MIN_ANNOTATORS=int(a.get("MIN_ANNOTATORS", 2)),
        HDF5=Path(s.get("HDF5", "data/snomed/snograscco_snomedct_20240401.hdf5")),
        BOOTSTRAP=int(x.get("BOOTSTRAP", 1000)),
        SEED=int(x.get("SEED", 13)),
        PREANNOTATION=_bool(x.get("PREANNOTATION", "True")),
        PREANNO_LOW_CHANGE=float(x.get("PREANNO_LOW_CHANGE", 0.05)),
        EXCLUDE_UNTOUCHED=_bool(a.get("EXCLUDE_UNTOUCHED", "True")),
        EXCLUDE_COPIES=_bool(a.get("EXCLUDE_COPIES", "True")),
        EXCLUDE_LOW_CHANGE=_bool(a.get("EXCLUDE_LOW_CHANGE", "False")),
    )
