"""Reading CAS objects from a zipped INCEpTION XMI project export.

Adapted from INCEpTALYTICS (``inceptalytics.utils.annotation_info_from_xmi_zip``, https://github.com/catalpa-cl/inceptalytics)
as used in the INCEpTALYTICS4GeMTeX baseline, so that this package does not depend on the baseline code.
"""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Collection, List, Optional, Tuple
from zipfile import ZipFile

import cassis

SENTENCE_TYPE_NAME = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
ANNOTATION_ZIP_RE = re.compile(r".*(annotation|curation)/.*/(?!\._).*zip$")


def annotation_info_from_xmi_zip(project_fp, documents: Optional[Collection[str]] = None
                                 ) -> List[Tuple[cassis.Cas, str, str]]:
    """Return ``(CAS, source file name, annotator)`` for every annotation/curation CAS of the export (only of
    ``documents`` if given; the others are not parsed).

    The annotator is the stem of the XMI file inside ``annotation/<document>/<user>.zip`` (the INCEpTION username;
    ``INITIAL_CAS`` and ``CURATION_USER`` included).
    """
    annotations = []
    with ZipFile(project_fp) as project_zip:
        typesystem = None
        files = [fp for fp in project_zip.namelist() if ANNOTATION_ZIP_RE.match(fp)]
        if not files:
            raise RuntimeError("Could not parse project or empty project.")
        for file_path in files:
            if documents is not None and Path(file_path).parent.name not in documents:
                continue
            with ZipFile(BytesIO(project_zip.read(file_path))) as annotation_zip:
                if typesystem is None:
                    typesystem = cassis.load_typesystem(BytesIO(annotation_zip.read("TypeSystem.xml")))
                cas_file = next(f for f in annotation_zip.namelist() if f.endswith(".xmi"))
                cas = cassis.load_cas_from_xmi(BytesIO(annotation_zip.read(cas_file)), typesystem)
                annotations.append((cas, Path(file_path).parent.name, Path(cas_file).stem))
    return annotations
