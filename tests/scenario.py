"""A synthetic two-document project: split, nested and stacked spans, unresolved codes, literals, an empty
document, a copy and an untouched pre-annotation (fake SNOMED IDs from mini_snomed)."""
from pathlib import Path

from mini_snomed import ID
from mock_export import Ann, M, build_export

DOC1 = ("Befund: verletzter linker Hand. Patient hat Fieber. Keine Lähmung. Die Brandwunde heilt. "
        "Husten seit gestern. Gabe von 3 mg Amiodaron. Die Operation der Hand. HWI.")
DOC2 = "Kopf ist verbrannt. Fieber."
DOCS = {"doc1.txt": DOC1, "doc2.txt": DOC2}
USERS = [("u1", "city0_anno1"), ("u2", "city1_anno1"), ("u3", "city1_anno2"), ("u4", "city2_anno1"),
         ("u5", "city2_anno2"), ("u6", "city3_anno1")]

U1_DOC1 = Ann([
    M("verletzter linker Hand", ID["INJLH"]),   # (8) precoordinated
    M("Fieber", ID["FEVER"]),
    M("Keine Lähmung", ID["NOPARA"]),            # (63) negated precoordinated concept
    M("Brandwunde", ID["BURN"]),
    M("Husten", ID["COUGH"]),
    M("gestern", ID["OLD"]),                      # inactive -> unresolved
    M("3", literal="3"),                          # number in the literal field
    M("mg", ID["MG"]),
    M("Amiodaron", ID["AMIO_P"]),
    M("Operation der Hand", ID["HANDPROC"]),
    M("HWI", ID["FEVER"]), M("HWI", ID["COUGH"]),  # stacked (Maxim III)
])
U2_DOC1 = Ann([
    M("verletzter", ID["INJ"]), M("linker", ID["LEFT"]), M("Hand", ID["HAND"]),   # (10) split
    M("Fieber", ID["AFEVER"]),                   # more specific
    M("Keine", ID["ABSENT"]), M("Lähmung", ID["PARA"]),                          # negation qualifier
    M("Brandwunde", ID["BURN_M"]),               # morphology instead of disorder
    M("Husten", ID["FEVER"]),                    # conflict
    M("gestern", ID["OLD"]),
    M("3", literal="3.0"),
    M("mg", ID["MG"]),
    M("Amiodaron", ID["AMIO"]),                  # substance instead of product
    M("Die Operation der Hand", ID["HANDPROC"]),  # determiner included
    M("HWI", ID["FEVER"]),
], relations=[(0, 1), (0, 2), (4, 5)])
U3_DOC1 = Ann([
    M("verletzter linker Hand", ID["INJH"]), M("linker", ID["LEFT"]),               # (11) nested
    M("Fieber", ID["BTEMP"]),                    # observable instead of finding
    M("Keine Lähmung", ID["NOPARA"]),
    M("Husten", ID["COUGH"]),
    M("gestern", "fever"),                       # malformed id
    M("Operation", ID["HANDPROC"]),              # shorter span
], no_human=[(DOC1.index("Die Brandwunde"), DOC1.index("Die Brandwunde") + len("Die Brandwunde heilt."))])
U4_DOC1 = Ann([M("Fieber", ID["FEVER"])], state="IN_PROGRESS")

ANNOTATIONS = {
    ("doc1.txt", "u1"): U1_DOC1,
    ("doc1.txt", "u2"): U2_DOC1,
    ("doc1.txt", "u3"): U3_DOC1,
    ("doc1.txt", "u4"): U4_DOC1,
    ("doc2.txt", "u1"): Ann([M("verbrannt", ID["BURN"]), M("Fieber", ID["FEVER"])]),
    ("doc2.txt", "u2"): Ann([M("Kopf", ID["HAND"]), M("verbrannt", ID["BURN"]), M("Fieber", ID["FEVER"])]),
    ("doc2.txt", "u3"): Ann([]),                 # finished but empty
    # excluded before the analysis: a copy of u1 and an untouched pre-annotation
    ("doc2.txt", "u5"): Ann([M("verbrannt", ID["BURN"]), M("Fieber", ID["FEVER"])]),
    ("doc2.txt", "u6"): Ann([M("Kopf", ID["HAND"]), M("verbrannt", ID["BURN"])]),
}

# pre-annotation (INITIAL_CAS): u1 keeps "verbrannt", deletes "Kopf" and adds "Fieber"; u2 keeps both and adds
# "Fieber"; u3 deletes everything; u6 leaves it untouched
INITIAL = {"doc2.txt": Ann([M("Kopf", ID["HAND"]), M("verbrannt", ID["BURN"])])}


def write_scenario(directory: Path) -> Path:
    return build_export(Path(directory) / "project-mock.zip", DOCS, USERS, ANNOTATIONS, initial=INITIAL)
