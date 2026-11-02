"""A tiny fake SNOMED CT HDF5 file for tests: only what the package reads (concept codes and whether they are
active), with fake concept IDs (90000xx) and no descriptions."""
from pathlib import Path

import h5py
import numpy as np

RELEASE = "20240401"

# name -> (id, active)
CONCEPTS = {
    "ROOT": ("9000001", True), "CF": ("9000010", True), "DIS": ("9000011", True), "INJ": ("9000012", True),
    "INJH": ("9000013", True), "INJLH": ("9000014", True), "BURN": ("9000015", True), "FEVER": ("9000016", True),
    "AFEVER": ("9000017", True), "PARA": ("9000018", True), "NOPARA": ("9000019", True),
    "COUGH": ("9000021", True), "BURNLH": ("9000025", True), "BTEMP": ("9000031", True), "HAND": ("9000042", True),
    "BURN_M": ("9000046", True), "LEFT": ("9000052", True), "ABSENT": ("9000055", True), "MG": ("9000057", True),
    "HANDPROC": ("9000061", True), "AMIO": ("9000071", True), "AMIO_P": ("9000073", True),
    "OLD": ("9000099", False),
}
ID = {name: sctid for name, (sctid, _) in CONCEPTS.items()}


def write_mini_hdf5(path: Path) -> Path:
    with h5py.File(path, "w") as fo:
        grp = fo.create_group("concepts")
        grp.attrs["release_date"] = RELEASE
        grp.create_dataset("codes", data=[sctid for sctid, _ in CONCEPTS.values()], dtype=h5py.string_dtype())
        grp.create_dataset("active", data=np.asarray([active for _, active in CONCEPTS.values()], dtype=bool))
    return Path(path)
