import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from mini_snomed import write_mini_hdf5  # noqa: E402

from gemtex_iaa_semantic.snomed import SnomedStore  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def site_config(directory: Path, scenario_zip: Path, mini_hdf5: Path, site: str, **settings) -> Path:
    """config/example.cfg for the mock export; configs in the same ``directory`` share DIR_OUTPUT (and thus the
    pseudonyms)."""
    text = (REPO / "config" / "example.cfg").read_text(encoding="utf-8")
    values = {"SITE": site, "PATH_PROJECT": scenario_zip, "DIR_OUTPUT": directory / "results",
              "DIR_CACHE": directory / "cache", "HDF5": mini_hdf5, "BOOTSTRAP": 50, **settings}
    for key, value in values.items():
        text, n = re.subn(rf"^{key} =.*$", f"{key} = {value}", text, flags=re.M)
        assert n == 1, key
    path = directory / f"{site or 'no-site'}.cfg"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def mini_hdf5(tmp_path_factory) -> Path:
    return write_mini_hdf5(tmp_path_factory.mktemp("snomed") / "mini.hdf5")


@pytest.fixture(scope="session")
def store(mini_hdf5) -> SnomedStore:
    return SnomedStore(mini_hdf5)


@pytest.fixture(scope="session")
def scenario_zip(tmp_path_factory) -> Path:
    from scenario import write_scenario
    return write_scenario(tmp_path_factory.mktemp("export"))


@pytest.fixture(scope="session")
def test_config(tmp_path_factory, scenario_zip, mini_hdf5) -> Path:
    """The example configuration for the mock export, without SITE (sites from the display names)."""
    return site_config(tmp_path_factory.mktemp("config"), scenario_zip, mini_hdf5, "")


@pytest.fixture(scope="session")
def cfg(test_config):
    from gemtex_iaa_semantic.config import load_config
    return load_config(test_config)


@pytest.fixture(scope="session")
def ctx(cfg, store):
    from gemtex_iaa_semantic.pipeline import prepare
    return prepare(cfg, store=store)
