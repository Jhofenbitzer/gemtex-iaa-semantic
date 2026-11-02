from collections import Counter

import pytest

from gemtex_iaa_semantic.io.loader import get_project, load_project, value_kind


def test_value_kind():
    assert value_kind("http://snomed.info/id/1", None) == "code"
    assert value_kind("http://snomed.info/id/410672004", "2021-12-xx") == "code_with_literal"
    assert value_kind(None, "3.0") == "number"
    assert value_kind("", "1999-01-xx") == "date"
    assert value_kind(None, None) == "empty"


def test_loader_filters_states_and_maps_sites(cfg):
    data = load_project(cfg)
    m = data.mentions
    assert set(m["user"]) == {"u1", "u2", "u3", "u5", "u6"}      # u4 IN_PROGRESS, INITIAL_CAS skipped
    assert data.meta["skipped"] == {"state": 1}                   # u4 doc1
    assert dict(zip(data.annotators["user"], data.annotators["site"])) == {"u1": "city0", "u2": "city1",
                                                                           "u3": "city1", "u5": "city2",
                                                                           "u6": "city3"}
    docs = data.docs.set_index("doc")
    assert list(docs.at["doc2.txt", "users"]) == ["u1", "u2", "u3", "u5", "u6"]   # u3 finished but empty
    assert docs.at["doc1.txt", "n_annotators"] == 3
    assert Counter(m["user"]) == {"u1": 14, "u2": 17, "u3": 7, "u5": 2, "u6": 2}
    assert len(data.relations) == 3 and set(data.relations["user"]) == {"u2"}
    assert len(data.no_human) == 1
    fever = m[(m["user"] == "u1") & (m["text"] == "Fieber") & (m["doc"] == "doc1.txt")].iloc[0]
    assert fever["sentence"] == 1 and fever["id_raw"].startswith("http://snomed.info/id/")
    assert (m["value_kind"] == "number").sum() == 2


def test_loader_reads_the_preannotation_separately(cfg):
    data = load_project(cfg)
    pre = data.preannotation
    assert list(pre["text"]) == ["Kopf", "verbrannt"] and set(pre["doc"]) == {"doc2.txt"}
    assert set(pre["user"]) == {"INITIAL_CAS"} and set(pre["site"]) == {"preannotation"}
    assert pre["mention_id"].min() == len(data.mentions)          # annotators' mention ids are unchanged
    assert "INITIAL_CAS" not in set(data.mentions["user"])
    assert data.meta["preannotation_relations"] == 0


def test_cache_roundtrip(cfg):
    first = get_project(cfg)
    second = get_project(cfg)
    assert first.meta["cache_key"] == second.meta["cache_key"]
    assert first.mentions.equals(second.mentions)
    assert first.preannotation.equals(second.preannotation)


# ---------------- several exports ----------------

TEXT_X = "Patient hat Fieber. Husten seit gestern."


def two_exports(directory):
    """batch-1 and batch-2 share x.txt (same text) and z.txt (different texts); y.txt is still in progress for
    ua2, w.txt has one annotator; ua1 annotated x.txt in both exports."""
    from mini_snomed import ID
    from mock_export import Ann, M, build_export
    build_export(directory / "batch-1.zip",
                 {"x.txt": TEXT_X, "y.txt": "Husten.", "z.txt": "Kopf ist verbrannt.", "w.txt": "Fieber."},
                 [("ua1", "city9_anno1"), ("ua2", "city9_anno2")],
                 {("x.txt", "ua1"): Ann([M("Fieber", ID["FEVER"]), M("Husten", ID["COUGH"])], relations=[(0, 1)]),
                  ("x.txt", "ua2"): Ann([M("Fieber", ID["FEVER"])]),
                  ("y.txt", "ua1"): Ann([M("Husten", ID["COUGH"])]),
                  ("y.txt", "ua2"): Ann([M("Husten", ID["COUGH"])], state="IN_PROGRESS"),
                  ("z.txt", "ua1"): Ann([M("verbrannt", ID["BURN"])]),
                  ("w.txt", "ua1"): Ann([M("Fieber", ID["FEVER"])])},
                 name="batch 1", initial={"x.txt": Ann([M("Fieber", ID["FEVER"])])})
    build_export(directory / "batch-2.zip", {"x.txt": TEXT_X, "z.txt": "Hand ist verbrannt."},
                 [("ua1", "city9_anno1"), ("ua3", "city9_anno3")],
                 {("x.txt", "ua3"): Ann([M("Husten", ID["COUGH"]), M("Fieber", ID["FEVER"])], relations=[(0, 1)]),
                  ("x.txt", "ua1"): Ann([M("Fieber", ID["AFEVER"])]),       # already loaded from batch-1
                  ("z.txt", "ua3"): Ann([M("verbrannt", ID["BURN"])])},
                 name="batch 2", initial={"x.txt": Ann([M("Husten", ID["COUGH"])])})
    return directory


@pytest.fixture
def exports_cfg(cfg, tmp_path):
    import dataclasses
    two_exports(tmp_path)
    return dataclasses.replace(cfg, PATH_PROJECT=(tmp_path,), DIR_CACHE=tmp_path / "cache")


def test_project_paths(exports_cfg, tmp_path):
    import dataclasses
    from gemtex_iaa_semantic.io.loader import project_paths
    expected = [tmp_path / "batch-1.zip", tmp_path / "batch-2.zip"]
    assert project_paths(exports_cfg) == expected                                    # a directory
    for entries in [(tmp_path / "batch-*.zip",), (tmp_path / "batch-1.zip", tmp_path)]:
        assert project_paths(dataclasses.replace(exports_cfg, PATH_PROJECT=entries)) == expected
    for entries in [(), (tmp_path / "missing.zip",), (tmp_path / "none-*.zip",)]:
        with pytest.raises(FileNotFoundError):
            project_paths(dataclasses.replace(exports_cfg, PATH_PROJECT=entries))


def test_several_exports_are_combined(exports_cfg):
    data = load_project(exports_cfg)
    assert data.meta["exports"] == 2 and data.meta["documents_in_exports"] == 4
    assert data.meta["document_status"] == {"analysable": 2, "pending": 1, "too_few": 1}     # x, z / y / w
    # z.txt has different texts: two documents with one annotator each, dropped
    assert data.meta["documents_dropped_too_few"] == 2
    assert data.meta["skipped"]["duplicate"] == 1                                    # ua1 x.txt from batch-2
    assert data.docs["doc"].tolist() == ["x.txt"] and data.docs["users"].tolist() == [["ua1", "ua2", "ua3"]]
    m = data.mentions
    assert Counter(m["user"]) == {"ua1": 2, "ua2": 1, "ua3": 2}
    assert m["mention_id"].is_unique and set(m["doc"]) == {"x.txt"}
    by_id = m.set_index("mention_id")
    for r in data.relations.itertuples(index=False):
        assert by_id.loc[r.governor, "user"] == by_id.loc[r.dependent, "user"] == r.user
    assert sorted(data.relations["user"]) == ["ua1", "ua3"]
    assert data.preannotation["text"].tolist() == ["Fieber"]                         # from batch-1 only
    assert data.preannotation["mention_id"].min() > m["mention_id"].max()
    assert dict(zip(data.annotators["user"], data.annotators["n_docs"])) == {"ua1": 1, "ua2": 1, "ua3": 1}


def test_min_annotators_and_site(exports_cfg):
    import dataclasses
    data = load_project(dataclasses.replace(exports_cfg, MIN_ANNOTATORS=0, SITE="site9"))
    assert sorted(data.docs["doc"]) == ["batch-1/z.txt", "batch-2/z.txt", "w.txt", "x.txt", "y.txt"]
    assert set(data.annotators["site"]) == {"site9"} and set(data.mentions["site"]) == {"site9"}
    assert data.meta["documents_dropped_too_few"] == 0
    assert load_project(dataclasses.replace(exports_cfg, MIN_ANNOTATORS=4)).docs.empty


def test_cache_of_several_exports(exports_cfg):
    from gemtex_iaa_semantic.io.loader import cache_dir
    first = get_project(exports_cfg)
    assert cache_dir(exports_cfg).name.startswith("2-exports-")
    assert get_project(exports_cfg).mentions.equals(first.mentions)
