import pandas as pd
import pytest

from gemtex_iaa_semantic import align as A
from gemtex_iaa_semantic import preannotation as PA

PRE = [(0, 5, "C1"), (10, 15, "C2"), (20, 25, "C3"), (30, 35, "C4")]
# keeps C1, relabels C2, reshapes C3, deletes C4 and adds C7
EDITED = [(0, 5, "C1"), (10, 15, "C9"), (18, 25, "C3"), (40, 45, "C7")]


def frame(user, rows, first_id=0, doc="d"):
    return pd.DataFrame([{"mention_id": first_id + i, "doc": doc, "user": user, "begin": b, "end": e, "key": k}
                         for i, (b, e, k) in enumerate(rows)], columns=["mention_id", "doc", "user", "begin", "end",
                                                                         "key"])


@pytest.fixture
def data():
    """Annotator a leaves the pre-annotation untouched, b edits it, c is an exact copy of b."""
    pre = frame("INITIAL_CAS", PRE, 100)
    mentions = pd.concat([frame("a", PRE, 0), frame("b", EDITED, 10), frame("c", EDITED, 20)], ignore_index=True)
    return pre, mentions, {"d": ["a", "b", "c"]}


def test_decisions_and_documents(data):
    pre, mentions, doc_users = data
    dec, docs = PA.compare(pre, mentions, {**doc_users, "d": ["a", "b", "c", "e"], "x": ["a"]})
    got = dec.pivot(index="mention_id", columns="user", values="decision")
    assert list(got["a"]) == ["kept"] * 4
    assert list(got["b"]) == ["kept", "relabelled", "reshaped", "deleted"]
    assert list(got["c"]) == list(got["b"])
    assert list(got["e"]) == ["deleted"] * 4                     # finished but empty
    assert set(docs["doc"]) == {"d"}                              # x has no pre-annotation
    d = docs.set_index("user")
    assert d.loc["a", ["n_pre", "n_annotator", "kept", "changed", "added"]].tolist() == [4, 4, 4, 0, 0]
    assert (d.loc["b", ["kept", "relabelled", "reshaped", "deleted", "changed", "added", "kept_span"]].tolist()
            == [1, 1, 1, 1, 2, 1, 2])
    assert d.loc["a", "f1"] == 1 and d.loc["e", "f1"] == 0
    assert d.loc["b", "f1"] == pytest.approx(2 * 1 / 8) and d.loc["b", "f1_span"] == pytest.approx(2 * 2 / 8)
    assert d["untouched"].tolist() == [True, False, False, False]
    assert d["low_change"].tolist() == [True, False, False, False]


def test_match_stacks_and_empty_sides():
    pre = frame("INITIAL_CAS", [(0, 5, "C1"), (0, 5, "C2")])
    keeps_one = frame("a", [(0, 5, "C1")])
    dec, cls = PA.match(pre, keeps_one)
    assert dec.tolist() == ["kept", "deleted"] and cls.tolist() == ["kept"]   # dropping a stacked concept
    assert PA.match(pre, frame("a", [(0, 5, "C3")]))[0].tolist() == ["relabelled", "relabelled"]
    assert PA.match(pre, frame("a", [(0, 5, "C2"), (0, 5, "C2")]))[1].tolist() == ["kept", "changed"]
    dec, cls = PA.match(pre, keeps_one.iloc[0:0])
    assert dec.tolist() == ["deleted", "deleted"] and cls.tolist() == []
    assert PA.match(pre.iloc[0:0], keeps_one)[1].tolist() == ["added"]


def test_region_pre_side(data):
    pre, mentions, doc_users = data
    docs = pd.DataFrame({"doc": ["d"], "users": [doc_users["d"]]})
    regions = list(A.all_pair_regions(mentions, docs))
    got = {}
    for region, side in zip(regions, PA.region_pre_side(regions, pre)):
        got.setdefault((region.user_a, region.user_b), {})[region.begin] = side
    edited = {0: "both", 10: "a", 18: "a", 30: "a", 40: ""}         # a left the pre-annotation untouched
    assert got[("a", "b")] == edited and got[("a", "c")] == edited
    assert got[("b", "c")] == {0: "both", 10: "neither", 18: "neither", 40: ""}
    assert set(PA.region_pre_side(regions, pre.iloc[0:0])) == {""}
