import krippendorff
import numpy as np
import pandas as pd
import pytest

from gemtex_iaa_semantic import align as A
from gemtex_iaa_semantic import measures as M


def mentions(rows, unresolved=()):
    """(user, begin, end, key) rows of document d."""
    df = pd.DataFrame(rows, columns=["user", "begin", "end", "key"]).assign(doc="d")
    df.insert(0, "mention_id", range(len(df)))
    return df.assign(unresolved=df["key"].isin(unresolved), concept=-1, status="ok")


def test_span_counts_and_is_exact():
    m = mentions([("a", 0, 10, "C1"), ("a", 0, 10, "C2"), ("a", 20, 25, "C3"),
                  ("b", 0, 10, "C1"), ("b", 18, 25, "C3"), ("b", 30, 35, "C4")])
    regions = A.all_pair_regions(m)
    counts = pd.DataFrame([M.span_counts(r) for r in regions]).sum()
    # spans a: (0,10) (stacked, once), (20,25); b: (0,10), (18,25), (30,35)
    assert counts[["n_a", "n_b", "strict", "lenient"]].tolist() == [2, 3, 1, 2]
    assert counts[["e2e_a", "e2e_b", "e2e"]].tolist() == [3, 3, 1]
    same = mentions([("a", 0, 10, "C1"), ("b", 0, 10, "C1")])
    assert M.is_exact(next(A.all_pair_regions(same)))
    unresolved = mentions([("a", 0, 10, "U:1"), ("b", 0, 10, "U:1")], unresolved=["U:1"])
    assert not M.is_exact(next(A.all_pair_regions(unresolved)))


def test_kappa_terms_give_cohens_kappa():
    pairs = pd.DataFrame({"doc": "d", "user_a": "a", "user_b": "b", "key_a": ["A", "A", "B", "C"],
                          "key_b": ["A", "B", "B", "C"]})
    k = M.kappa_terms(pairs.assign(same_key=pairs["key_a"] == pairs["key_b"])).iloc[0]
    # p_o = 3/4, p_e = 1/2·1/4 + 1/4·1/2 + 1/4·1/4 = 5/16
    assert k["num"] / k["den"] == pytest.approx((3 / 4 - 5 / 16) / (1 - 5 / 16))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_alpha_terms_add_up_to_the_alpha_of_each_document(seed):
    rng = np.random.default_rng(seed)
    units = pd.DataFrame([{"doc": f"d{d}", "cluster": 100 * d + c, "user": f"u{u}", "key": f"C{rng.integers(0, 3)}"}
                          for d in range(3) for c in range(20)
                          for u in rng.choice(5, size=rng.integers(1, 5), replace=False)])
    terms = M.alpha_terms(units)
    assert (terms["user_a"] < terms["user_b"]).all()
    per_doc = terms.groupby("doc")[["n", "num", "den"]].sum()
    for doc, g in units.groupby("doc"):
        matrix = g.assign(value=g["key"].str[1:].astype(float)).pivot(index="user", columns="cluster", values="value")
        expected = krippendorff.alpha(reliability_data=matrix.to_numpy(), level_of_measurement="nominal")
        assert per_doc.at[doc, "num"] / per_doc.at[doc, "den"] == pytest.approx(expected)
        sizes = g.groupby("cluster").size()
        assert per_doc.at[doc, "n"] == pytest.approx(sizes[sizes >= 2].sum())


def test_alpha_units_drop_clusters_with_several_mentions_of_one_annotator():
    m = mentions([("a", 0, 10, "C1"), ("a", 5, 12, "C2"), ("b", 0, 10, "C1"), ("a", 20, 25, "C3"),
                  ("b", 20, 25, "C3")])
    units = M.alpha_units(A.clusters(m))
    assert units["key"].tolist() == ["C3", "C3"]


def test_stacks_and_left_out_clusters():
    m = mentions([("a", 0, 10, "C1"), ("a", 0, 10, "C2"), ("b", 0, 10, "C1"), ("c", 0, 10, "C1"),   # a stacks
                  ("a", 20, 30, "C3"), ("b", 20, 24, "C3"), ("b", 25, 30, "C4"),                    # b splits
                  ("a", 40, 45, "C5"), ("a", 40, 45, "C6"), ("a", 42, 50, "C7"),                    # a alone
                  ("a", 60, 65, "C8"), ("b", 60, 65, "C8")])                                        # kept
    regions = {(r.user_a, r.user_b, r.begin): r for r in A.all_pair_regions(m)}
    assert M.stacked(regions[("a", "b", 0)]) and not M.stacked(regions[("a", "b", 20)])
    out = M.alpha_left_out(A.clusters(m)).set_index(["user_a", "user_b"])[["stack", "spans"]]
    # the stack: 3 annotators, 2 / (3 - 1) per pair; the split: 2 annotators, 2 for the pair
    assert out.loc[("a", "b")].tolist() == [1, 2] and out.loc[("a", "c")].tolist() == [1, 0]
    assert out.loc[("b", "c")].tolist() == [1, 0] and out[["stack", "spans"]].to_numpy().sum() == 3 + 2


def test_bootstrap_ratio():
    df = pd.DataFrame({"doc": ["x", "x", "y", "z"], "num": [1, 2, 3, 0], "den": [2, 2, 4, 2]})
    point, lo, hi = M.bootstrap_ratio(df, "num", "den", n_boot=200)
    assert point == pytest.approx(6 / 10) and lo <= point <= hi
    assert np.isnan(M.bootstrap_ratio(df, "num", "den", n_boot=0)[1:]).all()
