import pandas as pd
import pytest

from gemtex_iaa_semantic import selection as SEL
from gemtex_iaa_semantic.io.loader import ProjectData

PRE = [(0, 5, "C1"), (10, 15, "C2")]
EDITED = [(0, 5, "C1"), (10, 15, "C9")]
# document -> user -> concepts; d is pre-annotated, x and y are not
DOCS = {
    "d": {"u": PRE, "u2": PRE, "a": EDITED, "b": EDITED, "c": EDITED, "e": [], "f": [],
          "l": PRE + [(20, 25, "C3")]},                     # l: low change (F1 0.8)
    "x": {"a": [(0, 3, "K")], "b": [(0, 3, "K")]},
    "y": {"b": [(0, 3, "Z")]},
}
SITES = {"u": "s1", "u2": "s2", "a": "s1", "b": "s1", "c": "s2", "e": "s1", "f": "s2", "l": "s1"}


def mention_frame(docs, first_id=0):
    rows = [(doc, user, b, e, k) for doc, users in docs.items() for user, concepts in users.items()
            for b, e, k in concepts]
    df = pd.DataFrame(rows, columns=["doc", "user", "begin", "end", "key"])
    df.insert(0, "mention_id", range(first_id, first_id + len(df)))
    return df


@pytest.fixture
def inputs():
    mentions = mention_frame(DOCS)
    pre = mention_frame({"d": {"INITIAL_CAS": PRE}}, 1000)
    b_d = mentions[(mentions["doc"] == "d") & (mentions["user"] == "b")]["mention_id"].tolist()
    relations = pd.DataFrame([("d", "b", b_d[0], b_d[1], None)], columns=["doc", "user", "governor", "dependent",
                                                                        "kind"])
    doc_users = {doc: sorted(users) for doc, users in DOCS.items()}
    names = pd.DataFrame({"ui_name": {u: f"{s}_{u}" for u, s in SITES.items()}, "site": SITES})
    return mentions, pre, relations, doc_users, names


def test_untouched_and_copies(inputs):
    sel = SEL.select_documents(*inputs)
    ex = sel.excluded.set_index(["doc", "user"])
    assert ex["reason"].to_dict() == {("d", "u"): "untouched", ("d", "u2"): "untouched",     # not copies
                                      ("d", "a"): "copy", ("d", "c"): "copy", ("x", "a"): "copy"}
    # b has the smallest share of copied documents (2 of 3), a and c have 1 of 1
    assert set(ex.loc[ex["reason"] == "copy", "copy_of"]) == {"s1_b"}
    assert not ex.loc[("d", "a"), "relations_identical"]               # b has a relation, a has none
    assert ex.loc[("x", "a"), "relations_identical"]
    assert ex.loc[("d", "u"), "f1_preannotation"] == 1 and ex.loc[("d", "a"), "n_mentions"] == 2
    assert ("d", "e") not in ex.index and ("d", "f") not in ex.index    # empty documents are not copies
    assert sel.low_change.empty                                         # l: F1 0.8 < 1 - 0.05
    assert sel.removed_annotators == ["s1_a", "s1_u", "s2_c", "s2_u2"]
    assert sel.describe() == "2 untouched, 3 copy"


def test_switches(inputs):
    listed = SEL.select_documents(*inputs, low_change=0.25)
    assert listed.low_change[["user", "excluded_as"]].values.tolist() == [["l", ""]]      # listed, analysed
    assert listed.low_change["f1_preannotation"].tolist() == [pytest.approx(0.8)]
    assert "l" not in set(listed.excluded["user"])
    sel = SEL.select_documents(*inputs, low_change=0.25, exclude_low_change=True)
    assert sel.excluded.set_index("user").loc["l", "reason"] == "low_change"
    assert sel.low_change["excluded_as"].tolist() == ["low_change"]
    off = SEL.select_documents(*inputs, exclude_untouched=False, exclude_copies=False)
    assert off.excluded.empty and off.pairs == frozenset() and off.describe() == "none"
    assert off.low_change["user"].tolist() == []
    # without a pre-annotation, untouched documents cannot be recognised; copies still are
    mentions, pre, relations, doc_users, names = inputs
    no_pre = SEL.select_documents(mentions, pre.iloc[0:0], relations, doc_users, names)
    assert set(no_pre.excluded["reason"]) == {"copy"}
    assert ("d", "u2") in no_pre.pairs and ("d", "u") not in no_pre.pairs   # now copies of each other: tie


def test_tie_break_by_display_name():
    mentions = mention_frame({"z": {"p": [(0, 1, "K")], "q": [(0, 1, "K")]}})
    names = pd.DataFrame({"ui_name": {"p": "site_b", "q": "site_a"}, "site": {"p": "site", "q": "site"}})
    sel = SEL.select_documents(mentions, mentions.iloc[0:0], mentions.iloc[0:0].assign(governor=[], dependent=[],
                                                                                         kind=[]),
                               {"z": ["p", "q"]}, names)
    assert sel.excluded[["user", "copy_of"]].values.tolist() == [["p", "site_a"]]


def test_apply_and_summary(inputs):
    mentions, pre, relations, doc_users, names = inputs
    sel = SEL.select_documents(*inputs)
    data = ProjectData(
        docs=pd.DataFrame({"doc": list(doc_users), "text": "", "users": list(doc_users.values()),
                           "n_annotators": [len(u) for u in doc_users.values()]}),
        annotators=names.rename_axis("user").reset_index().assign(annotator="", n_docs=1),
        mentions=mentions, relations=relations,
        no_human=pd.DataFrame([("d", "a", 0, 1), ("d", "b", 0, 1)], columns=["doc", "user", "begin", "end"]))
    kept = SEL.apply(data, sel.pairs)
    assert set(kept.mentions["user"]) == {"b", "l"} and len(kept.relations) == 1
    assert kept.no_human["user"].tolist() == ["b"]
    assert dict(zip(kept.docs["doc"], kept.docs["users"])) == {"d": ["b", "e", "f", "l"], "x": ["b"], "y": ["b"]}
    assert set(kept.annotators["user"]) == {"b", "e", "f", "l"}
    summary = SEL.summary(sel, kept)
    assert summary.loc["loaded"].tolist() == [11, len(mentions)]
    assert summary.loc["excluded: untouched"].tolist() == [2, 4]
    assert summary.loc["excluded: copy"].tolist() == [3, 5]
    assert summary.loc["analysed", "annotation documents"] == 6
    assert SEL.by_annotator(sel).loc["s1_a"].tolist() == [0, 0, 2]
    assert SEL.apply(data, frozenset()) is data


def test_prepare_excludes_before_any_analysis(ctx):
    ex = ctx.selection.excluded.set_index("ui_name")
    assert ex.loc["city3_anno1", "reason"] == "untouched"
    assert ex.loc["city2_anno2", ["reason", "copy_of"]].tolist() == ["copy", "city0_anno1"]
    assert not {"u5", "u6"} & set(ctx.mentions["user"]) and set(ctx.data.annotators["user"]) == {"u1", "u2", "u3"}
    assert list(ctx.data.docs.set_index("doc").at["doc2.txt", "users"]) == ["u1", "u2", "u3"]
    assert ctx.selection.removed_annotators == ["city2_anno2", "city3_anno1"]


def test_require_annotators(ctx):
    """doc2.txt keeps 3 of its 5 annotators after the exclusions; requiring 4 drops it."""
    sel = SEL.Selection(ctx.selection.excluded, ctx.selection.low_change, removed_annotators=["x"])
    kept = SEL.require_annotators(ctx.data, 4, sel)
    assert kept.docs["doc"].tolist() == [] and kept.mentions.empty and kept.annotators.empty
    assert sel.too_few["doc"].tolist() == ["doc1.txt", "doc2.txt"]
    assert sel.too_few["annotators"].tolist() == [3, 3]
    assert sel.too_few["mentions"].sum() == len(ctx.data.mentions)
    assert sel.removed_annotators == ["city0_anno1", "city1_anno1", "city1_anno2", "x"]
    assert sel.settings["MIN_ANNOTATORS"] == 4
    same = SEL.Selection(ctx.selection.excluded, ctx.selection.low_change)
    assert SEL.require_annotators(ctx.data, 2, same) is ctx.data and same.too_few.empty
