import json
import re
import shutil

import krippendorff
import pandas as pd
import pytest

from conftest import site_config
from scenario import DOCS, USERS
from gemtex_iaa_semantic import consortium as C
from gemtex_iaa_semantic import measures as M
from gemtex_iaa_semantic.align import clusters
from gemtex_iaa_semantic.cli import main

ALPHA = "Krippendorff's alpha (nominal), chance per document (pooled)"


@pytest.fixture(scope="module")
def runs(tmp_path_factory, scenario_zip, mini_hdf5):
    """site-report of the mock export as two sites (same data)."""
    directory = tmp_path_factory.mktemp("consortium")
    out = []
    for site in ("site_a", "site_b"):
        assert main(["site-report", str(site_config(directory, scenario_zip, mini_hdf5, site))]) == 0
        out.append(sorted((directory / "results").glob("*_site-report"))[-1])
    return directory, out


def test_pair_counts_of_the_scenario(ctx):
    counts = C.pair_counts(ctx).set_index(["doc", "user_a", "user_b"])
    # doc2 after the exclusions (u5 copy, u6 untouched): u1 verbrannt + Fieber, u2 Kopf + verbrannt + Fieber,
    # u3 nothing; "verbrannt" is the pre-annotation kept by u1 and u2, "Fieber" was added by both
    row = counts.loc[("doc2.txt", "u1", "u2")]
    assert row[["n_a", "n_b", "strict", "lenient", "e2e", "kappa_n", "exact", "exact_inherited"]].tolist() == [
        2, 3, 2, 2, 2, 2, 2, 1]
    assert row["kappa_agree"] == 2 and row["kappa_expected"] == pytest.approx(1.0)       # p_e = 1/2, κ = 1
    assert counts.loc[("doc2.txt", "u2", "u3"), ["n_a", "n_b", "strict", "exact"]].tolist() == [3, 0, 0, 0]
    assert len(counts) == 6 and counts.loc["doc1.txt", "exact_inherited"].sum() == 0  # doc1: no pre-annotation


def test_alpha_is_krippendorffs_alpha_per_document(ctx):
    derived = C._derive(C.pair_counts(ctx)).groupby("doc")[["alpha_n", "alpha_num", "alpha_den"]].sum()
    units = M.alpha_units(clusters(ctx.mentions))
    for doc, g in units.groupby("doc"):
        codes = {k: i for i, k in enumerate(sorted(set(g["key"])))}
        matrix = g.assign(value=g["key"].map(codes).astype(float)).pivot(index="user", columns="cluster",
                                                                         values="value")
        expected = krippendorff.alpha(reliability_data=matrix.to_numpy(), level_of_measurement="nominal")
        assert derived.at[doc, "alpha_num"] / derived.at[doc, "alpha_den"] == pytest.approx(expected)
        sizes = g.groupby("cluster").size()
        assert derived.at[doc, "alpha_n"] == pytest.approx(sizes[sizes >= 2].sum())


def test_share_is_text_free_and_pseudonymised(runs):
    directory, (run, _) = runs
    share = run / "share"
    assert sorted(f.name for f in share.iterdir()) == sorted(C.SHARE_FILES)
    forbidden = [*DOCS, *[d.split(".")[0] for d in DOCS], *[ui for _, ui in USERS], "Fieber", "verbrannt", "Kopf",
                 "project-mock", "snomed", str(directory)]
    for path in share.iterdir():
        text = path.read_text(encoding="utf-8")
        for word in forbidden:
            assert word not in text, (path.name, word)
        assert not re.search(r"\bu[1-6]\b|city\d|anno\d", text), path.name    # usernames, display names
        assert not re.search(r"\b9000\d{3}\b", text), path.name                # SNOMED CT ids of the mock release
    counts = pd.read_csv(share / "pair_counts.csv")
    assert counts["doc"].str.fullmatch(r"D\d{4}").all()
    assert counts[["annotator_a", "annotator_b"]].stack().str.fullmatch(r"A\d{2}").all()
    assert (counts["annotator_a"] < counts["annotator_b"]).all()


def test_site_report_content(runs):
    directory, (run, second) = runs
    summary = json.loads((run / "share" / "summary.json").read_text(encoding="utf-8"))
    assert summary["site"] == "site_a" and summary["settings"]["pre-annotation"] is True
    data = summary["data"]
    assert data["documents_min_annotators_finished"] == 2 and data["documents_analysed"] == 2
    assert (data["excluded_untouched"], data["excluded_copy"]) == (1, 1)
    assert (data["annotators_analysed"], data["document_annotator_pairs"]) == (3, 6)
    assert [f["figure"] for f in summary["figures"]] == [f.name for f in C.FIGURES]
    report = (run / "share" / "report.md").read_text(encoding="utf-8")
    assert "# Agreement report: site_a" in report and "end-to-end F1" in report and "Krippendorff" in report
    annotators = pd.read_csv(run / "share" / "annotators.csv")
    assert len(annotators) == 5                                          # 3 analysed, 2 only excluded
    assert annotators[["untouched", "copies"]].sum().tolist() == [1, 1]
    # the local folder resolves the codes; the codes stay the same in the next run
    local = pd.read_csv(run / "local" / "annotators.csv")
    assert set(local["ui_name"]) == {ui for _, ui in USERS if ui != "city2_anno1"}
    assert (run / "local" / "exclusion_report.md").exists() and (run / "local" / "gemtex-iaa-semantic.log").exists()
    key = pd.read_csv(directory / "results" / "pseudonyms.csv")
    assert set(key["kind"]) == {"document", "annotator"} and len(key) == 7
    assert pd.read_csv(second / "share" / "pair_counts.csv").equals(pd.read_csv(run / "share" / "pair_counts.csv"))


def test_pool(runs, tmp_path):
    directory, (a, b) = runs
    assert main(["pool", str(a), str(b / "share"), "--out", str(tmp_path), "--bootstrap", "50"]) == 0
    pooled_run = sorted(tmp_path.glob("*_pool"))[-1]
    pooled = pd.read_csv(pooled_run / "pooled_figures.csv").set_index(["site", "figure"])["value"]
    single = pd.read_csv(a / "share" / "figures.csv").set_index("figure")["value"]
    for name, value in single.items():                                  # the same data twice: same ratios
        for site in ("ALL", "site_a", "site_b"):
            assert pooled[(site, name)] == pytest.approx(value)
    annotators = pd.read_csv(pooled_run / "pooled_annotators.csv")
    assert set(annotators["site"]) == {"site_a", "site_b"} and annotators["annotator"].str.match(r"site_[ab]:A").all()
    report = (pooled_run / "pooled_report.md").read_text(encoding="utf-8")
    assert "Pooled agreement report (2 sites)" in report and "## Warnings\n\nnone" in report
    with pytest.raises(ValueError, match="given twice"):
        C.pool([a, a])
    # a share made before α was added: pooled without that site's α, with a warning
    old = tmp_path / "old_share"
    shutil.copytree(b / "share", old)
    counts = pd.read_csv(old / "pair_counts.csv")
    counts.drop(columns=["alpha_n", "alpha_agree", "alpha_expected"]).to_csv(old / "pair_counts.csv", index=False)
    result = C.pool([a, old], n_boot=10)
    values = result["figures"].set_index(["site", "figure"])["value"]
    assert pd.isna(values[("site_b", ALPHA)]) and values[("ALL", ALPHA)] == pytest.approx(single[ALPHA])
    assert values[("ALL", "span F1 strict")] == pytest.approx(single["span F1 strict"])
    assert any(w.startswith("counts missing") and "site_b" in w for w in result["warnings"])


def test_site_report_needs_a_site_and_handles_no_documents(tmp_path, scenario_zip, mini_hdf5):
    with pytest.raises(SystemExit, match="SITE"):
        main(["site-report", str(site_config(tmp_path, scenario_zip, mini_hdf5, ""))])
    path = site_config(tmp_path, scenario_zip, mini_hdf5, "site_c", MIN_ANNOTATORS=9)
    assert main(["site-report", str(path)]) == 0
    run = sorted((tmp_path / "results").glob("*_site-report"))[-1]
    summary = json.loads((run / "share" / "summary.json").read_text(encoding="utf-8"))
    assert summary["data"]["documents_analysed"] == 0 and summary["data"]["documents_too_few_assigned"] == 2
    assert all(f["value"] is None for f in summary["figures"])


def test_inspect(test_config, capsys):
    assert main(["inspect", str(test_config)]) == 0
    out = capsys.readouterr().out
    assert "included annotators per document" in out and "resolution status" in out
    assert "Fieber" not in out                  # text-free output
    assert "pre-annotation (INITIAL_CAS): 2 concepts in 1 documents" in out
    assert "inactive:9000099" in out and "malformed:<unparsed>" in out
