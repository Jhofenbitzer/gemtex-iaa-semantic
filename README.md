# gemtex-iaa-semantic

This tool computes inter-annotator agreement for the semantic annotation in GeMTeX, i.e. SNOMED CT concepts
annotated in INCEpTION following the GeMTeX [annotation guidelines (v2.3.3)](https://zenodo.org/records/20085886).

The clinical texts can't leave the sites, so the agreement is computed where the data is. Every site runs
`site-report` on its own INCEpTION exports and gets a folder of counts and figures without any text, names or
concepts, which it sends back. `pool` then combines the folders of all sites into consortium-wide numbers.

## Setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer.

```bash
git clone <this repository> && cd gemtex-iaa-semantic
uv sync --no-dev
cp config/example.cfg config/site.cfg
```

Most settings in `config/site.cfg` can stay as they are, but three need your attention:

- `SITE` is the name of your site, as agreed in the consortium.
- `PATH_PROJECT` points to your zipped INCEpTION project exports (UIMA CAS XMI). You can give files, directories
  (all `*.zip` files in them are read) or glob patterns, separated by commas or line breaks.
- `HDF5` is the path to the SNOMED CT file (International Edition 2024-04-01) that we distribute within the
  consortium. It is licensed, so it isn't part of this repository.

Git ignores `config/site.cfg` and `data/`, so you can't commit your exports, the HDF5 file or your results by
mistake.

### Troubleshooting on macOS

If `uv run` stops with `ModuleNotFoundError: No module named 'gemtex_iaa_semantic'`, macOS has probably flagged the
files of the editable install as hidden (this happens in folders synced with iCloud), and Python skips hidden files.
Installing without editable mode gets around it:

```bash
uv sync --no-dev --no-editable
uv run --no-sync gemtex-iaa-semantic ...
```

In that case, run the `uv sync` command again whenever you update the code.

## Running the site report

```bash
uv run --no-dev gemtex-iaa-semantic inspect config/site.cfg
uv run --no-dev gemtex-iaa-semantic site-report config/site.cfg
```

Start with `inspect`. It prints a text-free overview of your exports: how many documents have several annotators
(and how many are still pending or have only one), annotation states, layers, value formats, and how many concept
ids resolve in the SNOMED CT release. It's a quick way to check the configuration before the actual run.

`site-report` creates a new folder `DIR_OUTPUT/<timestamp>_site-report/` with two subfolders:

- `share/` is what you send back. Documents and annotators only appear as codes (`D0001`, `A01`), and there is no
  document text, no file, project, account or display name, no concept and no path in it. It holds the report
  (`report.md`, `summary.json`), the tables `figures.csv`, `coverage.csv`, `annotators.csv` and
  `annotator_pairs.csv`, and `pair_counts.csv` with the raw counts per document and annotator pair that every
  figure is computed from.
- `local/` stays with you. It has the same tables with the real names, the exclusion report and the log.

The mapping from codes to names is kept locally in `DIR_OUTPUT/pseudonyms.csv`, and the codes don't change between
runs. They are numbered rather than hashed, because a hashed file name can be recovered by guessing the name.

The exports are parsed once and cached in `DIR_CACHE`. Pass `--no-cache` to read them again.

## Which documents are used

Only documents with at least `MIN_ANNOTATORS` annotators (default 2) in one of the `STATES` (default `FINISHED`)
are read; the rest are only counted. If you give several exports, they are merged. A document that appears in more
than one export counts as one document as long as its text is the same everywhere, and annotators are matched by
username.

Some annotation documents are taken out before anything is computed:

- `untouched`: identical to the automatic pre-annotation (`INITIAL_CAS`), so the annotator didn't change anything
  (`EXCLUDE_UNTOUCHED`).
- `copy`: identical to another annotator's version of the same document. One of each group of copies is kept
  (`EXCLUDE_COPIES`).
- `low_change`: only slightly different from the pre-annotation, with an F1 against it of at least
  1 - `PREANNO_LOW_CHANGE`. These are always counted, but only excluded if `EXCLUDE_LOW_CHANGE` is set.

A document that is left with fewer than `MIN_ANNOTATORS` annotators after this is dropped. Everything that was
excluded is listed in `local/exclusion_report.md`.

## What is measured

Two annotations are compared by their span and their *concept key*. The key is the SNOMED CT concept, if the `id`
resolves to an active concept of the release, together with its normalised literal. For numbers and dates it is the
literal alone, and for anything that doesn't resolve it is the code itself.

All figures are pooled over documents and annotator pairs (numerators and denominators are summed before dividing)
and come with a 95 % bootstrap confidence interval over documents.

- Span F1, strict and lenient: do two annotators mark the same spans (strict), or overlapping ones (lenient,
  matched one to one)?
- End-to-end F1: same span and same concept key.
- Cohen's κ on spans that both annotators marked exactly once and identically, with chance agreement per document
  and annotator pair (De Vries et al. 2008).
- Krippendorff's α (nominal) over clusters of overlapping mentions of all annotators of a document, as long as
  nobody has more than one mention in the cluster. Chance agreement is computed per document, and the result is
  pooled the same way as κ. Computing it over all documents at once would mean sharing concept frequencies, which we
  don't do.
- The inherited share: among the regions where two annotators agree exactly, how often both simply kept the
  pre-annotation. A high value means that much of the agreement comes from the pre-annotation rather than from the
  annotators.

`annotators.csv` and `annotator_pairs.csv` have the same figures per annotator and per pair. The annotator table
also counts each annotator's untouched, copied and low-change documents.

### Stacks, and what κ and α leave out

INCEpTION lets an annotator put several concepts on the same span (a stack), and the measures treat this
differently. End-to-end F1 is the only one that counts every concept of a stack. The span F1s count a stacked span
once, and a region only counts as exact agreement (as used for the inherited share) if both annotators have the same
stack.

κ and α need exactly one mention per annotator and skip everything else. To make that visible, the report has a
section "Coverage of κ and α" (also in `coverage.csv`) that shows how much of the data each of them uses and why the
rest was skipped:

- κ looks at the regions where both annotators marked the same spans. It skips regions with a stack, and regions
  with several spans that both marked identically, for example nested ones. Regions with different boundaries are
  outside κ anyway; that's what the span F1s are for.
- α looks at clusters with at least two annotators, counted as annotator × cluster values. It skips clusters where
  someone stacked concepts, and clusters where someone marked several spans (split, nested or overlapping).

The annotator and pair tables show the same shares as `kappa_coverage` and `alpha_coverage`, and `annotators.csv`
has a `stacked_spans` column, so you can see who stacks.

## Pooling the sites

Once the `share` folders are back (whole run directories work too):

```bash
uv run gemtex-iaa-semantic pool <share folders…> [--out DIR]
```

This recomputes every figure and the coverage from the `pair_counts.csv` files, over all documents of all sites and
for each site on its own. The report ends up in `DIR/<timestamp>_pool/pooled_report.md`, next to the CSV tables.
Documents and annotators are kept apart per site (`<site>:<code>`).

`pool` warns you if the sites used different versions of the tool or different settings. A share from an older
version that lacks some counts is still pooled, but that site is left out of the figures that need those counts.

## Development

```bash
uv sync
uv run pytest
```

The tests run on a small synthetic export and a fake SNOMED CT file with made-up concept ids, so you need neither
real data nor the licensed release.

## License

MIT, see [LICENSE](LICENSE).
