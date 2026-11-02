# gemtex-iaa-semantic

Inter-annotator agreement of the GeMTeX semantic annotation (SNOMED CT concepts in INCEpTION, GeMTeX annotation
scheme, [guidelines v2.3.3](https://zenodo.org/records/20085886)), computed locally at each consortium site and
pooled centrally.

Each site runs `site-report` on its own INCEpTION exports and sends back only aggregated, pseudonymised figures and
counts: no document text, no names and no concepts. `pool` combines the returned folders of all sites into the
consortium figures.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10.

```bash
git clone <this repository> && cd gemtex-iaa-semantic
uv sync --no-dev
cp config/example.cfg config/site.cfg
```

In `config/site.cfg`, set:
* `SITE`: the name of your site as agreed in the consortium;
* `PATH_PROJECT`: your zipped INCEpTION project exports (UIMA CAS XMI); files, directories (every `*.zip` in them)
  or glob patterns, comma or line separated;
* `HDF5`: the SNOMED CT file (International Edition 2024-04-01) distributed in the consortium. It is licensed and
  not part of this repository.

`config/site.cfg` and `data/` are ignored by git, so exports, the HDF5 file and results cannot be committed by
accident.

If `uv run` fails with `ModuleNotFoundError: No module named 'gemtex_iaa_semantic'` (on macOS, e.g. in folders
synced with iCloud, the files of the editable install can be flagged hidden, and Python skips them), install it
without editable mode and run without re-syncing: `uv sync --no-dev --no-editable`, then
`uv run --no-sync gemtex-iaa-semantic …` (after updating the code, run the `uv sync` again).

## At each site

```bash
uv run --no-dev gemtex-iaa-semantic inspect config/site.cfg
uv run --no-dev gemtex-iaa-semantic site-report config/site.cfg
```

`inspect` prints text-free counts: documents per number of annotators (multiply annotated, pending, single),
annotation states, layers, value formats and how many concept ids resolve in the SNOMED CT release.

`site-report` writes `DIR_OUTPUT/<timestamp>_site-report/` with two folders:
* `share/` is sent back. Documents and annotators appear only as codes (`D0001`, `A01`). It holds no document text,
  no file, project, account or display names, no concepts and no paths. Files:
  * `report.md` and `summary.json`: data counts, figures, settings, tool version;
  * `figures.csv`, `coverage.csv` (what κ and α leave out), `annotators.csv` and `annotator_pairs.csv`;
  * `pair_counts.csv`: the counts behind every figure per document and annotator pair.
* `local/` stays at the site: the same tables with real names, `exclusion_report.md` and the log.

The codes are kept in `DIR_OUTPUT/pseudonyms.csv` (local) and stay the same across runs. They are numbered, not
hashed, because a hash of a file name could be recovered by guessing the name. Exports are read once and cached in
`DIR_CACHE`; `--no-cache` reads them again.

## Data selection

* Only documents with at least `MIN_ANNOTATORS` (default 2) annotators in `STATES` (default `FINISHED`) are read.
  Several exports form one dataset: a document name in several exports is one document if its text is the same
  in all of them, and accounts are identified by username.
* Annotation documents are excluded before any figure is computed, and listed in `local/exclusion_report.md`:
  * `untouched`: identical to the automatic pre-annotation (`INITIAL_CAS`) of its document (`EXCLUDE_UNTOUCHED`);
  * `copy`: identical to another annotation document of the same document; one of each group is kept
    (`EXCLUDE_COPIES`);
  * `low_change`: F1 against the pre-annotation ≥ 1 − `PREANNO_LOW_CHANGE`; always counted, excluded only with
    `EXCLUDE_LOW_CHANGE`.
* Documents left with fewer than `MIN_ANNOTATORS` annotators are dropped.

## Figures

Annotation documents are compared by span and *concept key*: the concept (an `id` resolving to an active concept of
the release) plus its normalised literal, the literal alone for numbers and dates, or the unresolved code. Every
figure is pooled over documents and annotator pairs (sum of numerators / sum of denominators), with a 95 %
percentile bootstrap CI over documents:

* **span F1 strict / lenient**: mentions of two annotators with the same span / with overlapping spans (matched one
  to one); stacked spans count once;
* **end-to-end F1**: mentions with the same span and the same concept key;
* **Cohen's κ** on spans each annotator marked exactly once and identically, with chance agreement per document
  and annotator pair (De Vries et al. 2008);
* **Krippendorff's α** (nominal) on the concepts of clusters of overlapping mentions (all annotators of a
  document) in which no annotator has more than one mention, with chance agreement per document, pooled like κ.
  Chance agreement over all documents would need the concept frequencies, which are not shared;
* **share inherited**: of the regions where two annotators agree exactly, the share where both kept the
  pre-annotation unchanged. High values mean that the agreement largely repeats the pre-annotation.

Per annotator and annotator pair, `annotators.csv` and `annotator_pairs.csv` give the same figures; per annotator,
they also count the untouched, copied and low-change annotation documents.

### Stacked concepts and coverage

A *stack* is one annotator putting several concepts on the same span. Only end-to-end F1 counts each concept of a
stack. The span F1s count the span once, and an exactly agreeing region (share inherited) needs the same stack on
both sides. κ and α need one mention per annotator and leave the rest out, so `coverage.csv` (and the section
"Coverage of κ and α" of the report) says how much of their data they use and why the rest is left out:

* **κ**: of the regions where both annotators marked the same spans, those with a stack, and those with several
  spans marked identically by both (e.g. nested ones). Regions with different boundaries are outside κ by design;
* **α**: of the annotator × cluster values in clusters of at least two annotators, the clusters where an annotator
  stacked concepts, and those where an annotator marked several spans (split, nested or overlapping).

`annotators.csv` and `annotator_pairs.csv` have the same shares (`kappa_coverage`, `alpha_coverage`), and
`annotators.csv` counts each annotator's stacked spans (`stacked_spans`).

## Pooling

Centrally, with the returned `share` folders (or run directories):

```bash
uv run gemtex-iaa-semantic pool <share folders…> [--out DIR]
```

`pool` recomputes every figure and the coverage over all documents of all sites, overall and per site, from
`pair_counts.csv`, and writes `DIR/<timestamp>_pool/pooled_report.md` with the CSV tables. Documents and annotators are site-specific
(`<site>:<code>`). It warns if the sites used different tool versions or settings, and pools a share without some
counts (made with an older version) without the figures that need them.

## Development

```bash
uv sync
uv run pytest
```

The tests use a synthetic export and a tiny fake SNOMED CT file with made-up concept ids (`tests/`).

## License

MIT, see [LICENSE](LICENSE).
