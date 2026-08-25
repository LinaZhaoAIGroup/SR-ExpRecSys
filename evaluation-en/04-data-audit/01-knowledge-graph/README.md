# Knowledge-Graph Audit and Validation

The scripts in this directory read the released CSV and Excel files. They do
not overwrite the source knowledge graph.

## Install

```bash
python -m pip install openpyxl
```

## Structural Audit

```bash
python calculate_kg_statistics.py
```

The default output directory is `outputs/kg_statistics/`. It contains the
machine-readable report, a Markdown summary, entity-type and relation-label
distributions, quality flags, and mapping templates. The default main graph is
`02-full-kg/BSRF_HEPS_tuple.xlsx`, sheet `Sheet1`.

## Blind Expert-Validation Sample

```bash
python prepare_kg_validation_sample.py --sample-size 235 --seed 20260812
```

The default output is `validation/round_1/`. It contains two identical
annotation templates, one for each independent expert, plus source-unit and
gold-triple templates for a separately designed recall study. Experts should
complete their files independently and record the source document and location
for every judgment.

## Agreement and Precision Metrics

After both experts complete their forms:

```bash
python calculate_kg_validation_metrics.py
```

After adjudication, provide the completed adjudication template:

```bash
python calculate_kg_validation_metrics.py \
  --adjudicated validation/round_1/metrics/adjudication_template.csv
```

For full-map cleaning counts, provide a completed cleaning log with one row per
operation:

```bash
python calculate_kg_validation_metrics.py \
  --cleaning-log kg_cleaning_log_template.csv
```

Allowed operation codes are `REMOVE_EXACT_DUPLICATE`, `CORRECT_TRIPLE`,
`REMOVE_INVALID_TRIPLE`, `MERGE_SYNONYM`, `NORMALIZE_RELATION`,
`NORMALIZE_ENTITY_TYPE`, and `ADD_MISSING_TRIPLE`.

## Recall

Sampling extracted triples estimates precision, not recall. A recall estimate
requires an independent random sample of source documents or source passages;
experts must enumerate gold-standard triples without seeing system output. The
released triple table does not contain paragraph-level provenance, so it is not
sufficient by itself for a reliable recall estimate.
