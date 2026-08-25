# BSRF-HEPS Knowledge-Graph Scale and Structure Audit

## Counting Rules

- The main graph is read from `Sheet1` of `BSRF_HEPS_tuple.xlsx`.
- Unique directed edges are counted after whitespace normalization using `(sub_name, rel_name, obj_name)`.
- Unique nodes are counted by normalized entity name, consistent with the graph loader's name-based deduplication.
- Typed nodes are unique `(entity name, entity type)` pairs; one entity name may therefore contribute multiple typed nodes.
- Relation names and entity types use the original table labels; no semantic merging is applied.
- This report is a structural audit and does not represent extraction precision, recall, or expert agreement.

## Overall statistics

| Dataset | Nonblank rows | Complete triples | Unique directed edges | Unique nodes | Entity-type labels | Relation-name labels |
|---|---|---|---|---|---|---|
| Main knowledge graph | 607 | 607 | 604 | 526 | 36 | 318 |
| Experimental technology special map | 354 | 351 | 341 | 285 | 38 | 11 |

## Candidate Source-Corpus Size

| Metric | Value |
|---|---|
| Complete K-V records | 550 |
| Unique K-V pairs | 547 |
| Substantive K-V records | 527 |
| Placeholder records | 23 |
| Total K-V characters | 231,900 |
| Substantive K-V characters | 229,263 |

Note: the graph table has no source-document or paragraph identifiers, so the script cannot verify whether the K-V file is the complete extraction source. Do not interpret these values as graph source-corpus statistics without independent confirmation.

## Main knowledge graph

Quality audit: 3 extra exact duplicate rows; 35 entity names with multiple types; 7 self-loop candidates; and 31 unique edges with differing `rel_type` and `rel_name` values.

### Entity type distribution

| Raw entity type | Unique nodes |
|---|---|
| Line station introduction | 81 |
| Equipment parameters | 68 |
| Research areas | 61 |
| experimental techniques | 56 |
| Research content | 54 |
| line station | 40 |
| Beam line | 29 |
| Equipment introduction | 22 |
| sample environment | 19 |
| HEPS first phase | 16 |
| Official website page | 16 |
| person in charge | 16 |
| Experimental procedures | 14 |
| Equipment | 12 |
| Data processing steps | 10 |
| Data processing software | 9 |
| Contact person | 6 |
| Experimental Operation Manual | 6 |
| Sample processing steps | 6 |
| Experimental Notes | 4 |
| Line station layout diagram | 4 |
| Test method | 3 |
| HEPS parameters | 2 |
| Line station status | 2 |
| On and off light operation | 2 |
| Project application steps | 2 |
| Sample preparation | 2 |
| detector | 2 |
| Beijing synchrotron radiation | 1 |
| Common faults | 1 |
| Contact information | 1 |
| Entity | 1 |
| HEPS trial subject application steps | 1 |
| Running status | 1 |
| high energy synchrotron radiation | 1 |
| storage ring status | 1 |

### Top 15 Relation Names

| Raw relation name | Unique edges |
|---|---|
| Research content | 37 |
| Research areas | 35 |
| experimental techniques | 34 |
| Beam line | 29 |
| Line station status | 21 |
| HEPS construction | 17 |
| Official website page | 16 |
| Introduction | 14 |
| person in charge | 14 |
| Title | 11 |
| Main equipment | 9 |
| Beijing Synchrotron Radiation Experiment Application Process | 8 |
| Experiment manual | 8 |
| Equipment parameters | 4 |
| Experimental methods | 4 |

The remaining 303 relation names account for 343 unique edges. See the CSV output for the complete distribution.

## Experimental technology special map

Quality audit: 10 extra exact duplicate rows; 31 entity names with multiple types; 0 self-loop candidates; and 173 unique edges with differing `rel_type` and `rel_name` values.

### Entity type distribution

| Raw entity type | Unique nodes |
|---|---|
| Spot parameters | 134 |
| experimental techniques | 87 |
| Beam line | 29 |
| HEPS first phase | 16 |
| Research areas | 12 |
| sample environment | 10 |
| Line station status | 4 |
| Research content | 4 |
| HEPS parameters | 2 |
| 1B3-Lithography, LIGA experimental station | 1 |
| 1W1A Experimental Station | 1 |
| 1W1B experimental station | 1 |
| 1W2A Experimental Station | 1 |
| 1W2B experimental station | 1 |
| 3W1 Experimental Station | 1 |
| 4B7A Experiment Station | 1 |
| 4B7B Experiment Station | 1 |
| 4B8-Experiment Station | 1 |
| 4B9A Experiment Station | 1 |
| 4B9B Experiment Station | 1 |
| 4W1A Experimental Station | 1 |
| 4W1B Experimental Station | 1 |
| 4W2 Experimental Station | 1 |
| BM44 | 1 |
| Beijing synchrotron radiation | 1 |
| HEPS trial subject application steps | 1 |
| ID02 | 1 |
| ID05 | 1 |
| ID07 | 1 |
| ID08 | 1 |
| ID09 | 1 |
| ID19 | 1 |
| ID23 | 1 |
| ID30 | 1 |
| ID31 | 1 |
| ID33 | 1 |
| ID46 | 1 |
| high energy synchrotron radiation | 1 |

### Top 15 Relation Names

| Raw relation name | Unique edges |
|---|---|
| Spot parameters | 141 |
| experimental techniques | 97 |
| Beam line | 29 |
| Line station status | 28 |
| HEPS construction | 17 |
| Research areas | 10 |
| sample environment | 10 |
| Research content | 6 |
| HEPS brightness | 1 |
| HEPS electronic energy | 1 |
| HEPS trial subject application | 1 |

## Interpretation Limits

The raw table contains more entity-type and relation labels than the core ontology categories discussed in the manuscript, as well as duplicate records, self-loop candidates, and names assigned to multiple types. Domain experts should complete label mapping and exception adjudication before final category counts are reported. Historical synonym merges, corrections, and removals cannot be reconstructed from the final triple table alone.
