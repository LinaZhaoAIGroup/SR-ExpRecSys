# Evaluation and Data Audit

**Language:** English | [中文说明](README_CN.md)  
**Project README:** [English](../README.md) | [中文](../README_CN.md)

## Purpose

This directory contains the benchmark, archived repeated responses, similarity calculations, LLM-judge scores, anonymized human ratings, ablation analysis, and knowledge-resource audits supporting the manuscript **A large-language-model-based recommendation system for multi-beamline synchrotron experimental schemes**.

The released analysis is designed to reproduce the reported tables and figures without requiring new paid API calls. API reruns are optional and may not reproduce the archived text or scores exactly.

## Evaluation map

| Directory | Evaluation | Main question | Main entry point |
|---|---|---|---|
| 01-bge-m3-similarity/ | BGE-M3 similarity | How similar is each answer to the expert reference? | BGE-M3相似度计算.py |
| 02-llm-judge/ | LLM judge | How do RAG and No-RAG differ in correctness, usefulness, and safety? | 03--综合结果与SI/plot_gpt_scores_manuscript.py |
| 03-human-evaluation/ | Human evaluation | How do three anonymized experts rate the two systems? | analysis/analyze_human_evaluation.py |
| 04-data-audit/ | Corpus and graph audit | What is the size, completeness, and structure of the knowledge resources? | 00-语料库规模/ and 01-知识图谱数据/ |
| 05-ablation/ | Ablation | What changes when KG retrieval or vector retrieval is removed? | plot_ablation_from_csv.py |

The original Chinese filenames are retained because several scripts depend on them. Run commands from the repository root unless a command explicitly changes directory.

## Common experimental design

The benchmark contains 24 questions divided into four levels, with six questions per level:

| Level | Description | Questions |
|---|---|---:|
| L1 | Basic or single-technique information requests | 6 |
| L2 | Sample- and objective-specific planning | 6 |
| L3 | Complex, extreme-condition, multi-technique, or cross-beamline planning | 6 |
| L4 | Safety, feasibility, operating-procedure, and proposal-challenge cases | 6 |

The compared systems are RAG (回答1) and No-RAG (回答2). Each question-system combination has three response-generation runs. Every response is compared with the expert-authored Gold Standard in the same row.

The primary inferential unit is the question (n = 24). Repeated runs describe stability and form question-level summaries; they must not be treated as additional independent questions.

## 1. BGE-M3 similarity

### Input

~~~text
evaluation/01-bge-m3-similarity/测试问题.xlsx
~~~

The workbook has Sheet1--Sheet3, one for each response-generation run. The scripts calculate dense, sparse, ColBERT, and combined BGE-M3 answer-to-gold scores.

### Recompute the scores

~~~bash
cd evaluation/01-bge-m3-similarity
python BGE-M3相似度计算.py
python BGE-M3稀疏相似度计算.py
python BGE-M3综合相似度计算.py
cd ../..
~~~

Expected processed workbooks:

~~~text
evaluation/01-bge-m3-similarity/BGE-M3相似度计算结果.xlsx
evaluation/01-bge-m3-similarity/BGE-M3稀疏相似度计算结果.xlsx
evaluation/01-bge-m3-similarity/BGE-M3综合相似度计算结果.xlsx
~~~

The first run may download BAAI/bge-m3. To regenerate figures:

~~~bash
python 'evaluation/01-bge-m3-similarity/画图与结果分析/plot_dense_similarity_sci.py'
python 'evaluation/01-bge-m3-similarity/画图与结果分析/plot_colbert_similarity_sci.py'
python 'evaluation/01-bge-m3-similarity/02-三维空间距离相似度降维/plot_bge_m3_pca_3d.py'
~~~

Outputs are stored in the dense_similarity_figures/, colbert_similarity_figures/, and bge_m3_pca_3d_figures/ subdirectories.

## 2. LLM-judge evaluation

### Archived inputs

~~~text
evaluation/02-llm-judge/01--GPT-评估打分/测试问题_GPT评分_1.xlsx
evaluation/02-llm-judge/01--GPT-评估打分/测试问题_GPT评分2.xlsx
evaluation/02-llm-judge/01--GPT-评估打分/测试问题_GPT评分3.xlsx
~~~

Each workbook contains scores for all three response-generation sheets. The three workbooks are independent judge repeats. Correctness, usefulness, and safety are scored on a 1--10 scale.

### Reproduce the combined analysis

~~~bash
python 'evaluation/02-llm-judge/02--画图与结果分析/03--综合结果与SI/plot_gpt_scores_manuscript.py'
python 'evaluation/02-llm-judge/02--画图与结果分析/03--综合结果与SI/plot_gpt_scores_by_level.py'
python 'evaluation/02-llm-judge/02--画图与结果分析/04--四类问题配对图/plot_gpt_scores_four_levels.py'
~~~

The workflow crosses three response-generation runs with three judge repeats, yielding nine scores per question, system, and dimension. Scores are averaged within each question, followed by paired two-sided Wilcoxon signed-rank tests between RAG and No-RAG. Holm correction is applied across correctness, usefulness, and safety.

Main outputs:

~~~text
evaluation/02-llm-judge/02--画图与结果分析/03--综合结果与SI/plot_gpt_scores_manuscript/
evaluation/02-llm-judge/02--画图与结果分析/04--四类问题配对图/
~~~

### Optional API rerun

The API-scoring entry point is evaluation/02-llm-judge/01--GPT-评估打分/gpt_evaluate_excel.py. An API rerun requires a valid key supplied through the environment and may incur cost. Results can vary with model versions, prompts, service behavior, and sampling. Never put a key in source code or commit .env.

## 3. Human evaluation

### Input and design

~~~text
evaluation/03-human-evaluation/analysis/Evaluation_Human .xlsx
~~~

The anonymized workbook contains three sheets, 24 questions per sheet, two systems, three dimensions, and three evaluators identified only as Expert 1, Expert 2, and Expert 3.

~~~text
24 questions x 3 response runs x 2 systems x 3 dimensions x 3 experts
= 1,296 ratings
~~~

### Reproduce the analysis

~~~bash
python 'evaluation/03-human-evaluation/analysis/analyze_human_evaluation.py' --input 'evaluation/03-human-evaluation/analysis/Evaluation_Human .xlsx' --output 'evaluation/03-human-evaluation/analysis/human_evaluation_results' --bootstrap 20000 --seed 20260812
~~~

The analysis averages the three response runs and three experts for each question, system, and dimension. It reports descriptive summaries, paired effects, 20,000-replicate bootstrap confidence intervals, two-sided Wilcoxon tests with Holm correction, Cohen's dz, and descriptive ICC estimates.

Outputs:

~~~text
evaluation/03-human-evaluation/analysis/human_evaluation_results/figures/
evaluation/03-human-evaluation/analysis/human_evaluation_results/tables/
~~~

Named individual expert workbooks are excluded from the public release.

## 4. Corpus and knowledge-graph audit

### Corpus scale

~~~bash
python 'evaluation/04-data-audit/00-语料库规模/calculate_corpus_scale.py'
~~~

This audit reports record counts, empty values, duplicate keys, and placeholder-like entries in the key-value corpus.

### Knowledge-graph structure

~~~bash
python 'evaluation/04-data-audit/01-知识图谱数据/calculate_kg_statistics.py'
python 'evaluation/04-data-audit/01-知识图谱数据/calculate_kg_validation_metrics.py'
python 'evaluation/04-data-audit/01-知识图谱数据/prepare_kg_validation_sample.py'
~~~

The released audit distinguishes the application graph snapshot from the larger structural-audit table. It reports complete triples, unique edges, relation and node distributions, and validation samples. The original local kg_statistics_report.json was excluded because it contained author-computer paths; rerunning the audit in a clean clone generates a report for that environment.

## 5. Ablation analysis

Input workbook:

~~~text
evaluation/05-ablation/测试问题结果对比.xlsx
~~~

The compared conditions are KG-only, vector-only, combined KG-vector RAG, and No-RAG.

Replot archived scores without a model download:

~~~bash
python 'evaluation/05-ablation/plot_ablation_from_csv.py'
~~~

Recompute BGE-M3 scores:

~~~bash
python 'evaluation/05-ablation/plot_ablation_bge_m3.py'
~~~

Outputs are stored in evaluation/05-ablation/bge_m3_ablation_figures/.

## Installation and reproducibility

From the repository root:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt
python --version
python -m pip freeze
~~~

The requirements file records package names used by the current scripts but is not fully version-pinned. Record final Python and package versions for the archival release. For workbook fields and expected row counts, see [the data dictionary](../docs/DATA_DICTIONARY.md) and [the reproducibility guide](../docs/REPRODUCIBILITY.md).

## Interpretation and limitations

- The benchmark has 24 question-level units; repeated answers and ratings do not create additional independent questions.
- LLM-judge and API-generated results can change with model revisions and service behavior.
- Similarity scores are model- and corpus-dependent and are not universal calibration values.
- Human ratings are anonymized, but workbook metadata and cell contents still require author review before publication.
- The system is a preliminary planning aid and does not replace official procedures, safety review, instrument training, or beamline-staff consultation.

## Citation and availability

Add the final GitHub URL, archival DOI, article DOI, license, and contact information before publication. The repository root contains the project README and release-status checklist.
