# SR-ExpRecSys

Code and data for the study:

**A large-language-model-based recommendation system for multi-beamline synchrotron experimental schemes**

Repository: https://github.com/LinaZhaoAIGroup/SR-ExpRecSys

## Overview

This project develops a retrieval-augmented generation (RAG) system for preliminary synchrotron radiation experiment planning. The system uses knowledge-base retrieval, vector similarity, knowledge-graph information, and a large language model to provide experiment-related suggestions.

The system is designed for information support only. Final experimental conditions, safety requirements, operating procedures, and beamline arrangements must be confirmed with qualified beamline staff.

## Repository contents

- `app/`: Django web application and application knowledge resources.
- `evaluation/01-bge-m3-similarity/`: BGE-M3 similarity evaluation.
- `evaluation/02-llm-judge/`: LLM-based evaluation of correctness, usefulness, and safety.
- `evaluation/03-human-evaluation/`: anonymized human evaluation and statistical analysis.
- `evaluation/04-data-audit/`: corpus and knowledge-graph audits.
- `evaluation/05-ablation/`: ablation comparison of different retrieval settings.


The benchmark contains 24 questions. Each question was evaluated with RAG and No-RAG systems using repeated response-generation runs. The repository includes the archived workbooks, processed tables, analysis scripts, and figures.

## Installation

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
python -m pip install -r requirements-analysis.txt
~~~

## Reproduce the analyses

Run commands from the repository root:

~~~bash
python 'evaluation/01-bge-m3-similarity/BGE-M3相似度计算.py'
python 'evaluation/03-human-evaluation/analysis/analyze_human_evaluation.py' --input 'evaluation/03-human-evaluation/analysis/Evaluation_Human .xlsx' --output 'evaluation/03-human-evaluation/analysis/human_evaluation_results' --bootstrap 20000 --seed 20260812
python 'evaluation/05-ablation/plot_ablation_from_csv.py'
~~~


## Application status

The prepared public copy does not include source files containing embedded API keys or database credentials. Before running the web application, add sanitized versions of:

~~~text
app/chat/ai_service.py
app/chat/vector_service.py
scripts/data_preparation/emebedding.py
scripts/data_preparation/init_py2neo.py
scripts/data_preparation/kg-tripe.py
~~~

Never commit API keys, passwords, `.env` files, local databases, or private evaluation records.

