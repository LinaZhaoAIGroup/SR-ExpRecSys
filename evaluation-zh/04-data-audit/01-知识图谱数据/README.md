# 知识图谱规模与抽取验证脚本

本目录中的脚本只读原始知识图谱文件，不会覆盖 CSV 或 XLSX。

## 依赖

```bash
python -m pip install openpyxl
```

## 1. 统计图谱和源语料库规模

```bash
python calculate_kg_statistics.py
```

默认输出到 `outputs/kg_statistics/`，包括 JSON、Markdown、实体类型分布、关系分布、质量候选项以及关系和实体类型映射模板。默认主图谱为 `BSRF_HEPS_tuple.xlsx` 的 `Sheet1`；旧版 `Sheet1 (2)` 不参与主图谱统计。

## 2. 生成两位专家的独立复核样本

```bash
python prepare_kg_validation_sample.py --sample-size 235 --seed 20260812
```

默认输出到 `validation/round_1/`。两位专家分别填写自己的 CSV 文件，首次评审完成前不交换判断。每条三元组应补充可定位的来源证据。

## 3. 计算双专家指标

```bash
python calculate_kg_validation_metrics.py
```

首次运行后会生成完整的 `adjudication_template.csv` 和分歧清单。完成裁决后运行：

```bash
python calculate_kg_validation_metrics.py \
  --adjudicated validation/round_1/metrics/adjudication_template.csv
```

对全图谱实际执行并逐条登记清洗后，再增加 `--cleaning-log kg_cleaning_log_template.csv`。未填写的日志不能用于声称清洗数量。

清洗日志的 `operation` 只允许：`REMOVE_EXACT_DUPLICATE`、`CORRECT_TRIPLE`、`REMOVE_INVALID_TRIPLE`、`MERGE_SYNONYM`、`NORMALIZE_RELATION`、`NORMALIZE_ENTITY_TYPE` 和 `ADD_MISSING_TRIPLE`。

若已通过独立来源单元建立 recall 金标准，再附加：

```bash
--recall-file validation/round_1/recall_gold_triples.csv
```

抽样表中的 `CORRECT`/`REMOVE` 只表示样本处理建议。论文中的全图谱纠正、删除、同义词合并和关系规范化数量，必须来自逐条填写的全图谱清洗日志。
