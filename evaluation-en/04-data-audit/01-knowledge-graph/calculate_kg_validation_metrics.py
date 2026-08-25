#!/usr/bin/env python3
"""Calculate two-expert KG extraction validation metrics from CSV forms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VALIDATION_DIR = SCRIPT_DIR / "validation" / "round_1"
DEFAULT_EXPERT_1 = DEFAULT_VALIDATION_DIR / "expert_1_annotations.csv"
DEFAULT_EXPERT_2 = DEFAULT_VALIDATION_DIR / "expert_2_annotations.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_VALIDATION_DIR / "metrics"

ID_COLUMN = "sample_id"
BINARY_COLUMNS = [
    "source_support_Y_N_UNCERTAIN",
    "subject_correct_Y_N_UNCERTAIN",
    "relation_correct_Y_N_UNCERTAIN",
    "object_correct_Y_N_UNCERTAIN",
    "entity_types_correct_Y_N_UNCERTAIN",
    "triple_correct_Y_N_UNCERTAIN",
]
COMPONENT_CORRECTNESS_COLUMNS = BINARY_COLUMNS[:-1]
TRIPLE_CORRECT_COLUMN = "triple_correct_Y_N_UNCERTAIN"
ERROR_COLUMN = "error_types_semicolon_separated"
ACTION_COLUMN = "action_KEEP_CORRECT_REMOVE_UNCERTAIN"
VALID_BINARY = {"Y", "N", "UNCERTAIN"}
VALID_ACTIONS = {"KEEP", "CORRECT", "REMOVE", "UNCERTAIN"}
VALID_CLEANING_OPERATIONS = {
    "REMOVE_EXACT_DUPLICATE",
    "CORRECT_TRIPLE",
    "REMOVE_INVALID_TRIPLE",
    "MERGE_SYNONYM",
    "NORMALIZE_RELATION",
    "NORMALIZE_ENTITY_TYPE",
    "ADD_MISSING_TRIPLE",
}
VALID_ERROR_TYPES = {
    "UNSUPPORTED_TRIPLE",
    "WRONG_SUBJECT",
    "WRONG_RELATION",
    "WRONG_OBJECT",
    "WRONG_ENTITY_TYPE",
    "OVERLY_BROAD_OR_NARROW",
    "DUPLICATE",
    "SYNONYM_NOT_MERGED",
    "SELF_LOOP",
    "INCONSISTENT_SCHEMA",
    "MISSING_PROVENANCE",
    "OTHER",
}
ERROR_SPLIT_RE = re.compile(r"[;,, ; |]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the accuracy, consistency, error types and number of revisions of dual-expert extraction of knowledge graphs."
    )
    parser.add_argument("--expert-1", type=Path, default=DEFAULT_EXPERT_1)
    parser.add_argument("--expert-2", type=Path, default=DEFAULT_EXPERT_2)
    parser.add_argument(
        "--adjudicated",
        type=Path,
        help="Optional: Final annotation after adjudicationCSV; Provided for final useprecision and processing quantities.",
    )
    parser.add_argument(
        "--recall-file",
        type=Path,
        help="Optional: Gold standard triplet table of sources item by item, must includematched_by_system_Y_N_UNCERTAIN. ",
    )
    parser.add_argument(
        "--cleaning-log",
        type=Path,
        help="Optional: Full spectrum cleaning log; requiredoperation field.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalize_binary(value: Any) -> str:
    text = str(value or "").strip().upper()
    aliases = {
        "YES": "Y",
        "TRUE": "Y",
        "1": "Y",
        "Correct": "Y",
        "Yes": "Y",
        "NO": "N",
        "FALSE": "N",
        "0": "N",
        "Error": "N",
        "No": "N",
        "Not sure": "UNCERTAIN",
        "Unable to judge": "UNCERTAIN",
        "NA": "UNCERTAIN",
        "N/A": "UNCERTAIN",
    }
    return aliases.get(text, text)


def normalize_action(value: Any) -> str:
    text = str(value or "").strip().upper()
    aliases = {
        "Reserve": "KEEP",
        "correct": "CORRECT",
        "Correction": "CORRECT",
        "Delete": "REMOVE",
        "Not sure": "UNCERTAIN",
    }
    return aliases.get(text, text)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found:{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV The file has no header:{path}")
        rows = [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in reader
            if any(str(value or "").strip() for value in row.values())
        ]
    if not rows:
        raise ValueError(f"Annotation file has no data:{path}")
    return rows


def index_rows(rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        sample_id = row.get(ID_COLUMN, "")
        if not sample_id:
            raise ValueError(f"{path.name} No.{row_number} row missing{ID_COLUMN}")
        if sample_id in indexed:
            raise ValueError(f"{path.name} in{sample_id} Repeat")
        indexed[sample_id] = row
    return indexed


def validate_annotations(rows: Iterable[dict[str, str]], label: str) -> list[str]:
    errors: list[str] = []
    for row in rows:
        sample_id = row.get(ID_COLUMN, "[missing]")
        for column in BINARY_COLUMNS:
            value = normalize_binary(row.get(column, ""))
            if value not in VALID_BINARY:
                errors.append(f"{label} {sample_id}: {column}={value!r}")
        action = normalize_action(row.get(ACTION_COLUMN, ""))
        if action not in VALID_ACTIONS:
            errors.append(f"{label} {sample_id}: {ACTION_COLUMN}={action!r}")
        error_codes = {
            value.strip().upper()
            for value in ERROR_SPLIT_RE.split(row.get(ERROR_COLUMN, ""))
            if value.strip()
        }
        invalid_error_codes = sorted(error_codes - VALID_ERROR_TYPES)
        if invalid_error_codes:
            errors.append(
                f"{label} {sample_id}: Invalid error type{invalid_error_codes}"
            )
        if normalize_binary(row.get(TRIPLE_CORRECT_COLUMN, "")) == "N" and not error_codes:
            errors.append(f"{label} {sample_id}: The error triple must be filled with at least one error type")
        triple_label = normalize_binary(row.get(TRIPLE_CORRECT_COLUMN, ""))
        component_labels = [
            normalize_binary(row.get(column, ""))
            for column in COMPONENT_CORRECTNESS_COLUMNS
        ]
        if triple_label == "Y" and any(value != "Y" for value in component_labels):
            errors.append(
                f"{label} {sample_id}: triple_correct=Y When all component judgments must beY"
            )
        if triple_label == "N" and "N" not in component_labels:
            errors.append(
                f"{label} {sample_id}: triple_correct=N when at least one component judgment must beN"
            )
        if triple_label == "UNCERTAIN" and (
            "N" in component_labels or all(value == "Y" for value in component_labels)
        ):
            errors.append(
                f"{label} {sample_id}: triple_correct=UNCERTAIN Only applicable if there is no clear error and at least one component is judged to beUNCERTAIN"
            )
        expected_actions = {
            "Y": {"KEEP"},
            "N": {"CORRECT", "REMOVE"},
            "UNCERTAIN": {"UNCERTAIN"},
        }
        if triple_label in expected_actions and action not in expected_actions[triple_label]:
            errors.append(
                f"{label} {sample_id}: triple_correct={triple_label} withaction={action} inconsistent"
            )
        if triple_label == "Y" and error_codes:
            errors.append(f"{label} {sample_id}: Correct triplet should not be filled with wrong type")
        if normalize_binary(row.get("source_support_Y_N_UNCERTAIN", "")) == "Y":
            provenance_columns = ("source_document", "source_location", "source_excerpt")
            missing_provenance = [
                column for column in provenance_columns if not row.get(column, "").strip()
            ]
            if missing_provenance:
                errors.append(
                    f"{label} {sample_id}: source_support=Y but the source field is missing{missing_provenance}"
                )
        if action == "CORRECT":
            corrected_columns = (
                "corrected_sub_name",
                "corrected_sub_type",
                "corrected_rel_type",
                "corrected_rel_name",
                "corrected_obj_name",
                "corrected_obj_type",
            )
            if not any(row.get(column, "").strip() for column in corrected_columns):
                errors.append(
                    f"{label} {sample_id}: action=CORRECT Fill in at least onecorrected Field"
                )
    return errors


def wilson_interval(correct: int, evaluated: int, z: float = 1.95996398454) -> dict[str, float | None]:
    if evaluated == 0:
        return {"lower": None, "upper": None}
    proportion = correct / evaluated
    denominator = 1 + z * z / evaluated
    centre = (proportion + z * z / (2 * evaluated)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / evaluated
            + z * z / (4 * evaluated * evaluated)
        )
        / denominator
    )
    return {"lower": centre - half_width, "upper": centre + half_width}


def precision_summary(rows: Iterable[dict[str, str]], column: str) -> dict[str, Any]:
    labels = [normalize_binary(row.get(column, "")) for row in rows]
    correct = labels.count("Y")
    incorrect = labels.count("N")
    uncertain = labels.count("UNCERTAIN")
    evaluated = correct + incorrect
    interval = wilson_interval(correct, evaluated)
    return {
        "reviewed": len(labels),
        "evaluated_Y_or_N": evaluated,
        "correct_Y": correct,
        "incorrect_N": incorrect,
        "uncertain": uncertain,
        "precision": correct / evaluated if evaluated else None,
        "wilson_95_ci_lower": interval["lower"],
        "wilson_95_ci_upper": interval["upper"],
    }


def cohens_kappa(labels_1: list[str], labels_2: list[str]) -> dict[str, Any]:
    if len(labels_1) != len(labels_2):
        raise ValueError("The two sets of consistency labels have different lengths")
    if not labels_1:
        return {"n": 0, "agreement": None, "kappa": None, "note": "No comparison tags"}

    categories = sorted(set(labels_1) | set(labels_2))
    observed = sum(a == b for a, b in zip(labels_1, labels_2)) / len(labels_1)
    counts_1 = Counter(labels_1)
    counts_2 = Counter(labels_2)
    expected = sum(
        (counts_1[category] / len(labels_1))
        * (counts_2[category] / len(labels_2))
        for category in categories
    )
    if math.isclose(expected, 1.0):
        kappa = None
        note = "Both experts used only one category,kappa cannot be defined"
    else:
        kappa = (observed - expected) / (1 - expected)
        note = ""
    return {"n": len(labels_1), "agreement": observed, "kappa": kappa, "note": note}


def agreement_summary(
    expert_1: dict[str, dict[str, str]],
    expert_2: dict[str, dict[str, str]],
    column: str,
) -> dict[str, Any]:
    all_labels_1: list[str] = []
    all_labels_2: list[str] = []
    labels_1: list[str] = []
    labels_2: list[str] = []
    excluded_uncertain = 0
    for sample_id in sorted(expert_1):
        label_1 = normalize_binary(expert_1[sample_id].get(column, ""))
        label_2 = normalize_binary(expert_2[sample_id].get(column, ""))
        all_labels_1.append(label_1)
        all_labels_2.append(label_2)
        if "UNCERTAIN" in {label_1, label_2}:
            excluded_uncertain += 1
            continue
        labels_1.append(label_1)
        labels_2.append(label_2)
    all_category_result = cohens_kappa(all_labels_1, all_labels_2)
    all_category_result["decisive_binary_only"] = cohens_kappa(labels_1, labels_2)
    all_category_result["excluded_from_decisive_binary"] = excluded_uncertain
    return all_category_result


def disagreement_rows(
    expert_1: dict[str, dict[str, str]], expert_2: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    disagreements: list[dict[str, Any]] = []
    for sample_id in sorted(expert_1):
        row_1 = expert_1[sample_id]
        row_2 = expert_2[sample_id]
        differing_columns = [
            column
            for column in BINARY_COLUMNS + [ACTION_COLUMN]
            if (
                normalize_binary(row_1.get(column, ""))
                if column in BINARY_COLUMNS
                else normalize_action(row_1.get(column, ""))
            )
            != (
                normalize_binary(row_2.get(column, ""))
                if column in BINARY_COLUMNS
                else normalize_action(row_2.get(column, ""))
            )
        ]
        if row_1.get(ERROR_COLUMN, "").strip().upper() != row_2.get(
            ERROR_COLUMN, ""
        ).strip().upper():
            differing_columns.append(ERROR_COLUMN)
        if differing_columns:
            disagreements.append(
                {
                    "sample_id": sample_id,
                    "sub_name": row_1.get("sub_name", ""),
                    "rel_name": row_1.get("rel_name", ""),
                    "obj_name": row_1.get("obj_name", ""),
                    "differing_columns": ";".join(differing_columns),
                    "expert_1_triple_correct": normalize_binary(
                        row_1.get(TRIPLE_CORRECT_COLUMN, "")
                    ),
                    "expert_2_triple_correct": normalize_binary(
                        row_2.get(TRIPLE_CORRECT_COLUMN, "")
                    ),
                    "expert_1_action": normalize_action(row_1.get(ACTION_COLUMN, "")),
                    "expert_2_action": normalize_action(row_2.get(ACTION_COLUMN, "")),
                    "adjudicated_triple_correct_Y_N_UNCERTAIN": "",
                    "adjudicated_action_KEEP_CORRECT_REMOVE_UNCERTAIN": "",
                    "adjudicated_error_types_semicolon_separated": "",
                    "adjudication_notes": "",
                }
            )
    return disagreements


def adjudication_template_rows(
    expert_1: dict[str, dict[str, str]], expert_2: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    """Create a complete final-decision form that can be passed back to this script."""
    rows: list[dict[str, Any]] = []
    corrected_columns = [
        "corrected_sub_name",
        "corrected_sub_type",
        "corrected_rel_type",
        "corrected_rel_name",
        "corrected_obj_name",
        "corrected_obj_type",
    ]
    for sample_id in sorted(expert_1):
        row_1 = expert_1[sample_id]
        row_2 = expert_2[sample_id]
        output: dict[str, Any] = dict(row_1)
        for column in BINARY_COLUMNS + [ERROR_COLUMN, ACTION_COLUMN, "notes"]:
            output[column] = ""
        for column in corrected_columns:
            output[column] = ""
        output.update(
            {
                "expert_1_triple_correct": normalize_binary(
                    row_1.get(TRIPLE_CORRECT_COLUMN, "")
                ),
                "expert_2_triple_correct": normalize_binary(
                    row_2.get(TRIPLE_CORRECT_COLUMN, "")
                ),
                "expert_1_action": normalize_action(row_1.get(ACTION_COLUMN, "")),
                "expert_2_action": normalize_action(row_2.get(ACTION_COLUMN, "")),
                "expert_1_error_types": row_1.get(ERROR_COLUMN, ""),
                "expert_2_error_types": row_2.get(ERROR_COLUMN, ""),
            }
        )
        rows.append(output)
    return rows


def error_type_counts(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        if normalize_binary(row.get(TRIPLE_CORRECT_COLUMN, "")) != "N":
            continue
        raw = row.get(ERROR_COLUMN, "")
        for value in ERROR_SPLIT_RE.split(raw):
            code = value.strip().upper()
            if code:
                counter[code] += 1
    return [
        {"error_type": code, "count": count}
        for code, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def action_counts(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counter = Counter(normalize_action(row.get(ACTION_COLUMN, "")) for row in rows)
    return {action: counter[action] for action in sorted(VALID_ACTIONS)}


def cleaning_log_summary(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    if "operation" not in rows[0]:
        raise ValueError("Cleaning log is missingoperation Field")
    completed_rows = [row for row in rows if row.get("operation", "").strip()]
    counts = Counter(row["operation"].strip().upper() for row in completed_rows)
    invalid_operations = sorted(set(counts) - VALID_CLEANING_OPERATIONS)
    if invalid_operations:
        raise ValueError(
            f"Cleaning log contains invalidoperation: {invalid_operations}; "
            f"Allowed values are:{sorted(VALID_CLEANING_OPERATIONS)}"
        )
    return {
        "logged_changes": len(completed_rows),
        "operation_counts": dict(sorted(counts.items())),
        "note": "These numbers are from item-by-item cleaning logs and should not be confused with treatment recommendations from sampling.",
    }


def recall_summary(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    required = {
        "source_unit_id",
        "gold_triple_id",
        "matched_by_system_Y_N_UNCERTAIN",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Recall File is missing fields:{sorted(missing)}")

    matched = 0
    missed = 0
    uncertain = 0
    source_units: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        source_unit_id = row["source_unit_id"]
        gold_triple_id = row["gold_triple_id"]
        if not source_unit_id or not gold_triple_id:
            raise ValueError(
                f"Recall Document no.{row_number} row missingsource_unit_id orgold_triple_id"
            )
        label = normalize_binary(row["matched_by_system_Y_N_UNCERTAIN"])
        if label not in VALID_BINARY:
            raise ValueError(f"Recall Document no.{row_number} Invalid row matching judgment:{label!r}")
        source_units.add(source_unit_id)
        if label == "Y":
            matched += 1
        elif label == "N":
            missed += 1
        else:
            uncertain += 1

    evaluated = matched + missed
    interval = wilson_interval(matched, evaluated)
    return {
        "reviewed_source_units": len(source_units),
        "gold_triples_total": len(rows),
        "gold_triples_evaluable": evaluated,
        "matched_gold_triples": matched,
        "missed_gold_triples": missed,
        "uncertain_gold_triples": uncertain,
        "micro_recall": matched / evaluated if evaluated else None,
        "wilson_95_ci_lower": interval["lower"],
        "wilson_95_ci_upper": interval["upper"],
        "note": "Recall This is true only when source units are independently randomly sampled and experts blindly exhaustively exhaust gold standard triplets.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value: float | None) -> str:
    return "NA" if value is None else f"{value:.1%}"


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Knowledge graph extraction verification results",
        "",
        "## Independent review by two experts",
        "",
        "| indicator| expert1 | expert2 |",
        "|---|---:|---:|",
    ]
    for column in BINARY_COLUMNS:
        metric_1 = report["expert_1"][column]
        metric_2 = report["expert_2"][column]
        lines.append(
            f"| {column} | {percent(metric_1['precision'])} "
            f"({metric_1['evaluated_Y_or_N']}/{metric_1['reviewed']}) | "
            f"{percent(metric_2['precision'])} "
            f"({metric_2['evaluated_Y_or_N']}/{metric_2['reviewed']}) |"
        )

    lines.extend(
        [
            "",
            "## Inter-expert agreement",
            "",
            "| Judgment item| All samples| Three-category original agreement rate| Three categoriesCohen's kappa |",
            "|---|---:|---:|---:|",
        ]
    )
    for column, metric in report["inter_annotator_agreement"].items():
        kappa = "NA" if metric["kappa"] is None else f"{metric['kappa']:.3f}"
        lines.append(
            f"| {column} | {metric['n']} | {percent(metric['agreement'])} | {kappa} |"
        )

    if report.get("adjudicated"):
        final_metric = report["adjudicated"][TRIPLE_CORRECT_COLUMN]
        lines.extend(
            [
                "",
                "## final result after ruling",
                "",
                f"The strict triplet accuracy is{percent(final_metric['precision'])} "
                f"({final_metric['correct_Y']}/{final_metric['evaluated_Y_or_N']}), "
                f"Wilson 95% CI {percent(final_metric['wilson_95_ci_lower'])}–"
                f"{percent(final_metric['wilson_95_ci_upper'])}. ",
                "",
                "Suggestions for processing during sampling:"
                f"Reserve{report['adjudicated_actions']['KEEP']} Article,"
                f"correct{report['adjudicated_actions']['CORRECT']} Article,"
                f"Delete{report['adjudicated_actions']['REMOVE']} Article,"
                f"Not sure{report['adjudicated_actions']['UNCERTAIN']} article.",
            ]
        )
        if report["adjudicated_error_types"]:
            lines.extend(
                [
                    "",
                    "Main error types:"
                    + "; ".join(
                        f"{row['error_type']} ({row['count']})"
                        for row in report["adjudicated_error_types"]
                    )
                    + ". ",
                ]
            )
    else:
        lines.extend(
            [
                "",
                "## Not yet decided",
                "",
                "No adjudication files are currently available, so final accuracy, major error types, or final corrections cannot be reported/Delete quantity. Please deal with it first`disagreements_for_adjudication.csv`. ",
            ]
        )

    if report.get("full_kg_cleaning_log"):
        cleaning = report["full_kg_cleaning_log"]
        operations = "; ".join(
            f"{operation} {count} Article"
            for operation, count in cleaning["operation_counts"].items()
        )
        lines.extend(
            [
                "",
                "## Full spectrum cleaning results",
                "",
                f"Total cleaning log records{cleaning['logged_changes']} Changes:{operations}. ",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Full spectrum cleaning results",
                "",
                "No item-by-item cleaning log is provided, so the number of synonyms merged, corrected, and deleted for the full map cannot be reported.",
            ]
        )

    if report.get("recall"):
        recall = report["recall"]
        lines.extend(
            [
                "",
                "## Recall",
                "",
                f"in{recall['reviewed_source_units']} In independent source units,"
                f"System matching{recall['matched_gold_triples']}/"
                f"{recall['gold_triples_evaluable']} Bars can determine the gold standard triplet,"
                f"micro-recall for{percent(recall['micro_recall'])} "
                f"(Wilson 95% CI: {percent(recall['wilson_95_ci_lower'])}–"
                f"{percent(recall['wilson_95_ci_upper'])}); Also"
                f"{recall['uncertain_gold_triples']} The article cannot be determined.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Recall",
                "",
                "A gold standard for independently sourced units is not provided and therefore is not reportedrecall. It cannot be replaced by the accuracy of the extracted triples.recall. ",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows_1 = read_csv(args.expert_1.resolve())
    rows_2 = read_csv(args.expert_2.resolve())
    expert_1 = index_rows(rows_1, args.expert_1)
    expert_2 = index_rows(rows_2, args.expert_2)
    if set(expert_1) != set(expert_2):
        missing_in_1 = sorted(set(expert_2) - set(expert_1))
        missing_in_2 = sorted(set(expert_1) - set(expert_2))
        raise ValueError(
            f"two expertssample_id inconsistent; expert1missing{missing_in_1}; "
            f"expert2missing{missing_in_2}"
        )

    validation_errors = validate_annotations(expert_1.values(), "expert1")
    validation_errors.extend(validate_annotations(expert_2.values(), "expert2"))
    if validation_errors:
        preview = "\n".join(validation_errors[:20])
        remainder = len(validation_errors) - min(20, len(validation_errors))
        suffix = f"\nAlso{remainder} Item error" if remainder else ""
        raise ValueError(f"The annotation value is incomplete or does not meet the controlled value:\n{preview}{suffix}")

    report: dict[str, Any] = {
        "reviewed_sample_size_per_expert": len(expert_1),
        "expert_1": {
            column: precision_summary(expert_1.values(), column)
            for column in BINARY_COLUMNS
        },
        "expert_2": {
            column: precision_summary(expert_2.values(), column)
            for column in BINARY_COLUMNS
        },
        "inter_annotator_agreement": {
            column: agreement_summary(expert_1, expert_2, column)
            for column in BINARY_COLUMNS
        },
    }

    disagreements = disagreement_rows(expert_1, expert_2)
    adjudication_rows = adjudication_template_rows(expert_1, expert_2)
    report["disagreement_sample_count"] = len(disagreements)

    if args.adjudicated:
        adjudicated_rows = read_csv(args.adjudicated.resolve())
        adjudicated = index_rows(adjudicated_rows, args.adjudicated)
        if set(adjudicated) != set(expert_1):
            raise ValueError("The ruling document must contain all the same information as both expertssample_id")
        errors = validate_annotations(adjudicated.values(), "ruling")
        if errors:
            raise ValueError("The verdict file has invalid values:\n" + "\n".join(errors[:20]))
        report["adjudicated"] = {
            column: precision_summary(adjudicated.values(), column)
            for column in BINARY_COLUMNS
        }
        report["adjudicated_actions"] = action_counts(adjudicated.values())
        report["adjudicated_error_types"] = error_type_counts(adjudicated.values())

    if args.recall_file:
        report["recall"] = recall_summary(args.recall_file.resolve())
    if args.cleaning_log:
        report["full_kg_cleaning_log"] = cleaning_log_summary(
            args.cleaning_log.resolve()
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "disagreements_for_adjudication.csv", disagreements)
    write_csv(output_dir / "adjudication_template.csv", adjudication_rows)
    (output_dir / "kg_validation_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "kg_validation_metrics.md").write_text(
        build_markdown(report), encoding="utf-8"
    )
    print(f"Verification metrics have been generated:{output_dir}")


if __name__ == "__main__":
    main()
