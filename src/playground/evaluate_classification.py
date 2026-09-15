from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from src.config import get_settings
from src.evaluation.models import (
    DocumentClassificationResult,
    parse_optional_str,
    parse_rules_list,
)

RUN_ID = "24864a0343ce46b9ad5218201252b6fc"
GOLDEN_DATASET = Path(r".\data\GOLDEN_DATASET.csv")


def main():
    settings = get_settings()

    matches = list(Path(settings.mlflow_artifact_store).rglob(f"{RUN_ID}/**/records.csv"))

    artifact_file = matches[0]

    df_classification_results = pd.read_csv(artifact_file, encoding="utf-8", dtype={"act_id": str})
    df_golden_dataset = pd.read_csv(GOLDEN_DATASET, encoding="utf-8", dtype={"act_id": str})

    df_merged = pd.merge(
        df_golden_dataset,
        df_classification_results[["act_id", "data.label", "data.rules"]],
        how="left",
        on="act_id",
        validate="one_to_one",
    )

    df_merged = df_merged.rename(
        columns={
            "label": "label_true",
            "rules": "rules_true",
            "data.label": "label_pred",
            "data.rules": "rules_pred",
        }
    )

    records: list[DocumentClassificationResult] = []

    for row in df_merged.to_dict(orient="records"):
        record = DocumentClassificationResult(
            act_id=str(row["act_id"]),
            label_true=str(row["label_true"]),
            rules_true=parse_rules_list(row.get("rules_true")) or [],
            label_pred=parse_optional_str(row.get("label_pred")),
            rules_pred=parse_rules_list(row.get("rules_pred")),
        )
        records.append(record)

    y_true = [r.label_true for r in records]

    y_pred = [(r.label_pred if (r.label_pred is not None and str(r.label_pred) != "nan") else "ERROR") for r in records]

    dict_report: dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)  # type: ignore

    str_report = classification_report(y_true, y_pred, zero_division=0)

    values = {
        "accuracy": float(dict_report["accuracy"]),
        "macro_precision": float(dict_report["macro avg"]["precision"]),
        "macro_recall": float(dict_report["macro avg"]["recall"]),
        "macro_f1": float(dict_report["macro avg"]["f1-score"]),
        "weighted_precision": float(dict_report["weighted avg"]["precision"]),
        "weighted_recall": float(dict_report["weighted avg"]["recall"]),
        "weighted_f1": float(dict_report["weighted avg"]["f1-score"]),
        "classification_report": str_report,
    }


if __name__ == "__main__":
    main()
