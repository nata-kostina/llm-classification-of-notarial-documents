from pathlib import Path

import pandas as pd

from src.config import Settings, get_settings
from src.evaluation.evaluation_builder import EvaluationRunner
from src.evaluation.models import (
    DocumentClassificationResult,
    EvaluationResult,
    parse_optional_str,
    parse_rules_list,
)
from src.pipeline.base import PipelineContext, PipelineStep


class Evaluator:
    REQUIRED_GOLDEN_COLUMNS = {"act_id", "label", "rules"}
    REQUIRED_PRED_COLUMNS = {"act_id", "data.label", "data.rules"}

    def __init__(
        self,
        run_id: str,
        evaluation_runner: EvaluationRunner,
        settings: Settings | None = None,
        artifact_name: str = "records.csv",
    ) -> None:
        self.run_id = run_id
        self.artifact_name = artifact_name
        self.evaluation_runner = evaluation_runner

        resolved_settings = settings or get_settings()
        self.mlflow_artifact_store = Path(resolved_settings.mlflow_artifact_store)

    def run(self, *, golden_dataset: Path) -> EvaluationResult:
        if not golden_dataset.exists():
            raise FileNotFoundError(f"File {golden_dataset} not found")

        artifact_file = self._resolve_artifact_path()
        df_merged = self._load_and_merge_data(golden_dataset, artifact_file)

        records = [
            DocumentClassificationResult(
                act_id=str(row["act_id"]),
                label_true=str(row["label_true"]),
                rules_true=parse_rules_list(row.get("rules_true")) or [],
                label_pred=parse_optional_str(row.get("label_pred")),
                rules_pred=parse_rules_list(row.get("rules_pred")),
            )
            for row in df_merged.to_dict(orient="records")
        ]

        metrics_results = self.evaluation_runner.run(records)

        return EvaluationResult(data=records, metrics=metrics_results)

    def _resolve_artifact_path(self) -> Path:
        matches = list(self.mlflow_artifact_store.rglob(f"{self.run_id}/**/{self.artifact_name}"))
        if not matches:
            raise FileNotFoundError(f"File {self.artifact_name} not found for run ID {self.run_id}")
        return matches[0]

    def _validate_columns(self, df: pd.DataFrame, required: set[str], dataset_name: str) -> None:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{dataset_name} is missing required columns: {missing}")

    def _load_and_merge_data(self, golden_path: Path, artifact_path: Path) -> pd.DataFrame:
        df_pred = pd.read_csv(artifact_path, encoding="utf-8", dtype={"act_id": str})
        df_golden = pd.read_csv(golden_path, encoding="utf-8", dtype={"act_id": str})

        self._validate_columns(df_golden, self.REQUIRED_GOLDEN_COLUMNS, "Golden dataset")
        self._validate_columns(df_pred, self.REQUIRED_PRED_COLUMNS, "Prediction artifact")

        df_merged = pd.merge(
            df_golden,
            df_pred[["act_id", "data.label", "data.rules"]],
            how="left",
            on="act_id",
            validate="one_to_one",
        )

        return df_merged.rename(
            columns={
                "label": "label_true",
                "rules": "rules_true",
                "data.label": "label_pred",
                "data.rules": "rules_pred",
            }
        )


class EvaluationStep(PipelineStep):
    name = "evaluation"

    def __init__(self, evaluator: Evaluator) -> None:
        self._evaluator = evaluator

    def run(self, ctx: PipelineContext) -> PipelineContext:
        evaluation = self._evaluator.run(golden_dataset=ctx.documents_path)

        return ctx.model_copy(update={"evaluation": evaluation})
