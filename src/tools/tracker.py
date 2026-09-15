import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from pydantic import BaseModel

from src.config import get_settings
from src.pipeline.base import PipelineContext
from src.tools.loggers import logger

RULES_COLUMN = "data.rules"

mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)


def init_experiment(experiment_name: str):
    mlflow.set_experiment(experiment_name)


@contextmanager
def track_run(
    *,
    experiment_name: str,
    params: dict[str, float] | None,
    run_name: str | None = None,
) -> Generator[mlflow.ActiveRun, Any]:
    init_experiment(experiment_name)

    if not run_name:
        run_name = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)

        yield run


def _log_metrics(field_name: str, val: Any) -> None:
    if isinstance(val, BaseModel):
        mlflow.log_metrics(val.model_dump())
    elif isinstance(val, dict):
        mlflow.log_metrics(val)
    elif isinstance(val, int | float):
        mlflow.log_metric(field_name, val)


def _log_dictionary(field_name: str, val: Any) -> None:
    if isinstance(val, BaseModel):
        data = val.model_dump(mode="json")
    elif isinstance(val, list):
        data = [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in val]
    else:
        data = val
    mlflow.log_dict(data, artifact_file=f"{field_name}.json")  # type: ignore


def _log_artifact(val: Any) -> None:
    if isinstance(val, list | tuple):
        for item in val:
            mlflow.log_artifact(str(item))
    elif isinstance(val, str | Path):
        mlflow.log_artifact(str(val))


def _log_csv(field_name: str, val: Any) -> None:
    if isinstance(val, list):
        raw_data = [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in val]
    elif isinstance(val, BaseModel):
        raw_data = [val.model_dump(mode="json")]
    else:
        raw_data = val

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / f"{field_name}.csv"
        df = pd.json_normalize(raw_data)

        if RULES_COLUMN in df.columns:
            df[RULES_COLUMN] = df[RULES_COLUMN].apply(lambda r: ", ".join(r) if isinstance(r, list) else r)

        df.to_csv(csv_path, index=False, encoding="utf-8")
        mlflow.log_artifact(str(csv_path))


HANDLERS = {
    "metrics": lambda name, val: _log_metrics(name, val),
    "dictionary": lambda name, val: _log_dictionary(name, val),
    "artifact": lambda name, val: _log_artifact(val),
    "csv": lambda name, val: _log_csv(name, val),
}


def log_mlflow_results(context: PipelineContext) -> None:

    def walk(obj: Any) -> None:
        if not isinstance(obj, BaseModel):
            return

        for field_name, field_info in type(obj).model_fields.items():
            val = getattr(obj, field_name, None)
            if val is None:
                continue

            extra = field_info.json_schema_extra
            mlflow_type = extra.get("mlflow_type") if isinstance(extra, dict) else None

            if isinstance(mlflow_type, str):
                handler = HANDLERS.get(mlflow_type)
                if handler:
                    handler(field_name, val)

            if isinstance(val, BaseModel):
                walk(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, BaseModel):
                        walk(item)

    walk(context)


def log_run_info(run: mlflow.ActiveRun, ui_base_url: str = "http://127.0.0.1:5000") -> None:
    exp_id = run.info.experiment_id
    run_id = run.info.run_id
    base_url = ui_base_url.rstrip("/")

    logger.info(f"Run '{run.info.run_name}' finished successfully!")
    logger.info(f"Run ID: {run.info.run_id}")
    logger.info(f"Open in MLflow UI: {base_url}/#/experiments/{exp_id}/runs/{run_id}")
