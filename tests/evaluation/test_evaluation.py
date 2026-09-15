from pathlib import Path
import re
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.config import Settings
from src.evaluation.evaluation import Evaluator
from src.evaluation.models import DocumentClassificationResult, EvaluationMetrics, EvaluationResult


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner.run.return_value = EvaluationMetrics(
        accuracy=1.0
    )
    return runner


@pytest.fixture
def test_settings(tmp_path: Path):
    artifact_dir = tmp_path / "mlflow_artifacts"
    artifact_dir.mkdir()
    return Settings.model_construct(mlflow_artifact_store=str(artifact_dir))

def test_evaluator_raises_file_not_found(mock_runner, test_settings, tmp_path: Path):
    golden_path = tmp_path / "golden.csv"
    golden_path.touch()

    evaluator = Evaluator(
        run_id="run_123",
        evaluation_runner=mock_runner,
        settings=test_settings,
    )

    with pytest.raises(FileNotFoundError, match="File records.csv not found"):
        evaluator.run(golden_dataset=golden_path)


def test_evaluator_run_success(mock_runner, test_settings, tmp_path: Path):
    run_id = "run_123"
    artifact_name = "records.csv"

    run_dir = Path(test_settings.mlflow_artifact_store) / run_id / "artifacts"
    run_dir.mkdir(parents=True)
    
    df_pred = pd.DataFrame({
        "act_id": ["101", "102"],
        "data.label": ["cat", "dog"],
        "data.rules": ["rule1, rule2", None],
    })
    df_pred.to_csv(run_dir / artifact_name, index=False)


    df_golden = pd.DataFrame({
        "act_id": ["101", "102", "103"],
        "label": ["cat", "dog", "bird"],
        "rules": ["rule1", "rule3", "rule4"],
    })
    golden_path = tmp_path / "golden.csv"
    df_golden.to_csv(golden_path, index=False)

    evaluator = Evaluator(
        run_id=run_id,
        evaluation_runner=mock_runner,
        settings=test_settings,
    )
    result = evaluator.run(golden_dataset=golden_path)

    assert mock_runner.run.call_count == 1

    passed_records: list[DocumentClassificationResult] = mock_runner.run.call_args[0][0]
    assert len(passed_records) == 3

    assert passed_records[0].act_id == "101"
    assert passed_records[0].label_true == "cat"
    assert passed_records[0].label_pred == "cat"
    assert passed_records[0].rules_true == ["rule1"]
    assert passed_records[0].rules_pred == ["rule1", "rule2"]

    assert passed_records[2].act_id == "103"
    assert passed_records[2].label_pred is None
    assert passed_records[2].rules_pred == [] 

    assert isinstance(result, EvaluationResult)
    assert result.metrics.accuracy == 1.0
    
@pytest.fixture
def valid_pred_artifact(tmp_path: Path, test_settings):
    run_dir = Path(test_settings.mlflow_artifact_store) / "run_123" / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    df_pred = pd.DataFrame({
        "act_id": ["101"],
        "data.label": ["cat"],
        "data.rules": ["rule1"],
    })
    artifact_path = run_dir / "records.csv"
    df_pred.to_csv(artifact_path, index=False)
    return artifact_path\
        
def test_evaluator_golden_dataset_does_not_exist(mock_runner, test_settings, tmp_path: Path):
    non_existent_golden = tmp_path / "missing_golden.csv"

    evaluator = Evaluator(
        run_id="run_123",
        evaluation_runner=mock_runner,
        settings=test_settings,
    )

    expected_error = re.escape(f"File {non_existent_golden} not found")

    with pytest.raises(FileNotFoundError, match=expected_error):
        evaluator.run(golden_dataset=non_existent_golden)
        

def test_evaluator_golden_dataset_missing_columns(
    mock_runner, test_settings, tmp_path: Path, valid_pred_artifact
):
    bad_golden = tmp_path / "bad_golden.csv"
    pd.DataFrame({"wrong_column": [1], "label": ["cat"], "rules": ["1.1.1"]}).to_csv(bad_golden, index=False)

    evaluator = Evaluator(
        run_id="run_123",
        evaluation_runner=mock_runner,
        settings=test_settings,
    )

    with pytest.raises(ValueError, match="Golden dataset is missing required columns"):
        evaluator.run(golden_dataset=bad_golden)
        

def test_evaluator_prediction_artifact_missing_columns(
    mock_runner, test_settings, tmp_path: Path
):
    run_dir = Path(test_settings.mlflow_artifact_store) / "run_123" / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    bad_pred = pd.DataFrame({"act_id": ["101"], "wrong_data": ["abc"]})
    bad_pred.to_csv(run_dir / "records.csv", index=False)

    golden = tmp_path / "golden.csv"
    pd.DataFrame({"act_id": ["101"], "label": ["cat"], "rules": ["1.1.1"]}).to_csv(golden, index=False)

    evaluator = Evaluator(
        run_id="run_123",
        evaluation_runner=mock_runner,
        settings=test_settings,
    )

    with pytest.raises(ValueError, match="Prediction artifact is missing required columns"):
        evaluator.run(golden_dataset=golden)