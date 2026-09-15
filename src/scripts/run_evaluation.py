import argparse
from pathlib import Path

from src.config import get_settings
from src.evaluation.evaluation import EvaluationStep, Evaluator
from src.evaluation.evaluation_builder import EvaluationSuiteBuilder
from src.pipeline.base import Pipeline
from src.tools.loggers import logger
from src.tools.tracker import log_mlflow_results, log_run_info, track_run


def _build_parser():
    parser = argparse.ArgumentParser(description="Evaluate classification results.")
    parser.add_argument(
        "--data-src",
        type=Path,
        required=True,
        help="Path to the golden dataset.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        type=str,
        help="Run ID to use for evaluation.",
    )
    parser.add_argument(
        "--log-mlflow",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to log results to MLflow",
    )

    parser.add_argument("--eval-classification", action="store_true", default=True)
    parser.add_argument("--eval-rules", action="store_true", default=True)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    data_src, run_id, log_mlflow = args.data_src, args.run_id, args.log_mlflow

    settings = get_settings()

    builder = EvaluationSuiteBuilder()

    if args.eval_classification:
        builder.add_classification_metrics()

    if args.eval_rules:
        builder.add_rules_metrics()

    evaluator_runner = builder.build()

    evaluator = Evaluator(run_id=run_id, evaluation_runner=evaluator_runner, settings=settings)

    try:
        pipeline = Pipeline([EvaluationStep(evaluator)])

        if log_mlflow:
            params = {"source_run_id": run_id, "golden_dataset": data_src}
            with track_run(experiment_name="Evaluation", params=params) as run:
                context = pipeline.run(documents_path=data_src)

                log_mlflow_results(context)

                log_run_info(run)
        else:
            context = pipeline.run(documents_path=data_src)

    except Exception as error:
        message = str(error) or error.__class__.__name__
        logger.error(message, exc_info=True)


if __name__ == "__main__":
    main()
