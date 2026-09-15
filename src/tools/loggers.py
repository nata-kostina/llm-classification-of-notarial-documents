import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from src.config import get_settings

settings = get_settings()

logger.remove()

logger.level("INFO", color="<green>")
logger.level("WARNING", color="<yellow>")
logger.level("ERROR", color="<red>")

logger.add(
    sys.stderr,
    level="INFO",
    format="<level>[{level}] {time:DD-MM-YYYY HH:mm:ss} {message}</level>",
    colorize=True,
    filter=lambda record: "json_record" not in record["extra"],
)

logger.add(
    settings.log_file,
    level="INFO",
    format="[{level}] {time:DD-MM-YYYY HH:mm:ss} {message}",
    encoding="utf-8",
    filter=lambda record: "json_record" not in record["extra"],
)


def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        result = func(*args, **kwargs)

        execution_time = time.perf_counter() - start

        logger.info(f"Function '{func.__name__}' completed in {execution_time:.2f}s")

        return result

    return wrapper


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class JsonlLogger:
    def __init__(self, log_dir: Path | None = None, prefix: str = "classification"):
        self.log_dir = log_dir or (PROJECT_ROOT / "logs")
        self.prefix = prefix
        self._file = None

    def __enter__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
        filepath = self.log_dir / f"{self.prefix}_{timestamp}.jsonl"
        self._file = open(filepath, "a", encoding="utf-8")
        return self

    def log(self, record: BaseModel | dict) -> None:
        if isinstance(record, BaseModel):
            data = record.model_dump_json()
        else:
            import json

            data = json.dumps(record, ensure_ascii=False)

        if self._file:
            self._file.write(data + "\n")
            self._file.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
