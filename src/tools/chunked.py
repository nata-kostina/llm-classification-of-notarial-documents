from collections.abc import Generator
from typing import TypeVar

T = TypeVar("T")


def chunked[T](iterable: list[T], size: int) -> Generator[list[T]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]
