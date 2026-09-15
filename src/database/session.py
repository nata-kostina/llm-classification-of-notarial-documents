from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _session_factory
    if _engine is None:
        _engine = create_engine(get_settings().database_url, connect_args={"connect_timeout": 5})
        try:
            with _engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except OperationalError as e:
            raise ConnectionError("Could not establish connection to the Database") from e

        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)

    return _engine


@contextmanager
def get_session() -> Generator[Session]:
    if _session_factory is None:
        get_engine()

    assert _session_factory is not None

    session = _session_factory()
    try:
        yield session
    finally:
        session.close()
