"""One reentrant process context for the project's NFS submission lock."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

HELD_SUBMISSION_LOCKS: ContextVar[frozenset[Path]] = ContextVar(
    "held_submission_locks", default=frozenset()
)


@contextmanager
def submission_lock(root: Path) -> Iterator[None]:
    path = root.resolve() / ".submission.lock"
    held = HELD_SUBMISSION_LOCKS.get()
    if path in held:
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        token = HELD_SUBMISSION_LOCKS.set(held | {path})
        try:
            yield
        finally:
            HELD_SUBMISSION_LOCKS.reset(token)
