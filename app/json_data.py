"""Values written to the application logs and supporting JSON reports."""

from __future__ import annotations

from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
