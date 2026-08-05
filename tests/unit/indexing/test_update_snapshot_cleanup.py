# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

import re

import pytest
from graphrag.index.run.run_pipeline import _cleanup_update_snapshots


class _SnapshotStorage:
    def __init__(
        self, keys: list[str], cleared: list[str], namespace: str = ""
    ) -> None:
        self._keys = keys
        self._cleared = cleared
        self._namespace = namespace

    def find(self, _: re.Pattern[str]):
        return iter(self._keys)

    def child(self, name: str) -> "_SnapshotStorage":
        return _SnapshotStorage(self._keys, self._cleared, name)

    async def clear(self) -> None:
        self._cleared.append(self._namespace)


class _SnapshotTableProvider:
    def __init__(self, cleared: list[str], namespace: str = "") -> None:
        self._cleared = cleared
        self._namespace = namespace

    def child(self, name: str) -> "_SnapshotTableProvider":
        return _SnapshotTableProvider(self._cleared, name)

    async def clear(self) -> None:
        self._cleared.append(self._namespace)


@pytest.mark.asyncio
async def test_cleanup_update_snapshots_keeps_most_recent_runs() -> None:
    storage_cleared: list[str] = []
    table_cleared: list[str] = []
    storage = _SnapshotStorage(
        [
            "20260101-000000/delta/stats.json",
            "20260102-000000/delta/stats.json",
            "20260103-000000/delta/stats.json",
            "unrelated/file.txt",
        ],
        storage_cleared,
    )
    provider = _SnapshotTableProvider(table_cleared)

    await _cleanup_update_snapshots(storage, provider, retention_count=2)  # type: ignore[arg-type]

    assert storage_cleared == ["20260101-000000"]
    assert table_cleared == ["20260101-000000"]
