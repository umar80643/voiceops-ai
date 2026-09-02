"""Object storage abstraction.

Provides a local-filesystem implementation for development/tests, and the
interface a boto3-backed S3/MinIO implementation would satisfy in
production. Keeping this behind a Protocol lets services swap backends
without code changes (see ADR-0xx in docs/decisions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...


class LocalDiskObjectStore:
    """Development/test object store backed by the local filesystem.

    Mirrors the interface an S3/MinIO-backed store would expose so
    swapping implementations later requires no call-site changes.
    """

    def __init__(self, root: str = "data/raw") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()


# TODO(production): implement S3ObjectStore using boto3 against
# settings.s3_endpoint_url / s3_bucket / s3_access_key / s3_secret_key.
# Not implemented here because it requires network access to a real
# S3/MinIO endpoint to validate; the Protocol above ensures it's a
# drop-in replacement when added.
