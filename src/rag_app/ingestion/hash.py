import hashlib
from pathlib import Path


def hash_file(file_path: str | Path) -> str:
    """
    Returns SHA-256 hash of file bytes.
    Same file content means same hash.
    This enables idempotent ingestion.
    """
    path = Path(file_path)
    sha = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block) #weight accumulate

    return sha.hexdigest()
