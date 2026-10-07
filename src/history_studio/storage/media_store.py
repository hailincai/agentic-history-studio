"""Local immutable binary storage, independent of manifest/workflow publication."""
import hashlib
import os
import tempfile
from pathlib import Path

from history_studio.models.media_package import MediaAssetReference, MediaType


def _resolved_path(path: Path) -> Path:
    # Windows realpath can retain the extended-path prefix if a concurrent writer
    # creates a previously missing component between its native resolution calls.
    # Compare equivalent ordinary/extended spellings without relaxing containment.
    resolved = str(path.resolve())
    if resolved.startswith("\\\\?\\UNC\\"):
        resolved = "\\\\" + resolved[8:]
    elif resolved.startswith("\\\\?\\"):
        resolved = resolved[4:]
    return Path(resolved)


class MediaIntegrityError(ValueError):
    """Persisted bytes do not match the exact asset reference."""


class MediaAssetConflictError(FileExistsError):
    """Different bytes already occupy an immutable asset path."""


class MediaStore:
    """Explicit local root; no discovery, replacement or regeneration policy.

    Publication follows ArtifactStore's flushed temporary file + exclusive hard
    link pattern. Unsupported filesystems fail safely. Links/junctions in asset
    paths are rejected. The root must be controlled by the caller, not concurrently
    rearranged by another process; these checks are not an OS sandbox.
    """

    def __init__(self, root: Path) -> None:
        self.root = _resolved_path(Path(root))

    def _path(self, reference: MediaAssetReference) -> Path:
        path = self.root
        for part in (None, *reference.relative_path.split("/")):
            if part is not None:
                path = path / part
            if path.is_symlink() or path.is_junction():
                raise ValueError("Media paths must not contain symbolic links or junctions")
            if not _resolved_path(path).is_relative_to(self.root):
                raise ValueError("Media path escapes root")
        return path

    def read_bytes(self, reference: MediaAssetReference) -> bytes:
        """Read only the named asset and authenticate its expected SHA-256."""
        reference = MediaAssetReference.model_validate(reference.model_dump(mode="json"))
        data = self._path(reference).read_bytes()
        if hashlib.sha256(data).hexdigest() != reference.sha256:
            raise MediaIntegrityError(f"SHA-256 mismatch for {reference.relative_path}")
        return data

    def verify(self, reference: MediaAssetReference) -> bool:
        """Return True on exact integrity; missing/corrupt assets raise."""
        self.read_bytes(reference)
        return True

    def save_bytes(self, *, asset_id: str, relative_path: str,
                   media_type: MediaType, data: bytes) -> MediaAssetReference:
        if not isinstance(data, bytes) or not data:
            raise ValueError("Media data must be non-empty bytes")
        # Reuse P7-A field/path validation before any filesystem side effects.
        reference = MediaAssetReference(asset_id=asset_id, relative_path=relative_path,
                                        media_type=media_type, sha256=hashlib.sha256(data).hexdigest())
        path = self._path(reference)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = self._path(reference)
        fd, temporary = tempfile.mkstemp(prefix=".pending-", suffix=".tmp", dir=path.parent)
        temporary_path = Path(temporary)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # Recheck containment before publication; never replace a destination.
            path = self._path(reference)
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                existing = self._path(reference).read_bytes()
                if existing != data:
                    raise MediaAssetConflictError(f"Conflicting media at {relative_path}") from None
            persisted = self.read_bytes(reference)
            if persisted != data:
                raise MediaIntegrityError(f"Persisted bytes differ at {relative_path}")
            return reference
        finally:
            temporary_path.unlink(missing_ok=True)
