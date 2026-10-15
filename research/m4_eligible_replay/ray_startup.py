"""Short Ray socket roots and exception-safe cleanup, without importing Ray."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile


LINUX_SOCKET_LIMIT = 107
SOCKET_SUFFIX = (
    "/session_9999-12-31_23-59-59_999999_2147483647/sockets/plasma_store.999999"
)
ROOT_PREFIXES = {"preflight": "m4pf-", "a": "m4ra-", "b": "m4rb-"}


def validate_ray_root(root: Path) -> int:
    """Reject unsafe roots using a conservative Linux UTF-8 socket budget.

    Returns:
        Encoded socket bytes with ten PID digits and six collision digits.
    """
    if not root.is_absolute() or ".." in root.parts or "\0" in str(root):
        raise ValueError("Ray root must be an absolute, normalized pathname")
    size = len((str(root) + SOCKET_SUFFIX).encode("utf-8"))
    if size > LINUX_SOCKET_LIMIT:
        raise ValueError(
            f"Ray socket budget exceeded: {size} bytes > {LINUX_SOCKET_LIMIT}: {root}"
        )
    return size


def allocate_ray_root(stage: str) -> Path:
    """Allocate a distinct short node-local root, retaining logs until job teardown."""
    if stage not in ROOT_PREFIXES:
        raise ValueError("unknown Ray startup stage")
    root = Path(tempfile.mkdtemp(prefix=ROOT_PREFIXES[stage], dir="/tmp"))
    validate_ray_root(root)
    return root


@contextmanager
def ray_cleanup(shutdown: Callable[[], None]) -> Iterator[None]:
    """Cover partial startup without replacing an active exception with cleanup."""
    try:
        yield
    finally:
        original = sys.exception()
        try:
            shutdown()
        except Exception as cleanup_error:
            # Cleanup must not hide the failure that triggered it, including
            # interrupts. With no primary error, cleanup failure stays fatal.
            if original is None:
                raise
            original.add_note(f"Ray shutdown also failed: {cleanup_error!r}")


def check_layout() -> dict[str, int]:
    """Check every planned prefix without allocating directories or starting Ray."""
    return {
        stage: validate_ray_root(Path("/tmp") / (prefix + "abcdefgh"))
        for stage, prefix in ROOT_PREFIXES.items()
    }


if __name__ == "__main__":
    print(
        json.dumps({"status": "PASS_RAY_SOCKET_LAYOUT", "socket_bytes": check_layout()})
    )
