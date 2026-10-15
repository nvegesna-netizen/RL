"""Evaluation-local isolation of the already initialized Megatron RNG tracker."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol


class Tracker(Protocol):
    """The pinned Core/TE tracker interface used by this qualification."""

    def is_initialized(self) -> bool: ...

    def get_states(self) -> dict[str, Any]: ...

    def set_states(self, states: dict[str, Any]) -> None: ...


@contextmanager
def isolate_evaluation_tracker(
    tracker: Tracker,
    *,
    clone_state: Callable[[Any], Any],
    digest: Callable[[Any], str],
) -> Iterator[None]:
    """Clone and restore named states without resetting any training generator.

    Unknown state types must be rejected by clone_state. Cleanup failure is
    fatal; when evaluation also fails, retain that original exception and add
    cleanup diagnostics. No caller may treat either failure as qualification.
    """
    if not tracker.is_initialized():
        raise ValueError("evaluation requires an initialized RNG tracker")
    states = tracker.get_states()
    if (
        not isinstance(states, dict)
        or not states
        or not all(isinstance(name, str) for name in states)
    ):
        raise ValueError("expected nonempty named RNG tracker states")
    saved = {name: clone_state(value) for name, value in states.items()}
    expected = digest(saved)

    def restore() -> None:
        tracker.set_states({name: clone_state(value) for name, value in saved.items()})
        if not tracker.is_initialized() or digest(tracker.get_states()) != expected:
            raise ValueError("RNG tracker restore round trip failed")

    # Check the runtime implementation before allowing an evaluation forward.
    restore()
    primary = None
    try:
        yield
    except BaseException as error:
        # Cleanup must also cover cancellation and preserve the primary error.
        primary = error
        raise
    finally:
        try:
            current = tracker.get_states()
            restore()
            if set(current) != set(saved):
                raise ValueError("evaluation changed RNG tracker stream names")
            for value in current.values():
                clone_state(value)
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(
                f"RNG tracker cleanup failed: {type(error).__name__}: {error}"
            )
