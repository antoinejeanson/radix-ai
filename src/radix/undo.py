from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass, field


# Undo: snapshots file state per assistant turn so Assistant.undo() can
# restore changes and rewind the conversation to the right depth.
def _target_mode(path: str) -> int:
    """The permission mode a write to `path` should end up with.

    Existing files keep their mode; new files get what a plain
    `open(path, "w")` would create (0666 masked by the process umask).
    Ownership is not preserved (that would require root).

    Args:
        path: Target file path.

    Returns:
        The permission bits (mode & 0o777) to apply to the written file.
    """
    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def _write_atomic(path: str, content: str) -> None:
    """Write `content` to `path` atomically (write temp file, then rename).

    The target's permission mode is preserved: existing files keep their
    mode, new files get the default 0666 masked by the umask. Ownership is
    not preserved (that would require root).

    Args:
        path: Target file path.
        content: Text to write.
    """
    mode = _target_mode(path)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)) or ".")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


@dataclass
class FileSnapshot:
    """State of a file before a change.

    `content is None` means the file did not exist.
    """

    path: str
    content: str | None


@dataclass
class Turn:
    """File snapshots taken during one assistant turn (one `Assistant.chat` call).

    `history_depth` is the coordinator history length when the turn started,
    so undoing the turn knows how far to rewind the conversation.
    """

    history_depth: int
    files: dict[str, FileSnapshot] = field(default_factory=dict)


@dataclass
class UndoResult:
    """Outcome of an undo.

    `history_depth` is how far the conversation should be truncated to, or
    `None` when nothing was undone.
    """

    restored: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    history_depth: int | None = None


class UndoLog:
    """Tracks file changes per turn so they can be reverted.

    A turn starts with `begin_turn`; `snapshot(path)` records a file's state
    before it is modified, keeping the earliest pre-turn state per path.
    `undo(n)` restores the files of the last n turns and reports how far the
    conversation should be rewound.
    """

    def __init__(self) -> None:
        self._turns: list[Turn] = []

    def begin_turn(self, history_depth: int) -> None:
        """Start recording a new assistant turn.

        Args:
            history_depth: The coordinator's history length at the start of
                the turn; undoing the turn rewinds to this depth.
        """
        self._turns.append(Turn(history_depth=history_depth))

    def snapshot(self, path: str) -> None:
        """Record a file's pre-change state, once per turn.

        The earliest state within the turn wins, so a file modified several
        times is restored to what it was before the turn.

        Args:
            path: The file to snapshot; non-existent files are recorded as
                such (so undoing removes them). Unreadable binaries are
                skipped.
        """
        if not self._turns:
            return
        path = os.path.abspath(os.path.expanduser(path))
        turn = self._turns[-1]
        if path in turn.files:
            return
        if not os.path.isfile(path):
            turn.files[path] = FileSnapshot(path, None)
            return
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            return
        turn.files[path] = FileSnapshot(path, content)

    def undo(self, turns: int = 1) -> UndoResult:
        """Restore the files changed during the last `turns` turn(s).

        Restores each file to its earliest recorded pre-turn state and
        reports how far the conversation should be rewound.

        Args:
            turns: How many turns to revert. 0 or a negative value is a
                no-op; more than recorded reverts everything recorded.

        Returns:
            An UndoResult with the restored and failed paths and the
            history depth to truncate to (None when nothing was undone).
        """
        result = UndoResult()
        if turns <= 0 or not self._turns:
            return result
        popped = self._turns[-turns:]
        del self._turns[-turns:]
        result.history_depth = min(t.history_depth for t in popped)
        earliest: dict[str, FileSnapshot] = {}
        for turn in popped:
            for snap in turn.files.values():
                earliest.setdefault(snap.path, snap)
        for snap in earliest.values():
            try:
                if self._restore(snap):
                    result.restored.append(snap.path)
            except OSError:
                result.failed.append(snap.path)
        return result

    def clear(self) -> None:
        """Forget all recorded turns and snapshots."""
        self._turns.clear()

    def __len__(self) -> int:
        """The number of recorded turns (used by the REPL's `/undo all`)."""
        return len(self._turns)

    @staticmethod
    def _restore(snap: FileSnapshot) -> bool:
        """Restore one snapshot to disk.

        Args:
            snap: The FileSnapshot to restore.

        Returns:
            True when the file actually changed on disk, False when it was
            already in the snapshot state.
        """
        if snap.content is None:
            if os.path.exists(snap.path):
                os.remove(snap.path)
                return True
            return False
        try:
            with open(snap.path, encoding="utf-8") as f:
                if f.read() == snap.content:
                    return False
        except (OSError, UnicodeDecodeError):
            pass
        _write_atomic(snap.path, snap.content)
        return True
