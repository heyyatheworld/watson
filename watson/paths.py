"""Format filesystem paths shown in Discord messages."""

from __future__ import annotations

import os


def format_saved_paths_for_discord(
    paths: list[str], recordings_dir: str, hide_paths: bool
) -> list[str]:
    """
    Avoid leaking absolute host paths: optionally basename-only, otherwise paths
    relative to recordings_dir when under that directory.
    """
    if hide_paths:
        return [os.path.basename(p) for p in paths]
    rd_abs = os.path.normpath(os.path.abspath(recordings_dir))
    out: list[str] = []
    for p in paths:
        ap = os.path.normpath(os.path.abspath(p))
        prefix = rd_abs + os.sep
        if ap.startswith(prefix) or ap == rd_abs:
            try:
                out.append(os.path.relpath(ap, rd_abs))
            except ValueError:
                out.append(os.path.basename(ap))
        else:
            out.append(os.path.basename(ap))
    return out
