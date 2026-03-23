"""Shared interactive utilities for VFH CLI tools (catalog, run tracker viewer, etc.)."""

from __future__ import annotations

import readline  # noqa: F401 — enables line editing (arrow keys, etc.) in input()
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------


class C:
    """ANSI color codes for terminal output."""
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    RESET = "\033[0m"


def colored(text: str, *codes: str) -> str:
    """Wrap text in ANSI color codes."""
    return "".join(codes) + text + C.RESET


# ---------------------------------------------------------------------------
# Cancel handling
# ---------------------------------------------------------------------------


class UserCancelled(Exception):
    """Raised when user types 'esc' or Ctrl+C to cancel an interactive prompt."""
    pass


def input_or_esc(prompt: str, prefill: str = "") -> str:
    """Like input(), but raises UserCancelled if user types 'esc' or hits Ctrl+C.

    If *prefill* is given, pre-populates the input line so the user can edit it
    inline (requires readline).
    """
    import readline

    if prefill:
        def _hook() -> None:
            readline.insert_text(prefill)
            readline.redisplay()
        readline.set_startup_hook(_hook)

    try:
        value = input(prompt)
    except (KeyboardInterrupt, EOFError):
        print()  # newline after ^C
        raise UserCancelled()
    finally:
        if prefill:
            readline.set_startup_hook()  # clear hook
    if value.strip().lower() == "esc":
        raise UserCancelled()
    return value


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------


def copy_to_clipboard(text: str) -> bool:
    """Try to copy text to clipboard. Returns True on success.

    Tries OSC 52 first (works over SSH in iTerm2/modern terminals),
    then falls back to pbcopy/xclip/xsel for local sessions.
    """
    # OSC 52: terminal escape sequence that sets the local clipboard.
    # Works over SSH in iTerm2, kitty, alacritty, etc.
    import base64
    import sys
    try:
        encoded = base64.b64encode(text.encode()).decode()
        sys.stdout.write(f"\033]52;c;{encoded}\a")
        sys.stdout.flush()
        return True
    except Exception:
        pass

    # Fallback: local clipboard tools
    for cmd in [["pbcopy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]:
        try:
            subprocess.run(cmd, input=text.encode(), check=True, timeout=2)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    return False


def copy_and_print_url(url: str | None, label: str = "URL") -> None:
    """Print a URL and try to copy to clipboard."""
    if not url:
        print(f"  {colored(f'No {label} available.', C.RED)}")
        return
    print(f"\n  {colored(url, C.BLUE, C.BOLD)}")
    if copy_to_clipboard(text=url):
        print(f"  {colored('Copied to clipboard.', C.GREEN)}")
    else:
        print(f"  {colored('(copy manually — no clipboard tool found)', C.DIM)}")


def copy_and_print_path(path: str) -> None:
    """Print a path and try to copy to clipboard."""
    print(f"\n  {colored(path, C.BOLD)}")
    if copy_to_clipboard(text=path):
        print(f"  {colored('Copied to clipboard.', C.GREEN)}")
    else:
        print(f"  {colored('(copy manually)', C.DIM)}")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def trunc(text: str, max_len: int = 60) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
