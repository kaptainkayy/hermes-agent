"""Append-only helpers for Obsidian Markdown notes."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_OBSIDIAN_VAULT = Path("/home/kayai3/Documents/ObsidianVault")


class ObsidianNoteError(ValueError):
    """User-facing note append failure."""


@dataclass(frozen=True)
class NoteAppendRequest:
    title: str
    content: str


def obsidian_vault_path() -> Path:
    raw = os.getenv("HERMES_OBSIDIAN_VAULT", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_OBSIDIAN_VAULT


def parse_note_command_args(args: str) -> NoteAppendRequest:
    """Parse `/note` args into an existing note title and appended text."""

    text = str(args or "").strip()
    if text.lower().startswith("append "):
        text = text[7:].strip()

    if not text:
        raise ObsidianNoteError("Usage: /note \"Existing Note Title\" text to append")

    if text[0] in {'"', "'"}:
        try:
            parts = shlex.split(text)
        except ValueError as exc:
            raise ObsidianNoteError(f"Could not parse quoted note title: {exc}") from exc
        if len(parts) < 2:
            raise ObsidianNoteError("Add text after the quoted note title.")
        return _validated_request(parts[0], " ".join(parts[1:]))

    for delimiter in (" :: ", " -- ", " - "):
        if delimiter in text:
            title, content = text.split(delimiter, 1)
            return _validated_request(title, content)

    colon_match = re.match(r"(.+?):\s+(.+)", text, flags=re.DOTALL)
    if colon_match:
        return _validated_request(colon_match.group(1), colon_match.group(2))

    raise ObsidianNoteError(
        "Put the existing note title in quotes, or separate title and text with ` :: `."
    )


def parse_spoken_note_append(transcript: str) -> NoteAppendRequest | None:
    """Parse natural voice phrases like `append to note X: Y`."""

    text = re.sub(r"\s+", " ", str(transcript or "")).strip()
    if not text:
        return None

    match = re.match(
        r"(?i)^(?:please\s+)?(?:append|add)\s+(?:this\s+)?(?:to|into)\s+(?:the\s+)?note\s+(.+?)(?:\s*[:;]\s+|\s+(?:saying|that says|with)\s+)(.+)$",
        text,
    )
    if not match:
        return None
    return _validated_request(match.group(1), match.group(2))


def append_to_existing_note(
    request: NoteAppendRequest,
    *,
    vault: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Append a timestamped block to an existing Markdown note without rewriting it."""

    vault_path = (vault or obsidian_vault_path()).expanduser().resolve()
    note_path = resolve_existing_note(request.title, vault=vault_path)
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
    payload = f"\n\n## {timestamp} Hermes Note\n\n{request.content.strip()}\n"

    # Use O_APPEND and never unlink/rename/truncate. This makes this helper
    # safe for voice/Discord capture where the only intended operation is append.
    fd = os.open(note_path, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return note_path


def resolve_existing_note(title: str, *, vault: Path | None = None) -> Path:
    vault_path = (vault or obsidian_vault_path()).expanduser().resolve()
    if not vault_path.is_dir():
        raise ObsidianNoteError(f"Obsidian vault not found: {vault_path}")

    raw_title = str(title or "").strip().strip('"\'')
    if not raw_title:
        raise ObsidianNoteError("Note title is empty.")
    if any(part == ".." for part in Path(raw_title).parts):
        raise ObsidianNoteError("Note title cannot contain `..`.")

    wanted_name = raw_title if raw_title.lower().endswith(".md") else f"{raw_title}.md"
    candidates: list[Path] = []
    for note in vault_path.rglob("*.md"):
        if ".obsidian" in note.parts:
            continue
        rel = note.resolve().relative_to(vault_path)
        rel_text = rel.as_posix().lower()
        if note.name.lower() == wanted_name.lower() or rel_text == wanted_name.lower():
            candidates.append(note.resolve())

    if not candidates:
        raise ObsidianNoteError(f"Existing Obsidian note not found: {raw_title}")
    if len(candidates) > 1:
        options = ", ".join(path.relative_to(vault_path).as_posix() for path in candidates[:5])
        raise ObsidianNoteError(f"Note title is ambiguous. Match a path instead: {options}")

    return candidates[0]


def _validated_request(title: str, content: str) -> NoteAppendRequest:
    clean_title = str(title or "").strip().strip('"\'')
    clean_content = str(content or "").strip()
    if not clean_title:
        raise ObsidianNoteError("Note title is empty.")
    if not clean_content:
        raise ObsidianNoteError("Nothing to append.")
    return NoteAppendRequest(title=clean_title, content=clean_content)
