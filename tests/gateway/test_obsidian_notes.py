from datetime import datetime
from types import SimpleNamespace

import pytest

from gateway.obsidian_notes import (
    ObsidianNoteError,
    NoteAppendRequest,
    append_to_existing_note,
    parse_note_command_args,
    parse_spoken_note_append,
)


def test_append_to_existing_note_preserves_existing_content(tmp_path):
    note = tmp_path / "10 Projects" / "Project Plan.md"
    note.parent.mkdir(parents=True)
    note.write_text("# Project Plan\n\nExisting text.\n", encoding="utf-8")

    appended = append_to_existing_note(
        NoteAppendRequest("Project Plan", "New captured detail."),
        vault=tmp_path,
        now=datetime(2026, 5, 31, 12, 34),
    )

    assert appended == note.resolve()
    assert note.read_text(encoding="utf-8") == (
        "# Project Plan\n\nExisting text.\n"
        "\n\n## 2026-05-31 12:34 Hermes Note\n\n"
        "New captured detail.\n"
    )


def test_append_refuses_missing_note(tmp_path):
    with pytest.raises(ObsidianNoteError, match="not found"):
        append_to_existing_note(NoteAppendRequest("Missing", "Do not create this."), vault=tmp_path)


def test_parse_note_command_args_with_quoted_title():
    request = parse_note_command_args('"Project Plan" remember this detail')

    assert request == NoteAppendRequest("Project Plan", "remember this detail")


def test_parse_note_command_args_with_delimiter():
    request = parse_note_command_args("Project Plan :: remember this detail")

    assert request == NoteAppendRequest("Project Plan", "remember this detail")


def test_parse_spoken_note_append():
    request = parse_spoken_note_append("append to note Project Plan: remember this detail")

    assert request == NoteAppendRequest("Project Plan", "remember this detail")



def test_parse_spoken_note_append_ignores_normal_voice():
    assert parse_spoken_note_append("what is the weather") is None


@pytest.mark.asyncio
async def test_gateway_note_command_appends_existing_note(tmp_path, monkeypatch):
    note = tmp_path / "10 Projects" / "Project Plan.md"
    note.parent.mkdir(parents=True)
    note.write_text("# Project Plan\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_OBSIDIAN_VAULT", str(tmp_path))

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    event = SimpleNamespace(get_command_args=lambda: '"Project Plan" captured from Discord')

    result = await runner._handle_note_command(event)

    assert result == "Appended to Obsidian note `Project Plan`."
    assert "captured from Discord" in note.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_voice_note_response_appends_existing_note(tmp_path, monkeypatch):
    note = tmp_path / "10 Projects" / "Project Plan.md"
    note.parent.mkdir(parents=True)
    note.write_text("# Project Plan\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_OBSIDIAN_VAULT", str(tmp_path))

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)

    result = await runner._obsidian_note_voice_response(
        "append to note Project Plan: captured from voice"
    )

    assert result == "Appended to Project Plan."
    assert "captured from voice" in note.read_text(encoding="utf-8")
