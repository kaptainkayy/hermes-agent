from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from plugins.platforms.discord.adapter import DiscordAdapter, SecondBrainCorpusSelectView  # noqa: E402


def _adapter():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99999, name="HermesBot"), get_channel=lambda _id: None, fetch_channel=AsyncMock())
    adapter._allowed_user_ids = {"42"}
    adapter._allowed_role_ids = set()
    adapter._check_slash_authorization = AsyncMock(return_value=True)
    adapter._text_batch_delay_seconds = 0
    return adapter


def _interaction(user_id="42", channel_id=123):
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, display_name="Tester", roles=[]),
        channel_id=channel_id,
        channel=SimpleNamespace(id=channel_id),
        response=SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock(), edit_message=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )


def _entry(key="science", label="Science claims"):
    return SimpleNamespace(key=key, label=label, description="Claims and evidence", row_count=3, aliases=[key])


class FakeIndex:
    def enabled_entries(self):
        return [_entry(), _entry("robert_moore", "Robert Moore transcripts")]

    def get_entry(self, key):
        for entry in self.enabled_entries():
            if entry.key == key:
                return entry
        raise KeyError(key)


@pytest.mark.asyncio
async def test_secondbrain_slash_sends_ephemeral_select(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_load_secondbrain_index_for_discord", lambda: FakeIndex(), raising=False)
    interaction = _interaction()

    await adapter._handle_secondbrain_picker_slash(interaction)

    sent = interaction.response.send_message.await_args.kwargs
    assert sent["content"].startswith("Which database would you like")
    assert sent["ephemeral"] is True
    assert isinstance(sent["view"], SecondBrainCorpusSelectView)
    assert sent["view"].children[0].options[0].label == "Science claims"


@pytest.mark.asyncio
async def test_select_stores_pending_selection(monkeypatch):
    adapter = _adapter()
    view = SecondBrainCorpusSelectView(adapter=adapter, index=FakeIndex(), allowed_user_ids={"42"}, allowed_role_ids=set())
    interaction = _interaction()
    interaction.data = {"values": ["science"]}

    await view._on_corpus_selected(interaction)

    pending = adapter._secondbrain_pending[("42", "123")]
    assert pending["corpus_key"] == "science"
    assert pending["label"] == "Science claims"
    interaction.response.edit_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_unauthorized_select_is_rejected():
    adapter = _adapter()
    view = SecondBrainCorpusSelectView(adapter=adapter, index=FakeIndex(), allowed_user_ids={"42"}, allowed_role_ids=set())
    interaction = _interaction(user_id="99")
    interaction.data = {"values": ["science"]}

    await view._on_corpus_selected(interaction)

    assert adapter._secondbrain_pending == {}
    interaction.response.send_message.assert_awaited_once()
    assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True
