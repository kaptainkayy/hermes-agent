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


@pytest.mark.asyncio
async def test_next_message_starts_opencode_job_and_skips_agent(monkeypatch):
    adapter = _adapter()
    adapter._set_secondbrain_pending(
        user_id="42",
        channel_id="123",
        corpus_key="science",
        label="Science claims",
    )
    starts = []

    monkeypatch.setattr(
        adapter,
        "_start_secondbrain_opencode_job",
        lambda **kwargs: starts.append(kwargs) or {"status": "queued", "job_id": "job-1"},
        raising=False,
    )

    channel = SimpleNamespace(id=123, send=AsyncMock())
    message = SimpleNamespace(
        content="what claims mention sleep?",
        channel=channel,
        author=SimpleNamespace(id=42, display_name="Tester", bot=False),
        id=777,
    )

    handled = await adapter._maybe_handle_secondbrain_pending_question(message, "what claims mention sleep?")

    assert handled is True
    assert starts[0]["corpus_key"] == "science"
    assert starts[0]["question"] == "what claims mention sleep?"
    assert starts[0]["callback_target"] == "discord:123"
    assert channel.send.await_count == 1
    assert "asking OpenCode" in channel.send.await_args.args[0]
    assert adapter._secondbrain_pending == {}


@pytest.mark.asyncio
async def test_expired_pending_selection_asks_user_to_select_again(monkeypatch):
    adapter = _adapter()
    adapter._set_secondbrain_pending(
        user_id="42",
        channel_id="123",
        corpus_key="science",
        label="Science claims",
    )
    key = ("42", "123")
    adapter._secondbrain_pending[key]["expires_at"] = 1

    channel = SimpleNamespace(id=123, send=AsyncMock())
    message = SimpleNamespace(
        content="question",
        channel=channel,
        author=SimpleNamespace(id=42, display_name="Tester", bot=False),
        id=777,
    )

    handled = await adapter._maybe_handle_secondbrain_pending_question(message, "question")

    assert handled is True
    assert adapter._secondbrain_pending == {}
    channel.send.assert_awaited_once_with("That Second Brain selection expired. Run `/secondbrain` again.")


def test_start_secondbrain_opencode_job_dispatches_tool(monkeypatch):
    adapter = _adapter()
    dispatched = []

    class FakeRegistry:
        def dispatch(self, name, args, **kwargs):
            dispatched.append((name, args))
            return '{"status":"queued","job_id":"abc"}'

    monkeypatch.setitem(sys.modules, "tools.registry", type("M", (), {"registry": FakeRegistry()})())
    monkeypatch.setattr(
        adapter,
        "_build_secondbrain_opencode_prompt",
        lambda **_kwargs: "prompt text",
        raising=False,
    )

    result = adapter._start_secondbrain_opencode_job(
        corpus_key="science",
        label="Science claims",
        question="what?",
        callback_target="discord:123",
    )

    assert result == {"status": "queued", "job_id": "abc"}
    assert dispatched[0][0] == "opencode"
    assert dispatched[0][1]["action"] == "start_background"
    assert dispatched[0][1]["prompt"] == "prompt text"
    assert dispatched[0][1]["callback_target"] == "discord:123"


def test_start_secondbrain_opencode_job_returns_safe_error_if_prompt_build_fails(monkeypatch):
    adapter = _adapter()

    class FakeRegistry:
        def dispatch(self, name, args, **kwargs):
            raise AssertionError("registry.dispatch should not be called")

    monkeypatch.setitem(sys.modules, "tools.registry", type("M", (), {"registry": FakeRegistry()})())
    monkeypatch.setattr(
        adapter,
        "_build_secondbrain_opencode_prompt",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("secret trace")),
        raising=False,
    )

    result = adapter._start_secondbrain_opencode_job(
        corpus_key="science",
        label="Science claims",
        question="what?",
        callback_target="discord:123",
    )

    assert result == {"error": "OpenCode is unavailable"}


def test_start_secondbrain_opencode_job_returns_safe_error_if_dispatch_fails(monkeypatch):
    adapter = _adapter()

    class FakeRegistry:
        def dispatch(self, name, args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "tools.registry", type("M", (), {"registry": FakeRegistry()})())
    monkeypatch.setattr(
        adapter,
        "_build_secondbrain_opencode_prompt",
        lambda **_kwargs: "prompt text",
        raising=False,
    )

    result = adapter._start_secondbrain_opencode_job(
        corpus_key="science",
        label="Science claims",
        question="what?",
        callback_target="discord:123",
    )

    assert result == {"error": "OpenCode is unavailable"}


@pytest.mark.asyncio
async def test_pending_question_shows_fixed_unavailable_message(monkeypatch):
    adapter = _adapter()
    adapter._set_secondbrain_pending(
        user_id="42",
        channel_id="123",
        corpus_key="science",
        label="Science claims",
    )

    monkeypatch.setattr(
        adapter,
        "_start_secondbrain_opencode_job",
        lambda **_kwargs: {"error": "OpenCode is unavailable"},
    )

    channel = SimpleNamespace(id=123, send=AsyncMock())
    message = SimpleNamespace(
        content="question",
        channel=channel,
        author=SimpleNamespace(id=42, display_name="Tester", bot=False),
        id=777,
    )

    handled = await adapter._maybe_handle_secondbrain_pending_question(message, "question")

    assert handled is True
    assert channel.send.await_count == 2
    assert channel.send.await_args_list[0].args[0].startswith("Got it")
    assert "OpenCode is unavailable. Please try again in a moment." == channel.send.await_args_list[1].args[0]

    for call in channel.send.await_args_list:
        assert "OpenCode is unavailable: " not in call.args[0]
