from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType
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


def _fake_secondbrain_plugin(monkeypatch, answers):
    module = ModuleType("fake_secondbrain_plugin")
    module.answer_second_brain_question = AsyncMock(side_effect=answers)

    async def handler(raw_args: str):
        return ""

    handler.__module__ = "fake_secondbrain_plugin"

    monkeypatch.setitem(sys.modules, "fake_secondbrain_plugin", module)
    monkeypatch.setattr(
        sys.modules["hermes_cli.plugins"],
        "get_plugin_commands",
        lambda: {
            "secondbrain": {
                "handler": handler,
                "plugin": "second-brain",
            }
        },
        raising=False,
    )
    return module.answer_second_brain_question


class FakeIndex:
    def enabled_entries(self):
        return [
            _entry(),
            _entry("ai_science", "AI research papers"),
            _entry("robert_moore", "Robert Moore transcripts"),
        ]

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
async def test_secondbrain_picker_includes_ai_science(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_load_secondbrain_index_for_discord", lambda: FakeIndex(), raising=False)
    interaction = _interaction()

    await adapter._handle_secondbrain_picker_slash(interaction)

    options = interaction.response.send_message.await_args.kwargs["view"].children[0].options
    assert any(option.label == "AI research papers" for option in options)


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
async def test_select_stores_ai_science_selection(monkeypatch):
    adapter = _adapter()
    view = SecondBrainCorpusSelectView(adapter=adapter, index=FakeIndex(), allowed_user_ids={"42"}, allowed_role_ids=set())
    interaction = _interaction()
    interaction.data = {"values": ["ai_science"]}

    await view._on_corpus_selected(interaction)

    pending = adapter._secondbrain_pending[("42", "123")]
    assert pending["corpus_key"] == "ai_science"
    assert pending["label"] == "AI research papers"
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
@pytest.mark.parametrize(
    "corpus_key,label,expected_key",
    [
        ("science", "Science claims", "science"),
        ("ai_science", "AI research papers", "ai_science"),
    ],
)
async def test_next_message_answers_via_plugin_api_and_skips_opencode_job(monkeypatch, corpus_key, label, expected_key):
    adapter = _adapter()
    adapter._set_secondbrain_pending(
        user_id="42",
        channel_id="123",
        corpus_key=corpus_key,
        label=label,
    )
    answer_api = _fake_secondbrain_plugin(monkeypatch, ["Answer: sleep supports memory consolidation."])

    channel = SimpleNamespace(id=123, send=AsyncMock())
    message = SimpleNamespace(
        content="what claims mention sleep?",
        channel=channel,
        author=SimpleNamespace(id=42, display_name="Tester", bot=False),
        id=777,
    )

    handled = await adapter._maybe_handle_secondbrain_pending_question(message, "what claims mention sleep?")

    assert handled is True
    assert answer_api.await_count == 1
    assert answer_api.await_args.args[1] == "what claims mention sleep?"
    assert answer_api.await_args.args[2] == expected_key
    assert channel.send.await_count == 2
    assert channel.send.await_args_list[0].args[0].startswith("Got it")
    assert channel.send.await_args_list[1].args[0] == "Answer: sleep supports memory consolidation."


@pytest.mark.asyncio
async def test_next_message_long_answer_is_split_to_discord_chunks(monkeypatch):
    adapter = _adapter()
    adapter._set_secondbrain_pending(
        user_id="42",
        channel_id="123",
        corpus_key="science",
        label="Science claims",
    )
    answer_api = _fake_secondbrain_plugin(monkeypatch, ["x" * 5000])

    channel = SimpleNamespace(id=123, send=AsyncMock())
    message = SimpleNamespace(
        content="long question",
        channel=channel,
        author=SimpleNamespace(id=42, display_name="Tester", bot=False),
        id=777,
    )

    handled = await adapter._maybe_handle_secondbrain_pending_question(message, "long question")

    assert handled is True
    assert answer_api.await_count == 1
    sent_chunks = [call.args[0] for call in channel.send.await_args_list]
    assert sent_chunks[0].startswith("Got it")
    assert len(sent_chunks) >= 3
    assert all(len(chunk) <= 2000 for chunk in sent_chunks[1:])
    assert any(re.search(r"\(1/", chunk) for chunk in sent_chunks[1:])


@pytest.mark.asyncio
async def test_next_message_selected_question_ignores_second_brain_dm_target(monkeypatch):
    adapter = _adapter()
    adapter._set_secondbrain_pending(
        user_id="42",
        channel_id="123",
        corpus_key="science",
        label="Science claims",
    )
    answer_api = _fake_secondbrain_plugin(monkeypatch, ["Scoped second-brain answer."])
    monkeypatch.setenv("SECOND_BRAIN_DISCORD_DM_TARGET", "discord:sentinel-dm-target")

    channel = SimpleNamespace(id=123, send=AsyncMock())
    message = SimpleNamespace(
        content="what claims mention sleep?",
        channel=channel,
        author=SimpleNamespace(id=42, display_name="Tester", bot=False),
        id=777,
    )

    handled = await adapter._maybe_handle_secondbrain_pending_question(message, "what claims mention sleep?")

    assert handled is True
    assert answer_api.await_count == 1
    assert channel.send.await_args_list[0].args[0].startswith("Got it")
    assert channel.send.await_args_list[1].args[0] == "Scoped second-brain answer."
    assert "discord:sentinel-dm-target" not in channel.send.await_args_list[0].args[0]
    assert "discord:sentinel-dm-target" not in channel.send.await_args_list[1].args[0]


@pytest.mark.asyncio
async def test_next_message_pending_flow_does_not_call_opencode_callback_target(monkeypatch):
    adapter = _adapter()
    adapter._set_secondbrain_pending(
        user_id="42",
        channel_id="123",
        corpus_key="science",
        label="Science claims",
    )
    answer_api = _fake_secondbrain_plugin(monkeypatch, ["Answer from plugin API"])

    callback_target_calls = {}

    def _unexpected_callback_target(*_args, **_kwargs):
        callback_target_calls["called"] = True
        return "unexpected"

    monkeypatch.setattr(
        adapter,
        "_secondbrain_callback_target",
        _unexpected_callback_target,
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
    assert answer_api.await_count == 1
    assert callback_target_calls == {}


def test_secondbrain_pending_flow_removed_opencode_surface_helpers():
    adapter = _adapter()
    assert not hasattr(type(adapter), "_start_secondbrain_opencode_job")
    assert not hasattr(type(adapter), "_build_secondbrain_opencode_prompt")
    assert not hasattr(type(adapter), "_secondbrain_callback_target")


@pytest.mark.asyncio
async def test_follow_up_messages_keep_pending_selection_and_renew_expiry(monkeypatch):
    adapter = _adapter()
    adapter._set_secondbrain_pending(
        user_id="42",
        channel_id="123",
        corpus_key="science",
        label="Science claims",
    )
    answer_api = _fake_secondbrain_plugin(
        monkeypatch,
        ["Answer one", "Answer two"],
    )
    original_expiry = adapter._secondbrain_pending[("42", "123")]["expires_at"]

    channel = SimpleNamespace(id=123, send=AsyncMock())
    message = SimpleNamespace(
        content="question one",
        channel=channel,
        author=SimpleNamespace(id=42, display_name="Tester", bot=False),
        id=777,
    )

    handled_first = await adapter._maybe_handle_secondbrain_pending_question(message, "question one")

    message.content = "question two"
    handled_second = await adapter._maybe_handle_secondbrain_pending_question(message, "question two")

    assert handled_first is True
    assert handled_second is True
    assert answer_api.await_count == 2
    assert answer_api.await_args_list[0].args[1] == "question one"
    assert answer_api.await_args_list[1].args[1] == "question two"
    assert answer_api.await_args_list[0].args[2] == "science"
    assert answer_api.await_args_list[1].args[2] == "science"
    assert adapter._secondbrain_pending[("42", "123")]["corpus_key"] == "science"
    renewed_expiry = adapter._secondbrain_pending[("42", "123")]["expires_at"]
    assert renewed_expiry >= original_expiry


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


@pytest.mark.asyncio
async def test_pending_question_shows_fixed_unavailable_message(monkeypatch):
    adapter = _adapter()
    adapter._set_secondbrain_pending(
        user_id="42",
        channel_id="123",
        corpus_key="science",
        label="Science claims",
    )

    module = ModuleType("fake_secondbrain_plugin")
    module.answer_second_brain_question = AsyncMock(side_effect=RuntimeError("fail"))

    async def handler(raw_args: str):
        return ""

    handler.__module__ = "fake_secondbrain_plugin"

    monkeypatch.setitem(sys.modules, "fake_secondbrain_plugin", module)
    monkeypatch.setattr(
        sys.modules["hermes_cli.plugins"],
        "get_plugin_commands",
        lambda: {
            "secondbrain": {
                "handler": handler,
                "plugin": "second-brain",
            }
        },
        raising=False,
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
    assert "Second Brain is unavailable. Please try again in a moment." == channel.send.await_args_list[1].args[0]

    for call in channel.send.await_args_list:
        assert "OpenCode" not in call.args[0]

    for call in channel.send.await_args_list:
        assert "OpenCode is unavailable: " not in call.args[0]
