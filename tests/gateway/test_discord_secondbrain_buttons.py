"""Tests for the buttons that appear after picking a Second Brain corpus.

Picking a database already works by tapping. Everything after it was typed: a
question, or a memorized phrase like "show entries". These tests pin the tapping
replacements for the read-only actions.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_discord_secondbrain_picker import (
    FakeIndex,
    _adapter,
    _interaction,
)

from plugins.platforms.discord.adapter import (
    SecondBrainActionView,
    SecondBrainCorpusSelectView,
)


def _fake_browse_plugin(monkeypatch, results):
    """Stand in for the plugin module the adapter looks up by handler __module__."""
    module = ModuleType("fake_secondbrain_plugin")
    module.browse_second_brain_corpus = AsyncMock(side_effect=results)
    module.answer_second_brain_question = AsyncMock(return_value="unused")

    async def handler(raw_args: str):
        return ""

    handler.__module__ = "fake_secondbrain_plugin"

    monkeypatch.setitem(sys.modules, "fake_secondbrain_plugin", module)
    monkeypatch.setattr(
        sys.modules["hermes_cli.plugins"],
        "get_plugin_commands",
        lambda: {"secondbrain": {"handler": handler, "plugin": "second-brain"}},
        raising=False,
    )
    return module.browse_second_brain_corpus


def _labels(view):
    return [child.label for child in view.children if getattr(child, "label", None)]


def button(view, label):
    return next(child for child in view.children if getattr(child, "label", None) == label)


async def _pick_corpus(adapter, monkeypatch, key="science"):
    view = SecondBrainCorpusSelectView(
        adapter=adapter, index=FakeIndex(), allowed_user_ids={"42"}, allowed_role_ids=set()
    )
    interaction = _interaction()
    interaction.data = {"values": [key]}
    await view._on_corpus_selected(interaction)
    return interaction.response.edit_message.await_args.kwargs["view"]


@pytest.mark.asyncio
async def test_picking_a_corpus_offers_actions_for_it(monkeypatch):
    adapter = _adapter()

    action_view = await _pick_corpus(adapter, monkeypatch)

    assert isinstance(action_view, SecondBrainActionView)
    assert _labels(action_view) == ["Recent entries", "Change database"]


@pytest.mark.asyncio
async def test_picking_a_corpus_still_arms_the_typed_question(monkeypatch):
    """The buttons are additive: sending a message must still ask the question."""
    adapter = _adapter()

    await _pick_corpus(adapter, monkeypatch)

    assert adapter._secondbrain_pending[("42", "123")]["corpus_key"] == "science"


@pytest.mark.asyncio
async def test_recent_entries_browses_the_corpus_that_was_picked(monkeypatch):
    adapter = _adapter()
    browse = _fake_browse_plugin(monkeypatch, ["Most recent entries in science.claims:\n- science.claims:c1 - Sleep"])

    action_view = await _pick_corpus(adapter, monkeypatch)
    press = _interaction()
    press.edit_original_response = AsyncMock()
    await button(action_view, "Recent entries").callback(press)

    assert browse.await_args.args[0] == "science", "must browse the picked corpus, not everything"
    shown = press.edit_original_response.await_args.kwargs["content"]
    assert "science.claims:c1" in shown


@pytest.mark.asyncio
async def test_recent_entries_fits_inside_a_discord_message(monkeypatch):
    """Twelve rows of long titles can exceed Discord's 2000-character limit."""
    adapter = _adapter()
    _fake_browse_plugin(monkeypatch, ["x" * 5000])

    action_view = await _pick_corpus(adapter, monkeypatch)
    press = _interaction()
    press.edit_original_response = AsyncMock()
    await button(action_view, "Recent entries").callback(press)

    shown = press.edit_original_response.await_args.kwargs["content"]
    assert len(shown) <= 2000


@pytest.mark.asyncio
async def test_change_database_returns_to_the_picker(monkeypatch):
    """Re-running the slash command to switch databases is the thing being avoided."""
    adapter = _adapter()

    action_view = await _pick_corpus(adapter, monkeypatch)
    press = _interaction()
    press.edit_original_response = AsyncMock()
    await button(action_view, "Change database").callback(press)

    back = press.edit_original_response.await_args.kwargs["view"]
    assert isinstance(back, SecondBrainCorpusSelectView)


@pytest.mark.asyncio
async def test_unauthorized_press_cannot_read_the_knowledge_base(monkeypatch):
    adapter = _adapter()
    browse = _fake_browse_plugin(monkeypatch, ["secrets"])

    action_view = await _pick_corpus(adapter, monkeypatch)
    intruder = _interaction(user_id="999")
    intruder.edit_original_response = AsyncMock()
    await button(action_view, "Recent entries").callback(intruder)

    assert browse.await_count == 0
    intruder.response.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_browse_api_reports_itself(monkeypatch):
    """An older plugin version has no scoped browse; say so rather than crash."""
    adapter = _adapter()
    module = ModuleType("fake_secondbrain_plugin")

    async def handler(raw_args: str):
        return ""

    handler.__module__ = "fake_secondbrain_plugin"
    monkeypatch.setitem(sys.modules, "fake_secondbrain_plugin", module)
    monkeypatch.setattr(
        sys.modules["hermes_cli.plugins"],
        "get_plugin_commands",
        lambda: {"secondbrain": {"handler": handler, "plugin": "second-brain"}},
        raising=False,
    )

    action_view = await _pick_corpus(adapter, monkeypatch)
    press = _interaction()
    press.edit_original_response = AsyncMock()
    await button(action_view, "Recent entries").callback(press)

    assert "unavailable" in press.edit_original_response.await_args.kwargs["content"].lower()
