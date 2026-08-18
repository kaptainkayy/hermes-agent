"""Tests for the /factory button controls on Discord.

The factory plugin can only return a string, so tapping instead of typing has to
come from the adapter. These tests pin the behavior that matters on a phone: the
buttons exist, they run the plugin, and the destructive one asks first.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_discord_slash_commands import FakeTree

from plugins.platforms.discord.adapter import DiscordAdapter


FACTORY_PLUGIN = {
    "factory": {
        "handler": None,  # set per test
        "description": "Check the software factory",
        "args_hint": "[status|plan|unblock]",
        "plugin": "factory-control",
        "dispatch": "direct",
        "thread_response": False,
    }
}


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    adapter._check_slash_authorization = AsyncMock(return_value=True)
    return adapter


def make_interaction() -> SimpleNamespace:
    return SimpleNamespace(
        user=SimpleNamespace(name="Keith", id=301523463926185989, display_name="Keith"),
        channel=SimpleNamespace(id=123),
        channel_id=123,
        guild_id=999,
        response=SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock(), edit_message=AsyncMock()),
        edit_original_response=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_bare_factory_slash_offers_buttons(adapter):
    """Typing subcommand names on a phone keyboard is the thing being avoided."""
    adapter._register_slash_commands()
    interaction = make_interaction()

    await adapter._client.tree.commands["factory"](interaction, args="")

    interaction.response.send_message.assert_awaited_once()
    kwargs = interaction.response.send_message.await_args.kwargs
    assert kwargs["ephemeral"] is True
    labels = [child.label for child in kwargs["view"].children]
    assert labels == ["Status", "Plan", "Unblock", "Clear STOP"]


async def open_buttons(adapter, interaction):
    await adapter._client.tree.commands["factory"](interaction, args="")
    return interaction.response.send_message.await_args.kwargs["view"]


def button(view, label: str):
    return next(child for child in view.children if child.label == label)


@pytest.mark.asyncio
async def test_pressing_a_button_runs_that_subcommand_and_shows_the_answer(adapter):
    calls = []

    async def handler(args):
        calls.append(args)
        return "idle: DONE — nothing left to build, timer active"

    plugin = {"factory": {**FACTORY_PLUGIN["factory"], "handler": handler}}
    adapter._register_slash_commands()

    with patch("hermes_cli.plugins.get_plugin_commands", return_value=plugin):
        view = await open_buttons(adapter, make_interaction())
        press = make_interaction()
        await button(view, "Status").callback(press)

    assert calls == ["status"], "should invoke the same plugin handler the slash command uses"
    # Deferring first is what keeps a cold node start from blowing Discord's
    # three-second acknowledgement window.
    press.response.defer.assert_awaited_once()
    assert "idle: DONE" in press.edit_original_response.await_args.kwargs["content"]
    # The view stays attached so the next question is another tap, not another slash.
    assert press.edit_original_response.await_args.kwargs["view"] is view


@pytest.mark.asyncio
async def test_clearing_stop_asks_before_it_acts(adapter):
    """STOP is the one marker no code writes.

    It exists only because a human decided to halt the factory, so a single
    mis-tap on a phone must not overrule that decision. The CLI refuses unless the
    subcommand names STOP; the button has to make that intent just as deliberate.
    """
    calls = []

    async def handler(args):
        calls.append(args)
        return "Cleared STOP and started a run."

    plugin = {"factory": {**FACTORY_PLUGIN["factory"], "handler": handler}}
    adapter._register_slash_commands()

    with patch("hermes_cli.plugins.get_plugin_commands", return_value=plugin):
        view = await open_buttons(adapter, make_interaction())

        first = make_interaction()
        await button(view, "Clear STOP").callback(first)

        assert calls == [], "the first press must not clear anything"
        asked = first.edit_original_response.await_args.kwargs["content"]
        assert "again" in asked.lower(), f"should ask for confirmation, said: {asked}"

        second = make_interaction()
        await button(view, "Clear STOP").callback(second)

    assert calls == ["unblock stop"], "the second press is the deliberate one"
    assert "Cleared STOP" in second.edit_original_response.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_unauthorized_press_cannot_run_the_factory(adapter):
    """The buttons live on an ephemeral message, but the interaction is still forgeable.

    These commands start runs and clear halts, so the component gate has to hold on
    its own rather than trusting that only the invoker can see the message.
    """
    calls = []

    async def handler(args):
        calls.append(args)
        return "should never run"

    plugin = {"factory": {**FACTORY_PLUGIN["factory"], "handler": handler}}
    adapter._allowed_user_ids = {"301523463926185989"}
    adapter._register_slash_commands()

    with patch("hermes_cli.plugins.get_plugin_commands", return_value=plugin):
        view = await open_buttons(adapter, make_interaction())

        intruder = make_interaction()
        intruder.user = SimpleNamespace(name="Nobody", id=1, display_name="Nobody")
        await button(view, "Unblock").callback(intruder)

    assert calls == [], "an unlisted user must not be able to start a run"
    intruder.response.send_message.assert_awaited_once()
    assert "authorized" in intruder.response.send_message.await_args.args[0].lower()


@pytest.mark.asyncio
async def test_typed_subcommand_skips_the_buttons(adapter):
    """Buttons are for phones; the typed form still has to work from a desktop."""
    adapter._run_simple_slash = AsyncMock()
    adapter._register_slash_commands()
    interaction = make_interaction()

    await adapter._client.tree.commands["factory"](interaction, args="plan")

    adapter._run_simple_slash.assert_awaited_once_with(interaction, "/factory plan")
    interaction.response.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_plugin_reports_itself_instead_of_failing(adapter):
    """The adapter ships with Hermes; the factory plugin is separate and may be absent."""
    adapter._register_slash_commands()

    with patch("hermes_cli.plugins.get_plugin_commands", return_value={}):
        view = await open_buttons(adapter, make_interaction())
        press = make_interaction()
        await button(view, "Status").callback(press)

    assert "unavailable" in press.edit_original_response.await_args.kwargs["content"].lower()


@pytest.mark.asyncio
async def test_explicit_factory_slash_prevents_generic_plugin_registration(adapter):
    """The generic plugin mirror would replace the button command with a text-only one."""
    adapter._run_simple_slash = AsyncMock()

    with patch(
        "hermes_cli.commands._iter_plugin_command_entries",
        return_value=[("factory", "Check the software factory", "[status|plan|unblock]")],
    ):
        adapter._register_slash_commands()

    command = adapter._client.tree.commands["factory"]
    assert getattr(command, "__name__", "") != "auto_slash_factory"
