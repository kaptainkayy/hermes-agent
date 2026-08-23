"""Discord dynamic voice auto-join and empty-channel disconnect behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _AwaitableCancelTask:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def __await__(self):
        if False:
            yield None
        return None


def _make_adapter():
    from gateway.config import Platform, PlatformConfig
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._client = MagicMock()
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_empty_disconnect_tasks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_input_callback = None
    adapter._on_voice_disconnect = None
    adapter._allowed_user_ids = set()
    return adapter


def _channel(channel_id=444, guild_id=111, members=None):
    guild = SimpleNamespace(id=guild_id)
    return SimpleNamespace(id=channel_id, name="General", guild=guild, members=members or [])


def _member(user_id=222, *, bot=False, channel=None, guild_id=111):
    voice = SimpleNamespace(channel=channel) if channel is not None else None
    guild = SimpleNamespace(id=guild_id)
    return SimpleNamespace(id=user_id, bot=bot, display_name="KaptainKay", guild=guild, voice=voice)


def _runner():
    async def handle_voice_channel_input(**kwargs):
        return None

    def voice_key(platform, chat_id):
        return f"{platform.value}:{chat_id}"

    runner = SimpleNamespace(
        _voice_mode={},
        saved=False,
        enabled=[],
        _handle_voice_channel_input=handle_voice_channel_input,
        _handle_voice_timeout_cleanup=lambda chat_id: None,
        _voice_key=voice_key,
    )
    runner._save_voice_modes = lambda: setattr(runner, "saved", True)
    runner._set_adapter_auto_tts_enabled = lambda adapter, chat_id, enabled: runner.enabled.append((chat_id, enabled))
    return runner


@pytest.mark.asyncio
async def test_user_joining_configured_channel_makes_bot_join_and_maps_text(monkeypatch):
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = _make_adapter()
    target = _channel(444, 111, members=[])
    user = _member(222, channel=target)
    target.members = [user]
    before = SimpleNamespace(channel=None)
    after = SimpleNamespace(channel=target)

    monkeypatch.setenv("HERMES_DISCORD_AUTO_JOIN_CHANNEL_ID", "444")
    monkeypatch.setenv("HERMES_DISCORD_AUTO_JOIN_TEXT_CHANNEL_ID", "999")
    monkeypatch.setenv("HERMES_DISCORD_AUTO_JOIN_ON_USER", "true")

    with patch.object(DiscordAdapter, "join_voice_channel", new=AsyncMock(return_value=True)) as join_mock:
        await adapter._handle_voice_state_update(user, before, after)

    join_mock.assert_awaited_once_with(target)
    assert adapter._voice_text_channels[111] == 999


@pytest.mark.asyncio
async def test_auto_join_on_start_joins_configured_channel_and_maps_text(monkeypatch):
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = _make_adapter()
    runner = _runner()
    adapter.gateway_runner = runner
    target = _channel(444, 111, members=[])
    adapter._client.get_channel.return_value = target

    monkeypatch.setenv("HERMES_DISCORD_AUTO_JOIN_CHANNEL_ID", "444")
    monkeypatch.setenv("HERMES_DISCORD_AUTO_JOIN_TEXT_CHANNEL_ID", "999")
    monkeypatch.setenv("HERMES_DISCORD_AUTO_JOIN_ON_START", "true")

    with patch.object(DiscordAdapter, "join_voice_channel", new=AsyncMock(return_value=True)) as join_mock:
        await adapter._auto_join_on_start()

    join_mock.assert_awaited_once_with(target)
    assert adapter._voice_text_channels[111] == 999
    assert adapter._voice_input_callback is runner._handle_voice_channel_input
    assert adapter._voice_sources[111]["chat_id"] == "999"
    assert runner._voice_mode["discord:999"] == "all"
    assert runner.saved is True
    assert runner.enabled == [("999", True)]


@pytest.mark.asyncio
async def test_empty_configured_channel_schedules_30_second_disconnect(monkeypatch):
    adapter = _make_adapter()
    bot = _member(333, bot=True)
    target = _channel(444, 111, members=[bot])
    user = _member(222, channel=None)
    before = SimpleNamespace(channel=target)
    after = SimpleNamespace(channel=None)
    vc = MagicMock()
    vc.channel = target
    vc.is_connected.return_value = True
    adapter._voice_clients[111] = vc

    monkeypatch.setenv("HERMES_DISCORD_AUTO_JOIN_CHANNEL_ID", "444")
    monkeypatch.setenv("HERMES_DISCORD_EMPTY_DISCONNECT_SECONDS", "30")
    monkeypatch.setenv("HERMES_DISCORD_AUTO_JOIN_ON_USER", "true")

    created = []

    def fake_ensure_future(coro):
        try:
            coro.close()
        except AttributeError:
            pass
        task = _AwaitableCancelTask()
        created.append(task)
        return task

    with patch("plugins.platforms.discord.adapter.asyncio.ensure_future", side_effect=fake_ensure_future):
        await adapter._handle_voice_state_update(user, before, after)

    assert created, "empty-channel disconnect timer should be scheduled"
    assert adapter._voice_empty_disconnect_tasks[111] is created[0]


@pytest.mark.asyncio
async def test_user_rejoining_cancels_pending_empty_disconnect(monkeypatch):
    adapter = _make_adapter()
    pending = _AwaitableCancelTask()
    adapter._voice_empty_disconnect_tasks[111] = pending
    target = _channel(444, 111, members=[])
    user = _member(222, channel=target)
    target.members = [user]
    before = SimpleNamespace(channel=None)
    after = SimpleNamespace(channel=target)

    monkeypatch.setenv("HERMES_DISCORD_AUTO_JOIN_CHANNEL_ID", "444")
    monkeypatch.setenv("HERMES_DISCORD_AUTO_JOIN_ON_USER", "true")

    with patch.object(type(adapter), "join_voice_channel", new=AsyncMock(return_value=True)):
        await adapter._handle_voice_state_update(user, before, after)

    assert pending.cancelled is True
    assert 111 not in adapter._voice_empty_disconnect_tasks
