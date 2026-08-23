import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


class _FakeClient:
    def get_channel(self, channel_id):
        return None


class _FakeDiscordAdapter:
    def __init__(self):
        self._voice_text_channels = {123: 456}
        self._voice_sources = {}
        self._client = _FakeClient()
        self.played = []
        self.events = []

    async def play_in_voice_channel(self, guild_id, audio_path):
        self.played.append((guild_id, audio_path))
        return True

    async def handle_message(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_discord_voice_instant_ack_is_off_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_DISCORD_VOICE_INSTANT_ACK", raising=False)
    monkeypatch.delenv("HERMES_DISCORD_VOICE_ACK_PATH", raising=False)
    monkeypatch.setenv("HERMES_DISCORD_VOICE_BREVITY", "false")
    monkeypatch.setenv("HERMES_DISCORD_VOICE_WAKE_ENABLED", "false")

    runner = object.__new__(GatewayRunner)
    adapter = _FakeDiscordAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._is_user_authorized = lambda source: True
    runner._is_duplicate_voice_transcript = lambda guild_id, user_id, transcript: False

    await GatewayRunner._handle_voice_channel_input(runner, 123, 999, "hello")

    assert adapter.played == []
    assert len(adapter.events) == 1
    assert adapter.events[0].text == "hello"


@pytest.mark.asyncio
async def test_discord_voice_instant_ack_requires_explicit_enabled_path(tmp_path, monkeypatch):
    ack = tmp_path / "ack.ogg"
    ack.write_bytes(b"fake")
    monkeypatch.setenv("HERMES_DISCORD_VOICE_INSTANT_ACK", "true")
    monkeypatch.setenv("HERMES_DISCORD_VOICE_ACK_PATH", str(ack))
    monkeypatch.setenv("HERMES_DISCORD_VOICE_BREVITY", "false")
    monkeypatch.setenv("HERMES_DISCORD_VOICE_WAKE_ENABLED", "false")

    runner = object.__new__(GatewayRunner)
    adapter = _FakeDiscordAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._is_user_authorized = lambda source: True
    runner._is_duplicate_voice_transcript = lambda guild_id, user_id, transcript: False

    await GatewayRunner._handle_voice_channel_input(runner, 123, 999, "hello")

    # Let the scheduled ack task run.
    import asyncio

    await asyncio.sleep(0)
    assert adapter.played == [(123, str(ack))]
    assert len(adapter.events) == 1
