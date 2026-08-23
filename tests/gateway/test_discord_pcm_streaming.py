"""Regression tests for direct Orpheus PCM streaming into Discord voice playback."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import signal

import pytest


def test_pcm24_mono_source_resamples_to_discord_frame_size():
    from plugins.platforms.discord.adapter import PCM24MonoToDiscordAudioSource

    # 20 ms of 24 kHz mono s16le is 480 samples = 960 bytes.  Discord wants
    # 20 ms of 48 kHz stereo s16le = 3840 bytes.
    source = PCM24MonoToDiscordAudioSource(iter([b"\x00\x00" * 480]), tail_silence_ms=0)

    frame = source.read()

    assert len(frame) == 3840
    assert source.read() == b""
    assert source.is_opus() is False


def test_pcm24_mono_source_adds_tail_silence_to_avoid_clipped_endings():
    from plugins.platforms.discord.adapter import PCM24MonoToDiscordAudioSource

    source = PCM24MonoToDiscordAudioSource(iter([b"\x01\x00" * 480]), tail_silence_ms=40)

    assert len(source.read()) == 3840
    assert source.read() == b"\x00" * 3840
    assert source.read() == b"\x00" * 3840
    assert source.read() == b""


def test_pcm24_mono_source_default_tail_silence_is_300ms():
    from plugins.platforms.discord.adapter import PCM24MonoToDiscordAudioSource

    source = PCM24MonoToDiscordAudioSource(iter([b"\x01\x00" * 480]))

    assert len(source.read()) == 3840
    silence_frames = 0
    while True:
        frame = source.read()
        if frame == b"":
            break
        assert frame == b"\x00" * 3840
        silence_frames += 1
    assert silence_frames == 15


def test_pcm24_mono_source_combines_short_stream_chunks_without_hanging():
    from plugins.platforms.discord.adapter import PCM24MonoToDiscordAudioSource

    # Real HTTP streaming chunks are not aligned to Discord's 20 ms frame size.
    # A short first chunk must be topped up by reading later chunks, not spin
    # forever with a partial internal buffer.
    source = PCM24MonoToDiscordAudioSource(
        iter([
            b"\x01\x00" * 120,
            b"\x02\x00" * 360,
        ]),
        tail_silence_ms=0,
    )
    def timeout_handler(signum, frame):
        raise TimeoutError("read() hung on partial PCM chunks")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        frame = source.read()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

    assert len(frame) == 3840
    assert source.read() == b""


@pytest.mark.asyncio
async def test_pipeline_uses_direct_pcm_stream_for_orpheus_when_adapter_supports_it(monkeypatch):
    from gateway.voice_pipeline import DiscordVoicePipeline

    adapter = MagicMock()
    adapter.play_pcm_stream_in_voice_channel = AsyncMock(return_value=True)
    adapter.play_in_voice_channel = AsyncMock(return_value=True)

    with patch("tools.tts_tool.iter_orpheus_pcm_chunks", return_value=iter([b"\x00\x00" * 480])) as mock_pcm:
        pipeline = DiscordVoicePipeline(
            adapter=adapter,
            guild_id=12345,
            max_chars=80,
            tts_provider_override="orpheus",
        )
        pipeline.start()
        pipeline.feed("Direct stream this sentence! ")
        pipeline.flush()
        await pipeline.close()

    mock_pcm.assert_called_once()
    adapter.play_pcm_stream_in_voice_channel.assert_awaited_once()
    adapter.play_in_voice_channel.assert_not_called()
    assert pipeline.delivered_any is True


def test_discord_voice_prewarm_stt_env_guard(monkeypatch):
    from plugins.platforms.discord.adapter import _discord_voice_prewarm_stt_enabled

    monkeypatch.delenv("HERMES_DISCORD_VOICE_PREWARM_STT", raising=False)
    assert _discord_voice_prewarm_stt_enabled() is True

    for value in ("0", "false", "no", "off", "FALSE", "No"):
        monkeypatch.setenv("HERMES_DISCORD_VOICE_PREWARM_STT", value)
        assert _discord_voice_prewarm_stt_enabled() is False

    monkeypatch.setenv("HERMES_DISCORD_VOICE_PREWARM_STT", "true")
    assert _discord_voice_prewarm_stt_enabled() is True


@pytest.mark.asyncio
async def test_discord_voice_prewarm_scheduled_when_enabled(monkeypatch):
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("HERMES_DISCORD_VOICE_PREWARM_STT", "true")

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._voice_stt_prewarm_task = None

    captured = {}

    def fake_ensure_future(coro):
        captured["called"] = True
        captured["coro"] = coro
        return MagicMock(done=lambda: False)

    monkeypatch.setattr(asyncio, "ensure_future", fake_ensure_future)

    adapter._schedule_voice_stt_prewarm()

    assert captured.get("called") is True
    # The coroutine itself should be awaitable; awaiting it exercises the inner
    # import path without needing faster-whisper installed.
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread, \
         patch("tools.transcription_tools.prewarm_local_stt_model", return_value=True) as mock_prewarm:
        await captured["coro"]

    mock_to_thread.assert_awaited_once_with(mock_prewarm)


@pytest.mark.asyncio
async def test_discord_voice_prewarm_skipped_when_disabled(monkeypatch):
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("HERMES_DISCORD_VOICE_PREWARM_STT", "false")

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._voice_stt_prewarm_task = None

    captured = []

    def fake_ensure_future(coro):
        captured.append(coro)
        return MagicMock(done=lambda: False)

    monkeypatch.setattr(asyncio, "ensure_future", fake_ensure_future)

    adapter._schedule_voice_stt_prewarm()

    assert captured == []


@pytest.mark.asyncio
async def test_discord_voice_prewarm_avoids_duplicate_tasks(monkeypatch):
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("HERMES_DISCORD_VOICE_PREWARM_STT", "true")

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    existing_task = MagicMock(done=lambda: False)
    adapter._voice_stt_prewarm_task = existing_task

    captured = []

    def fake_ensure_future(coro):
        captured.append(coro)
        return MagicMock(done=lambda: False)

    monkeypatch.setattr(asyncio, "ensure_future", fake_ensure_future)

    adapter._schedule_voice_stt_prewarm()

    assert captured == []
