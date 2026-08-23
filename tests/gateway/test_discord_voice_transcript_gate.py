import pytest

from plugins.platforms.discord.adapter import DiscordAdapter, VoiceReceiver


@pytest.mark.asyncio
async def test_discord_voice_rejects_unusable_stt_transcript(monkeypatch, tmp_path):
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_input_callback = None
    called = []

    async def callback(**kwargs):
        called.append(kwargs)

    adapter._voice_input_callback = callback

    monkeypatch.setattr(VoiceReceiver, "pcm_to_wav", lambda pcm, path: None)
    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path: {
            "success": True,
            "transcript": "Hello everyone, welcome to my channel.",
            "language": "en",
            "duration": 0.46,
            "segments": [{"avg_logprob": -0.2, "no_speech_prob": 0.05}],
        },
    )

    pcm = b"\0" * int(0.46 * VoiceReceiver.SAMPLE_RATE * VoiceReceiver.CHANNELS * 2)
    await DiscordAdapter._process_voice_input(adapter, 123, 456, pcm)

    assert called == []


@pytest.mark.asyncio
async def test_discord_voice_accepts_usable_stt_transcript(monkeypatch):
    adapter = object.__new__(DiscordAdapter)
    called = []

    async def callback(**kwargs):
        called.append(kwargs)

    adapter._voice_input_callback = callback

    monkeypatch.setattr(VoiceReceiver, "pcm_to_wav", lambda pcm, path: None)
    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path: {
            "success": True,
            "transcript": "Fix that.",
            "language": "en",
            "duration": 0.9,
            "segments": [{"avg_logprob": -0.2, "no_speech_prob": 0.05}],
        },
    )

    pcm = b"\0" * int(0.9 * VoiceReceiver.SAMPLE_RATE * VoiceReceiver.CHANNELS * 2)
    await DiscordAdapter._process_voice_input(adapter, 123, 456, pcm)

    assert called == [{"guild_id": 123, "user_id": 456, "transcript": "Fix that."}]
