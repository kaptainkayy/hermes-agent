"""Tests for Discord voice token-budget configuration."""

import gateway.run as gateway_run


def test_discord_voice_max_tokens_defaults_to_short_spoken_budget(monkeypatch):
    monkeypatch.delenv("HERMES_DISCORD_VOICE_MAX_TOKENS", raising=False)

    assert gateway_run._discord_voice_max_tokens() == 80


def test_discord_voice_max_tokens_invalid_env_falls_back_to_safe_budget(monkeypatch):
    monkeypatch.setenv("HERMES_DISCORD_VOICE_MAX_TOKENS", "not-an-int")

    assert gateway_run._discord_voice_max_tokens() == 80


def test_discord_voice_max_tokens_respects_explicit_env(monkeypatch):
    monkeypatch.setenv("HERMES_DISCORD_VOICE_MAX_TOKENS", "512")

    assert gateway_run._discord_voice_max_tokens() == 512


def test_discord_voice_route_has_safe_defaults(monkeypatch):
    monkeypatch.delenv("HERMES_DISCORD_VOICE_MODEL", raising=False)
    monkeypatch.delenv("HERMES_DISCORD_VOICE_PROVIDER", raising=False)

    assert gateway_run._discord_voice_model() == "google/gemini-3.1-flash-lite"
    assert gateway_run._discord_voice_provider() == "openrouter"


def test_discord_voice_prompt_forces_english_and_warmth(monkeypatch):
    monkeypatch.delenv("HERMES_DISCORD_VOICE_FAST_PROMPT", raising=False)

    prompt = gateway_run._discord_voice_fast_prompt().lower()

    assert "english only" in prompt
    assert "warm" in prompt
    assert "one short sentence" in prompt


def test_deterministic_count_response_handles_count_to_100():
    response = gateway_run.GatewayRunner._deterministic_count_response("please count to 100")

    assert response is not None
    assert response.startswith("one, two, three")
    assert response.endswith("ninety eight, ninety nine, one hundred.")


def test_deterministic_count_chunks_bypass_llm_chunker():
    chunks = gateway_run.GatewayRunner._deterministic_count_chunks("count from one to one hundred")

    assert chunks is not None
    assert len(chunks) == 100
    assert chunks[0] == "one,"
    assert "twenty two, twenty three, twenty four," not in chunks
    assert "twenty three, twenty four," not in chunks
    assert chunks[-1] == "one hundred."


def test_deterministic_count_target_accepts_count_1_to_100():
    assert gateway_run.GatewayRunner._deterministic_count_target("count 1 to 100") == 100


def test_deterministic_count_target_catches_stt_count_100_variants():
    assert gateway_run.GatewayRunner._deterministic_count_target("count 100") == 100
    assert gateway_run.GatewayRunner._deterministic_count_target("can you count one hundred") == 100
    assert gateway_run.GatewayRunner._deterministic_count_target("Talk to 100.") == 100


def test_deterministic_count_response_ignores_non_count_requests():
    assert gateway_run.GatewayRunner._deterministic_count_response("what is one hundred minus five?") is None


def test_orpheus_voice_command_lists_voices():
    response = gateway_run.GatewayRunner._orpheus_voice_command_response("list all voices")

    assert response == "Available Persephone voices are: tara, leah, jess, leo, dan, mia, zac, zoe, julia."


def test_orpheus_voice_command_rejects_unknown_voice():
    response = gateway_run.GatewayRunner._orpheus_voice_command_response("switch to bob")

    assert response is not None
    assert response.startswith("I don't know the Persephone voice bob.")


def test_orpheus_voice_command_switches_config_voice(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "tts:\n"
        "  provider: orpheus\n"
        "  orpheus:\n"
        "    base_url: http://127.0.0.1:5005/v1\n"
        "    voice: auto\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    response = gateway_run.GatewayRunner._orpheus_voice_command_response("switch to julia")

    assert response == "Switched Persephone voice to julia."
    updated = config_path.read_text(encoding="utf-8")
    assert "voice: julia" in updated
