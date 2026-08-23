"""Tests for the Orpheus TTS provider in tools/tts_tool.py."""

import uuid
from unittest.mock import MagicMock, patch
import pytest

@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # Ensure no conflicting environment variables are set
    for key in (
        "OPENAI_API_KEY",
        "HERMES_SESSION_PLATFORM",
        "HERMES_ORPHEUS_REPETITION_PENALTY",
    ):
        monkeypatch.delenv(key, raising=False)


class TestGenerateOrpheusTts:
    def _run(self, text, output_path, tts_config):
        mock_response = MagicMock()
        mock_client = MagicMock()
        mock_client.audio.speech.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client)

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls):
            from tools.tts_tool import _generate_orpheus_tts
            _generate_orpheus_tts(text, output_path, tts_config)

        return mock_cls, mock_client.audio.speech.create

    def test_default_orpheus_config(self, tmp_path):
        """Orpheus uses expected default base_url, api_key, model, voice, response_format."""
        out_path = str(tmp_path / "out.mp3")
        client_cls, create = self._run("Hello", out_path, {})

        client_cls.assert_called_once_with(api_key="not-needed", base_url="http://localhost:5005/v1")
        kwargs = create.call_args[1]
        assert kwargs["model"] == "not-needed"
        assert kwargs["voice"] == "tara"
        assert kwargs["input"] == "Hello"
        assert kwargs["response_format"] == "mp3"
        assert "speed" not in kwargs

    def test_format_derived_from_extension(self, tmp_path):
        """wav -> wav, ogg -> opus, other -> mp3."""
        # Test wav
        _, create_wav = self._run("Hello", str(tmp_path / "out.wav"), {})
        assert create_wav.call_args[1]["response_format"] == "wav"

        # Test ogg
        _, create_ogg = self._run("Hello", str(tmp_path / "out.ogg"), {})
        assert create_ogg.call_args[1]["response_format"] == "opus"

        # Test other/mp3
        _, create_mp3 = self._run("Hello", str(tmp_path / "out.xyz"), {})
        assert create_mp3.call_args[1]["response_format"] == "mp3"

    def test_config_overrides(self, tmp_path):
        """tts.orpheus configurations override defaults."""
        config = {
            "orpheus": {
                "base_url": "http://orpheus:5000/v1",
                "api_key": "custom-key",
                "model": "custom-model",
                "voice": "custom-voice",
                "speed": 1.25,
            }
        }
        out_path = str(tmp_path / "out.wav")
        client_cls, create = self._run("Hello", out_path, config)

        client_cls.assert_called_once_with(api_key="custom-key", base_url="http://orpheus:5000/v1")
        kwargs = create.call_args[1]
        assert kwargs["model"] == "custom-model"
        assert kwargs["voice"] == "custom-voice"
        assert kwargs["speed"] == 1.25

    def test_speed_fallback_global(self, tmp_path):
        """Global tts.speed works as fallback if tts.orpheus.speed not specified."""
        config = {
            "speed": 1.5,
            "orpheus": {}
        }
        out_path = str(tmp_path / "out.mp3")
        _, create = self._run("Hello", out_path, config)
        assert create.call_args[1]["speed"] == 1.5

    def test_speed_clamp(self, tmp_path):
        """Speed is clamped between 0.25 and 4.0."""
        config = {
            "orpheus": {"speed": 0.1}
        }
        _, create_low = self._run("Hello", str(tmp_path / "out1.mp3"), config)
        assert create_low.call_args[1]["speed"] == 0.25

        config = {
            "orpheus": {"speed": 5.0}
        }
        _, create_high = self._run("Hello", str(tmp_path / "out2.mp3"), config)
        assert create_high.call_args[1]["speed"] == 4.0

    # -----------------------------------------------------------------------
    # repetition_penalty — file path path (_generate_orpheus_tts)
    # -----------------------------------------------------------------------

    def test_repetition_penalty_not_sent_by_default(self, tmp_path):
        """No config or env => no repetition_penalty in create_kwargs."""
        _, create = self._run("Hello", str(tmp_path / "out.mp3"), {})
        kwargs = create.call_args[1]
        assert "extra_body" not in kwargs

    def test_repetition_penalty_from_env(self, tmp_path, monkeypatch):
        """HERMES_ORPHEUS_REPETITION_PENALTY env var adds extra_body."""
        monkeypatch.setenv("HERMES_ORPHEUS_REPETITION_PENALTY", "1.35")
        _, create = self._run("Hello", str(tmp_path / "out.mp3"), {})
        kwargs = create.call_args[1]
        assert kwargs["extra_body"] == {"repetition_penalty": 1.35}

    def test_repetition_penalty_from_config(self, tmp_path):
        """tts.orpheus.repetition_penalty in config adds extra_body."""
        config = {"orpheus": {"repetition_penalty": 1.5}}
        _, create = self._run("Hello", str(tmp_path / "out.mp3"), config)
        kwargs = create.call_args[1]
        assert kwargs["extra_body"] == {"repetition_penalty": 1.5}

    def test_repetition_penalty_env_overrides_config(self, tmp_path, monkeypatch):
        """Env var wins over config value."""
        monkeypatch.setenv("HERMES_ORPHEUS_REPETITION_PENALTY", "1.8")
        config = {"orpheus": {"repetition_penalty": 1.2}}
        _, create = self._run("Hello", str(tmp_path / "out.mp3"), config)
        kwargs = create.call_args[1]
        assert kwargs["extra_body"] == {"repetition_penalty": 1.8}

    def test_repetition_penalty_invalid_env_logs_warning(self, tmp_path, monkeypatch, caplog):
        """Invalid non-float env logs warning and does not send repetition_penalty."""
        monkeypatch.setenv("HERMES_ORPHEUS_REPETITION_PENALTY", "not-a-float")
        _, create = self._run("Hello", str(tmp_path / "out.mp3"), {})
        kwargs = create.call_args[1]
        assert "extra_body" not in kwargs
        assert any("repetition_penalty" in msg and "not-a-float" in msg for msg in caplog.messages)

    def test_repetition_penalty_clamp_low(self, tmp_path, monkeypatch):
        """Values below 1.0 are clamped to 1.0."""
        monkeypatch.setenv("HERMES_ORPHEUS_REPETITION_PENALTY", "0.5")
        _, create = self._run("Hello", str(tmp_path / "out.mp3"), {})
        kwargs = create.call_args[1]
        assert kwargs["extra_body"] == {"repetition_penalty": 1.0}

    def test_repetition_penalty_clamp_high(self, tmp_path, monkeypatch):
        """Values above 2.0 are clamped to 2.0."""
        monkeypatch.setenv("HERMES_ORPHEUS_REPETITION_PENALTY", "3.0")
        _, create = self._run("Hello", str(tmp_path / "out.mp3"), {})
        kwargs = create.call_args[1]
        assert kwargs["extra_body"] == {"repetition_penalty": 2.0}

    def test_repetition_penalty_at_min_edge(self, tmp_path, monkeypatch):
        """1.0 is exactly the minimum boundary and passes through."""
        monkeypatch.setenv("HERMES_ORPHEUS_REPETITION_PENALTY", "1.0")
        _, create = self._run("Hello", str(tmp_path / "out.mp3"), {})
        kwargs = create.call_args[1]
        assert kwargs["extra_body"] == {"repetition_penalty": 1.0}

    def test_repetition_penalty_at_max_edge(self, tmp_path, monkeypatch):
        """2.0 is exactly the maximum boundary and passes through."""
        monkeypatch.setenv("HERMES_ORPHEUS_REPETITION_PENALTY", "2.0")
        _, create = self._run("Hello", str(tmp_path / "out.mp3"), {})
        kwargs = create.call_args[1]
        assert kwargs["extra_body"] == {"repetition_penalty": 2.0}


class TestIterOrpheusPcmChunks:
    """Tests for iter_orpheus_pcm_chunks()."""

    def _run(self, text, tts_config):
        """Run iter_orpheus_pcm_chunks and return the JSON payload that was POSTed."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.iter_content.return_value = [b"pcmdata"]
        mock_post = MagicMock(return_value=mock_response)

        from tools.tts_tool import iter_orpheus_pcm_chunks
        with patch("requests.post", mock_post):
            list(iter_orpheus_pcm_chunks(text, tts_config))

        return mock_post.call_args[1]["json"]

    def test_default_no_repetition_penalty(self):
        """No config => no repetition_penalty in JSON payload."""
        payload = self._run("Hello", {})
        assert "repetition_penalty" not in payload

    def test_pcm_repetition_penalty_from_env(self, monkeypatch):
        """Env var adds repetition_penalty to the streaming payload."""
        monkeypatch.setenv("HERMES_ORPHEUS_REPETITION_PENALTY", "1.5")
        payload = self._run("Hello", {})
        assert payload["repetition_penalty"] == 1.5

    def test_pcm_repetition_penalty_from_config(self):
        """Config adds repetition_penalty to the streaming payload."""
        payload = self._run("Hello", {"orpheus": {"repetition_penalty": 1.7}})
        assert payload["repetition_penalty"] == 1.7

    def test_pcm_repetition_penalty_env_overrides_config(self, monkeypatch):
        """Env overrides config for streaming payload."""
        monkeypatch.setenv("HERMES_ORPHEUS_REPETITION_PENALTY", "1.9")
        payload = self._run("Hello", {"orpheus": {"repetition_penalty": 1.2}})
        assert payload["repetition_penalty"] == 1.9

    def test_pcm_invalid_env_logs_warning(self, monkeypatch, caplog):
        """Invalid env does not send repetition_penalty."""
        monkeypatch.setenv("HERMES_ORPHEUS_REPETITION_PENALTY", "bad")
        payload = self._run("Hello", {})
        assert "repetition_penalty" not in payload
        assert any("repetition_penalty" in msg and "bad" in msg for msg in caplog.messages)

    def test_pcm_clamp_low(self, monkeypatch):
        """Below 1.0 clamped to 1.0."""
        monkeypatch.setenv("HERMES_ORPHEUS_REPETITION_PENALTY", "0.3")
        payload = self._run("Hello", {})
        assert payload["repetition_penalty"] == 1.0

    def test_pcm_clamp_high(self, monkeypatch):
        """Above 2.0 clamped to 2.0."""
        monkeypatch.setenv("HERMES_ORPHEUS_REPETITION_PENALTY", "2.5")
        payload = self._run("Hello", {})
        assert payload["repetition_penalty"] == 2.0


class TestResolveOrpheusProfileVoice:
    """Tests for _resolve_orpheus_profile_voice()."""

    @pytest.fixture
    def profile_yaml(self, tmp_path):
        """Create a temporary Orpheus voice profile YAML for testing."""
        import yaml
        data = {
            "default_voice": "tara",
            "fallback_profile": "default",
            "profiles": {
                "default": {"voice": "tara", "aliases": ["favorite", "tara"]},
                "calm": {"voice": "leah", "aliases": ["soothing"]},
                "epic_story": {"voice": "leo", "aliases": ["epic"]},
            },
            "routing_rules": [
                {"profile": "epic_story", "keywords": ["use epic voice", "epic voice"]},
                {"profile": "calm", "keywords": ["use calm voice", "calm down"]},
            ],
        }
        path = tmp_path / "orpheus_voice_profiles.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        return str(path)

    def _resolve(self, requested_voice: str = "auto", text: str = "", profile_path: str = ""):
        from tools.tts_tool import _resolve_orpheus_profile_voice
        return _resolve_orpheus_profile_voice(requested_voice, text, profile_path)

    def test_explicit_voice_passes_through(self):
        """Explicit voice names like 'tara' or 'leah' are returned unchanged."""
        voice, label = self._resolve("tara")
        assert voice == "tara"
        assert label == "explicit"

        voice, label = self._resolve("leah")
        assert voice == "leah"
        assert label == "explicit"

    def test_auto_without_profile_returns_default(self):
        """With 'auto' and no profile file, defaults to 'tara'."""
        voice, label = self._resolve("auto", "Hello world")
        assert voice == "tara"
        assert label == "default"

    def test_auto_with_profile_defaults_to_fallback(self, profile_yaml):
        """With 'auto' and profile file but no keywords matched, uses fallback profile."""
        voice, label = self._resolve("auto", "Hello world", profile_yaml)
        assert voice == "tara"
        assert label == "default"

    def test_auto_with_keyword_epic(self, profile_yaml):
        """'use epic voice' in text matches the epic_story profile -> leo."""
        voice, label = self._resolve("auto", "use epic voice to tell this story", profile_yaml)
        assert voice == "leo"
        assert label == "profile:epic_story"

    def test_auto_with_keyword_calm(self, profile_yaml):
        """'use calm voice' in text matches the calm profile -> leah."""
        voice, label = self._resolve("auto", "use calm voice for this", profile_yaml)
        assert voice == "leah"
        assert label == "profile:calm"

    def test_auto_with_keyword_calm_down(self, profile_yaml):
        """'calm down' keyword also matches calm profile."""
        voice, label = self._resolve("auto", "Please calm down", profile_yaml)
        assert voice == "leah"
        assert label == "profile:calm"

    def test_missing_profile_file_falls_back_gracefully(self):
        """Non-existent profile path -> default tara without raising."""
        voice, label = self._resolve("auto", "hello", "/nonexistent/profiles.yaml")
        assert voice == "tara"
        assert label == "default"
