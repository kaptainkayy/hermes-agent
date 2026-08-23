import asyncio
import json
from unittest.mock import AsyncMock, patch
import pytest

from gateway.platforms.base import MessageType
from gateway.session import build_session_key
from tests.gateway.test_base_topic_sessions import DummyTelegramAdapter, TestTelegramAutoTtsCaptionDelivery

class TestSuppressAutoTtsGuard:
    @pytest.mark.asyncio
    async def test_voice_event_with_suppress_auto_tts_skips_base_auto_tts(self, tmp_path):
        adapter = DummyTelegramAdapter()
        adapter._keep_typing = TestTelegramAutoTtsCaptionDelivery._hold_typing()
        adapter._should_auto_tts_for_chat = lambda _chat_id: True
        adapter.play_tts = AsyncMock(return_value={"success": True})
        adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="Short reply"))

        tts_path = tmp_path / "reply.ogg"
        tts_path.write_text("audio", encoding="utf-8")
        event = TestTelegramAutoTtsCaptionDelivery._make_voice_event()
        setattr(event, "suppress_auto_tts", True)

        with patch("tools.tts_tool.check_tts_requirements", return_value=True) as mock_req, patch(
            "tools.tts_tool.text_to_speech_tool",
            return_value=json.dumps({"file_path": str(tts_path)}),
        ) as mock_tts:
            await adapter._process_message_background(event, build_session_key(event.source))

        mock_tts.assert_not_called()
        assert adapter.play_tts.await_count == 0
        assert len(adapter.sent) == 1
        assert adapter.sent[0]["content"] == "Short reply"

    @pytest.mark.asyncio
    async def test_voice_event_without_suppress_auto_tts_still_runs_base_auto_tts(self, tmp_path):
        adapter = DummyTelegramAdapter()
        adapter._keep_typing = TestTelegramAutoTtsCaptionDelivery._hold_typing()
        adapter._should_auto_tts_for_chat = lambda _chat_id: True
        adapter.play_tts = AsyncMock(return_value={"success": True})
        adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="Short reply"))

        tts_path = tmp_path / "reply.ogg"
        tts_path.write_text("audio", encoding="utf-8")
        event = TestTelegramAutoTtsCaptionDelivery._make_voice_event()

        with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
            "tools.tts_tool.text_to_speech_tool",
            return_value=json.dumps({"file_path": str(tts_path)}),
        ) as mock_tts:
            await adapter._process_message_background(event, build_session_key(event.source))

        mock_tts.assert_called_once()
        assert adapter.play_tts.await_count == 1

    @pytest.mark.asyncio
    async def test_text_event_with_suppress_auto_tts_does_not_enable_auto_tts(self, tmp_path):
        adapter = DummyTelegramAdapter()
        adapter._keep_typing = TestTelegramAutoTtsCaptionDelivery._hold_typing()
        adapter._should_auto_tts_for_chat = lambda _chat_id: True
        adapter.play_tts = AsyncMock(return_value={"success": True})
        adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="Short reply"))

        tts_path = tmp_path / "reply.ogg"
        tts_path.write_text("audio", encoding="utf-8")
        event = TestTelegramAutoTtsCaptionDelivery._make_voice_event()
        event.message_type = MessageType.TEXT
        setattr(event, "suppress_auto_tts", True)

        with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
            "tools.tts_tool.text_to_speech_tool",
            return_value=json.dumps({"file_path": str(tts_path)}),
        ) as mock_tts:
            await adapter._process_message_background(event, build_session_key(event.source))

        mock_tts.assert_not_called()
        assert adapter.play_tts.await_count == 0
        assert len(adapter.sent) == 1
        assert adapter.sent[0]["content"] == "Short reply"
