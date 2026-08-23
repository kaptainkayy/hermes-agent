"""Unit tests for live voice phrase chunking and pipeline helpers in gateway/voice_pipeline.py."""

import asyncio
import json
import os
from unittest.mock import ANY, AsyncMock, MagicMock, patch
import pytest

# Since the module may not exist yet, we write tests that will import it once created.

def test_text_chunker_punctuation():
    """Test chunker splits text correctly on natural punctuation boundaries."""
    from gateway.voice_pipeline import TextChunker
    chunker = TextChunker(max_chars=100)

    # Simple sentence boundaries
    chunks = chunker.feed("Hello there! ")
    assert chunks == ["Hello there!"]

    # Comma should NOT be a chunk boundary by default unless max_chars is hit
    chunks = chunker.feed("This is a long sentence, ")
    assert chunks == []

    # Question mark split
    chunks = chunker.feed("how are you? Yes, fine.")
    assert chunks == ["This is a long sentence, how are you?", "Yes, fine."]

    # Flush tail
    chunks = chunker.feed(" And a final sentence")
    assert chunks == []
    chunks = chunker.flush()
    assert chunks == ["And a final sentence"]


def test_text_chunker_max_chars():
    """Test chunker splits text correctly on max characters when no punctuation is present."""
    from gateway.voice_pipeline import TextChunker
    chunker = TextChunker(max_chars=20)

    # Feed a continuous string without punctuation
    chunks = chunker.feed("This is a very long string without punctuation")
    # "This is a very long " has 20 characters, split at space
    # "This is a very long" is 19 chars
    assert len(chunks) >= 1
    assert chunks[0] == "This is a very long"


def test_text_chunker_target_words_prefers_natural_pause():
    """Target-word chunking avoids tiny mid-sentence Orpheus chunks."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=160, target_words=20)
    chunks = chunker.feed(
        "I don't have access to a web search tool or live terminal in this "
        "session to pull the latest headlines for you. I can only search "
        "our past conversations, and we haven't discussed current news "
        "regarding Iran yet."
    )

    assert chunks == [
        "I don't have access to a web search tool or live terminal in this session to pull the latest headlines for you.",
        "I can only search our past conversations, and we haven't discussed current news regarding Iran yet.",
    ]


def test_text_chunker_target_words_prefers_comma_near_target():
    """Long sentence chunks can split on commas instead of arbitrary words."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=160, target_words=20)
    chunks = chunker.feed(
        "I don't have a live weather tool active at the moment to give you "
        "this week's forecast for Fayetteville, but I can tell you it was "
        "sunny and seventy-six degrees there earlier today."
    )

    assert chunks[0].endswith("Fayetteville,")
    assert len(chunks[0].split()) >= 15


def test_text_chunker_numeric_lists_use_small_chunks():
    """Counting lists group naturally so Orpheus does not drift or sound robotic."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=100, target_words=10)
    chunks = chunker.feed(
        "One, two, three, four, five, six, seven, eight, nine, ten, "
        "eleven, twelve, thirteen, fourteen, fifteen,"
    )

    assert chunks == [
        "One, two, three,",
        "four, five, six,",
        "seven, eight, nine,",
        "ten, eleven, twelve,",
        "thirteen, fourteen, fifteen,",
    ]


def test_text_chunker_short_comma_lists_use_bounded_chunks():
    """Long name/list recitations should not become giant Orpheus requests."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=160, target_words=20)
    chunks = chunker.feed(
        "Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, "
        "Delaware, Florida, Georgia, Hawaii, Idaho, Illinois, Indiana, Iowa, Kansas."
    )
    chunks.extend(chunker.flush())

    assert chunks == [
        "Alabama, Alaska, Arizona, Arkansas, California, Colorado,",
        "Connecticut, Delaware, Florida, Georgia, Hawaii, Idaho,",
        "Illinois, Indiana, Iowa, Kansas.",
    ]


def test_normalize_voice_text_repairs_missing_space_after_comma():
    """Chunk boundaries and provider output should not produce comma-joined words."""
    from gateway.voice_pipeline import normalize_voice_text

    assert normalize_voice_text("Nevada,New Hampshire,Jefferson City,Helena") == (
        "Nevada, New Hampshire, Jefferson City, Helena"
    )


def test_orpheus_warm_style_preset_does_not_add_nonverbal_noise(monkeypatch):
    """Preset names should not inject audible gasps/chuckles before every chunk."""
    from tools.tts_tool import _apply_orpheus_style_prefix

    monkeypatch.delenv("HERMES_ORPHEUS_STYLE_PREFIX", raising=False)

    assert _apply_orpheus_style_prefix(
        "I hear you loud and clear.",
        {"orpheus": {"style_preset": "warm"}},
    ) == "I hear you loud and clear."


def test_orpheus_explicit_style_prefix_still_applies(monkeypatch):
    from tools.tts_tool import _apply_orpheus_style_prefix

    monkeypatch.delenv("HERMES_ORPHEUS_STYLE_PREFIX", raising=False)

    assert _apply_orpheus_style_prefix(
        "I hear you loud and clear.",
        {"orpheus": {"style_prefix": "<chuckle>"}},
    ) == "<chuckle> I hear you loud and clear."


def test_orpheus_warm_style_prefix_skips_counts_and_lists(monkeypatch):
    """Style tags should not contaminate numeric/list recitations."""
    from tools.tts_tool import _apply_orpheus_style_prefix

    monkeypatch.delenv("HERMES_ORPHEUS_STYLE_PREFIX", raising=False)
    cfg = {"orpheus": {"style_preset": "warm"}}

    assert _apply_orpheus_style_prefix("one, two, three,", cfg) == "one, two, three,"
    assert _apply_orpheus_style_prefix(
        "Alabama, Alaska, Arizona, Arkansas, California, Colorado,",
        cfg,
    ) == "Alabama, Alaska, Arizona, Arkansas, California, Colorado,"


def test_text_chunker_numeric_list_ignores_trailing_punctuation_fragment():
    """Splitting exact numeric groups should not enqueue a standalone period."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=100, target_words=10)
    chunks = chunker.feed("forty-six, forty-seven, forty-eight, forty-nine, fifty.")
    chunks.extend(chunker.flush())

    assert chunks == ["forty six, forty seven, forty eight,", "forty nine, fifty."]


def test_text_chunker_count_to_100_keeps_one_hundred_together():
    """Counting to 100 must not split the final number into an orphan 'hundred.' chunk."""
    from gateway.voice_pipeline import TextChunker

    count_to_100 = (
        "one, two, three, four, five, six, seven, eight, nine, ten, "
        "eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen, twenty, "
        "twenty one, twenty two, twenty three, twenty four, twenty five, "
        "twenty six, twenty seven, twenty eight, twenty nine, thirty, "
        "thirty one, thirty two, thirty three, thirty four, thirty five, "
        "thirty six, thirty seven, thirty eight, thirty nine, forty, "
        "forty one, forty two, forty three, forty four, forty five, "
        "forty six, forty seven, forty eight, forty nine, fifty, "
        "fifty one, fifty two, fifty three, fifty four, fifty five, "
        "fifty six, fifty seven, fifty eight, fifty nine, sixty, "
        "sixty one, sixty two, sixty three, sixty four, sixty five, "
        "sixty six, sixty seven, sixty eight, sixty nine, seventy, "
        "seventy one, seventy two, seventy three, seventy four, seventy five, "
        "seventy six, seventy seven, seventy eight, seventy nine, eighty, "
        "eighty one, eighty two, eighty three, eighty four, eighty five, "
        "eighty six, eighty seven, eighty eight, eighty nine, ninety, "
        "ninety one, ninety two, ninety three, ninety four, ninety five, "
        "ninety six, ninety seven, ninety eight, ninety nine, one hundred."
    )
    chunker = TextChunker(max_chars=100, target_words=10)
    chunks = chunker.feed(count_to_100)
    chunks.extend(chunker.flush())

    assert len(chunks) == 34
    assert chunks[-1] == "one hundred."
    assert "hundred." not in chunks


def test_text_chunker_count_backward_groups_numbers_naturally():
    """A backward count must not become one tiny TTS request per number."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=100, target_words=10)
    chunks = chunker.feed(
        "one hundred, ninety nine, ninety eight, ninety seven, ninety six, "
        "ninety five, ninety four, ninety three, ninety two, ninety one,"
    )
    chunks.extend(chunker.flush())

    assert chunks == [
        "one hundred, ninety nine, ninety eight,",
        "ninety seven, ninety six, ninety five,",
        "ninety four, ninety three, ninety two,",
        "ninety one,",
    ]


def test_text_chunker_digit_count_sequences_group_naturally():
    """Provider fallback digit lists should not produce giant final TTS chunks."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=100, target_words=10)
    chunks = chunker.feed("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15.")
    chunks.extend(chunker.flush())

    assert chunks == [
        "one two",
        "three four",
        "five six",
        "seven eight",
        "nine ten",
        "eleven twelve",
        "thirteen fourteen",
        "fifteen.",
    ]


def test_text_chunker_digit_count_commas_are_spoken_as_words():
    """Bare digit count chunks make Orpheus drop numbers; speak words instead."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=100, target_words=10)
    chunks = chunker.feed("1, 2, 3, 4, 5, 6,")
    chunks.extend(chunker.flush())

    assert chunks == ["one, two, three,", "four, five, six,"]


def test_text_chunker_keeps_decimals_together():
    """Decimal numbers must not split into orphan TTS chunks like '0.' then '8'."""
    from gateway.voice_pipeline import TextChunker

    assert [match.group(0) for match in TextChunker._word_spans("from 0.8 to about 1.5 seconds")] == [
        "from",
        "0.8",
        "to",
        "about",
        "1.5",
        "seconds",
    ]

    chunker = TextChunker(max_chars=80, target_words=10)
    chunks = chunker.feed("Barge-in: from 0.8 to about 1.5 seconds. The timer is set to 0.8 seconds.")
    chunks.extend(chunker.flush())

    assert "0." not in chunks
    assert "8 to about 1." not in chunks
    assert chunks == ["Barge-in: from 0.8 to about 1.5 seconds.", "The timer is set to 0.8 seconds."]


def test_text_chunker_merges_tiny_structure_fragments():
    """Orpheus should not receive standalone list markers or dangling transition words."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=160, target_words=20)
    assert chunker.feed("1. ") == []
    chunks = chunker.feed("Architecture and voice: we verified Orpheus is the speaker. ")
    assert chunks == ["First, architecture and voice: we verified Orpheus is the speaker."]

    assert chunker.feed("Essentially") == []
    assert chunker.flush() == ["Essentially."]


def test_text_chunker_min_chunk_words_default_disabled():
    """Default min_chunk_words keeps existing punctuation splitting behavior."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=100)

    assert chunker.min_chunk_words == 0
    assert chunker.feed("This short sentence has six words. ") == ["This short sentence has six words."]


def test_text_chunker_min_chunk_words_holds_short_midstream_sentence():
    """Short mid-stream sentences wait for more words instead of resetting prosody."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=100, min_chunk_words=12)

    assert chunker.feed("This short sentence has six words. ") == []
    assert chunker.feed("Another sentence adds enough extra words today. ") == [
        "This short sentence has six words. Another sentence adds enough extra words today."
    ]


def test_text_chunker_min_chunk_words_flush_emits_short_tail():
    """Final tails are emitted even when shorter than the mid-stream floor."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=100, min_chunk_words=12)

    assert chunker.feed("This short sentence has six words. ") == []
    assert chunker.flush() == ["This short sentence has six words."]


def test_text_chunker_min_chunk_words_does_not_block_numeric_lists():
    """Numeric list splits keep their configured grouping despite min_chunk_words."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=100, target_words=10, numeric_list_target=3, min_chunk_words=12)

    assert chunker.feed("One, two, three, four, five, six,") == ["One, two, three,", "four, five, six,"]


def test_text_chunker_waits_on_streamed_partial_word_fragment():
    """Do not speak chunks like 'someone who can p' before the next delta completes the word."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=40, target_words=8)

    assert chunker.feed("Please reach out to someone who can p ") == []
    chunks = chunker.feed("rovide physical support.")
    chunks.extend(chunker.flush())

    assert "Please reach out to someone who can p" not in chunks
    assert "provide physical support." in " ".join(chunks)


def test_normalize_voice_text_strips_markdown_and_spoken_list_markers():
    """Voice text keeps the warm response content but removes markdown/code/list artifacts."""
    from gateway.voice_pipeline import normalize_voice_text

    text = "1. **Architecture & Voice:** We tested `heart_young` and 76°F.\n2. **Next:** tune barge-in."
    assert normalize_voice_text(text) == (
        "First, architecture and voice: We tested heart young and seventy six degrees Fahrenheit. "
        "Second, next: tune barge-in."
    )


def test_normalize_voice_text_strips_streamed_partial_markdown():
    """Streaming chunks can split bold/list markdown before the closing token arrives."""
    from gateway.voice_pipeline import TextChunker, normalize_voice_text

    assert normalize_voice_text("**1.") == "1."
    assert normalize_voice_text("System and Environment Control** I can help.") == (
        "System and Environment Control I can help."
    )

    chunker = TextChunker(max_chars=160, target_words=20)
    assert chunker.feed("**1. ") == []
    assert chunker.feed("System and Environment Control** I can help. ") == [
        "First, system and Environment Control I can help."
    ]


def test_normalize_voice_text_dehyphenates_number_words_only():
    """Hyphenated number words can make Orpheus over-generate counting chunks."""
    from gateway.voice_pipeline import normalize_voice_text

    assert normalize_voice_text("forty-six, forty-seven") == "forty six, forty seven"
    assert normalize_voice_text("low-latency audio") == "low-latency audio"


def test_normalize_voice_text_strips_leaked_chat_labels():
    """Continuation/session labels must never be spoken by live TTS."""
    from gateway.voice_pipeline import normalize_voice_text

    assert normalize_voice_text("Crayola crayons, first introducedAssistant (me):") == (
        "Crayola crayons, first introduced"
    )
    assert normalize_voice_text("Assistant: here is the answer.") == "here is the answer."


def test_normalize_voice_text_repairs_repetition_and_missing_space_join():
    """Stream-splice artifacts should not be spoken literally."""
    from gateway.voice_pipeline import normalize_voice_text

    text = "drop audio, or just make you waitor just make you wait for me to stop."

    assert normalize_voice_text(text) == "drop audio, or just make you wait for me to stop."


def test_voice_control_messages_are_suppressed_before_tts():
    """Internal orchestration sentinels must never be spoken through voice TTS."""
    from gateway.voice_pipeline import TextChunker, is_voice_control_message, prepare_voice_response_text

    message = "Operation interrupted: waiting for model response (12.4s elapsed)."

    assert is_voice_control_message(message)
    assert prepare_voice_response_text(message, max_words=100, max_sentences=3) == ""

    chunker = TextChunker(max_chars=80, target_words=20)
    assert chunker.feed(message) == []
    assert chunker.flush() == []


def test_prepare_voice_response_text_trims_at_sentence_boundary():
    """Final-only voice TTS should not read long walls of text by default."""
    from gateway.voice_pipeline import prepare_voice_response_text

    text = (
        "First sentence is useful. Second sentence is also useful. Third sentence still fits. "
        "Fourth sentence should not be spoken by default."
    )

    assert prepare_voice_response_text(text, max_words=100, max_sentences=3) == (
        "First sentence is useful. Second sentence is also useful. Third sentence still fits."
    )


def test_prepare_voice_response_text_trims_long_first_sentence_without_mid_word_cut():
    """If one sentence is too long, trim on a word boundary and end cleanly."""
    from gateway.voice_pipeline import prepare_voice_response_text

    assert prepare_voice_response_text(
        "one two three four five six seven eight nine ten eleven", max_words=5, max_sentences=3
    ) == "one two three four five."


def test_text_chunker_cleans_leading_punctuation_from_spoken_chunks():
    """Live TTS chunks should not start with punctuation-only fragments."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=80, target_words=20)

    assert chunker.feed(", I can hear you perfectly. ") == ["I can hear you perfectly."]


def test_text_chunker_never_cuts_mid_word_on_max_chars_fallback():
    """Long no-comma deltas must wait for whitespace instead of slicing mid-word.

    Regression: a state/capital enumeration streamed as one delta would hit
    the ``len(buffer) >= max_chars`` branch with no space inside the window
    and clamp at ``max_chars``, producing chunks like ``Kentucky, Lou`` and
    ``Orego``. The fix waits for the next delta to deliver a whitespace
    boundary unless the buffer grows past 3x ``max_chars``.
    """
    import re as _re
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=40, target_words=20)
    # No whitespace in the first 40 chars -- a contiguous run of a long word
    # plus punctuation. The chunker should refuse to split mid-word here.
    chunks = chunker.feed("Massachusetts,Connecticut,Pennsylvania,")
    for chunk in chunks:
        # No chunk should end inside a word character (mid-token slice).
        assert not _re.search(r"\w$", chunk) or chunk.endswith(("Massachusetts", "Connecticut", "Pennsylvania"))
    # Once a space arrives, normal splitting resumes.
    chunks += chunker.feed(" Ohio, Indiana.")
    chunks += chunker.flush()
    joined = " ".join(chunks)
    for state in ("Massachusetts", "Connecticut", "Pennsylvania", "Ohio", "Indiana"):
        assert state in joined, f"missing intact token {state!r} in {chunks!r}"


def test_text_chunker_streaming_deltas_preserve_separator_space():
    """Per-delta comma boundaries must not join into ``Nevada,New Hampshire``.

    Regression: ``_target_word_split`` previously returned ``target_pos``
    (raw word end) when no natural pause was found, leaving the buffer
    starting with a non-space character that joined to the next streamed
    delta inside the same chunk.
    """
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=160, target_words=20)
    chunks: list[str] = []
    for delta in (
        "The state capitals include Montgomery, Juneau, ",
        "Phoenix, Little Rock, Sacramento, Denver, ",
        "Hartford, Dover, Tallahassee, Atlanta.",
    ):
        chunks += chunker.feed(delta)
    chunks += chunker.flush()

    joined = " ".join(chunks)
    # No two words should be joined by a comma with no space between them.
    import re as _re
    assert not _re.search(r"[A-Za-z],[A-Za-z]", joined), (
        f"comma-join bug in chunks: {chunks!r}"
    )


def test_strip_markdown_preserves_separator_space_on_tight_input():
    """Stripping URLs / inline code / bold must not collide adjacent words.

    Regression for live logs: ``stress-testingmy``, ``anddigital``,
    ``toolto``, ``webI`` appeared because the markdown stripper substituted
    matched fragments with empty / capture-only replacements, eating the
    surrounding whitespace when input was tight or stream-spliced.
    """
    from tools.tts_tool import _strip_markdown_for_tts

    # URL stripping must leave a space behind so trailing words don't collide.
    # Previously the URL was replaced with '' and "see the web https://x.com I" became
    # "see the webI" when input lost a space, or any URL strip left no separator.
    assert _strip_markdown_for_tts("see https://example.com now") == "see now"
    # Inline code adjacent to following word -- previously produced ``toolto``.
    assert _strip_markdown_for_tts("use this `tool`to do work") == "use this tool to do work"
    # Bold immediately followed by next word -- previously produced ``anddigital``.
    assert _strip_markdown_for_tts("and**bold**digital") == "and bold digital"
    # Well-formed input must still produce single-spaced output (no regression).
    assert _strip_markdown_for_tts("This is **bold** text") == "This is bold text"
    assert _strip_markdown_for_tts("Run `pip install foo`") == "Run pip install foo"


def test_prepare_voice_response_text_keeps_observed_hear_you_prefix():
    """Regression for live output that sounded like it dropped the opening words."""
    from gateway.voice_pipeline import TextChunker, prepare_voice_response_text

    text = prepare_voice_response_text(
        "I hear you, I promise. I caught every word.",
        max_words=45,
        max_sentences=3,
    )
    chunker = TextChunker(max_chars=80, target_words=20)
    chunks = chunker.feed(text)
    chunks.extend(chunker.flush())

    assert chunks[0].startswith("I hear you")
    assert not chunks[0].startswith((",", ".", ";", ":", "!", "?"))


def test_prepare_voice_response_text_keeps_observed_only_caught_prefix():
    """Regression for live output that sounded like the first phrase was dropped."""
    from gateway.voice_pipeline import TextChunker, prepare_voice_response_text

    text = prepare_voice_response_text(
        "I only caught part of that — you trailed off. Please repeat the last bit.",
        max_words=45,
        max_sentences=3,
    )
    chunker = TextChunker(max_chars=80, target_words=20)
    chunks = chunker.feed(text)
    chunks.extend(chunker.flush())

    assert chunks[0].startswith("I only caught")
    assert not chunks[0].startswith((",", ".", ";", ":", "!", "?"))


@pytest.mark.asyncio
async def test_voice_pipeline_flow():
    """Test the complete voice pipeline flow with feed, synthesis, playback, and cleanup."""
    from gateway.voice_pipeline import DiscordVoicePipeline

    mock_adapter = MagicMock()
    mock_adapter.play_in_voice_channel = AsyncMock(return_value=True)

    mock_result = json.dumps({"success": True, "file_path": "/tmp/fake_audio.mp3"})

    with patch("tools.tts_tool.text_to_speech_tool", return_value=mock_result) as mock_tts, \
         patch("os.path.isfile", return_value=True), \
         patch("os.unlink") as mock_unlink:

         pipeline = DiscordVoicePipeline(
             adapter=mock_adapter,
             guild_id=12345,
             max_chars=50,
             tts_provider_override="orpheus"
         )
         pipeline.start()

         # Feed delta stream
         pipeline.feed("First sentence! ")
         pipeline.feed("Second sentence? ")
         pipeline.flush()

         # Wait for tasks to process
         await pipeline.close()

         # Check synthesis was called with each chunk
         assert mock_tts.call_count == 2
         mock_tts.assert_any_call(text="First sentence!", output_path=ANY)
         mock_tts.assert_any_call(text="Second sentence?", output_path=ANY)

         # Check playback was called sequentially
         assert mock_adapter.play_in_voice_channel.call_count == 2
         mock_adapter.play_in_voice_channel.assert_any_call(12345, ANY)

         # Check environmental override during synthesis
         # Since we run in thread, we can check that tts_provider_override was passed/set
         assert pipeline.tts_provider_override == "orpheus"


@pytest.mark.asyncio
async def test_voice_pipeline_starts_playback_before_final_flush(tmp_path):
    """A complete streamed phrase is synthesized/played before final_response exists."""
    from gateway.voice_pipeline import DiscordVoicePipeline

    mock_adapter = MagicMock()
    mock_adapter.play_in_voice_channel = AsyncMock(return_value=True)

    def fake_tts(*, text, output_path):
        with open(output_path, "wb") as fh:
            fh.write(b"audio")
        return json.dumps({"success": True, "file_path": output_path})

    with patch("tools.tts_tool.text_to_speech_tool", side_effect=fake_tts):
        pipeline = DiscordVoicePipeline(
            adapter=mock_adapter,
            guild_id=12345,
            max_chars=80,
            tts_provider_override="orpheus",
        )
        pipeline.start()
        pipeline.feed("First streamed sentence! ")

        for _ in range(20):
            if mock_adapter.play_in_voice_channel.call_count:
                break
            await asyncio.sleep(0.02)

        assert mock_adapter.play_in_voice_channel.call_count == 1
        pipeline.flush()
        await pipeline.close()


@pytest.mark.asyncio
async def test_pipeline_attempted_state_feed():
    """Test that pipeline.attempted_any becomes True when a chunk is queued via feed()."""
    from gateway.voice_pipeline import DiscordVoicePipeline

    mock_adapter = MagicMock()
    mock_adapter.play_in_voice_channel = AsyncMock(return_value=True)

    pipeline = DiscordVoicePipeline(
        adapter=mock_adapter,
        guild_id=12345,
        max_chars=50,
    )
    assert pipeline.attempted_any is False

    # Feed text that does NOT produce a chunk (no punctuation, within max_chars)
    pipeline.feed("short")
    assert pipeline.attempted_any is False  # buffer not full

    # Feed text that produces a chunk
    pipeline.feed("Hello! ")
    assert pipeline.attempted_any is True

    pipeline.flush()
    await pipeline.close()


@pytest.mark.asyncio
async def test_pipeline_attempted_state_flush():
    """Test that pipeline.attempted_any becomes True when a chunk is produced via flush()."""
    from gateway.voice_pipeline import DiscordVoicePipeline

    mock_adapter = MagicMock()
    mock_adapter.play_in_voice_channel = AsyncMock(return_value=True)

    pipeline = DiscordVoicePipeline(
        adapter=mock_adapter,
        guild_id=12345,
        max_chars=50,
    )
    assert pipeline.attempted_any is False

    # Feed some text without producing a chunk
    pipeline.feed("no punctuation yet")
    assert pipeline.attempted_any is False

    # Flush forces the buffer into a chunk
    pipeline.flush()
    assert pipeline.attempted_any is True

    await pipeline.close()


@pytest.mark.asyncio
async def test_pipeline_active_for_turn():
    """Test active_for_turn reflects started and not closed."""
    from gateway.voice_pipeline import DiscordVoicePipeline

    mock_adapter = MagicMock()
    mock_adapter.play_in_voice_channel = AsyncMock(return_value=True)

    pipeline = DiscordVoicePipeline(
        adapter=mock_adapter,
        guild_id=12345,
    )
    # Not started yet
    assert pipeline.active_for_turn is False

    pipeline.start()
    assert pipeline.active_for_turn is True

    # Still active with text queued
    assert pipeline.active_for_turn is True

    pipeline.flush()
    await pipeline.close()
    assert pipeline.active_for_turn is False


@pytest.mark.asyncio
async def test_pipeline_barge_in_skips_queued_pcm_playback():
    """If the receiver has barge-in set, queued TTS must not continue playing stale audio."""
    from types import SimpleNamespace
    from gateway.voice_pipeline import DiscordVoicePipeline

    mock_adapter = MagicMock()
    mock_adapter._voice_receivers = {12345: SimpleNamespace(_barge_in_triggered=True)}
    mock_adapter.play_pcm_stream_in_voice_channel = AsyncMock(return_value=True)

    pipeline = DiscordVoicePipeline(
        adapter=mock_adapter,
        guild_id=12345,
        tts_provider_override="orpheus",
    )
    pipeline.start()
    pipeline.feed_chunks(["This should not keep playing after barge-in."])
    pipeline.flush()
    await pipeline.close()

    assert pipeline.attempted_any is True
    assert mock_adapter.play_pcm_stream_in_voice_channel.call_count == 0


class TestShouldSendVoiceReply:
    """Tests for _should_send_voice_reply logic in gateway/run.py."""

    @pytest.fixture
    def mock_event(self):
        """Create a minimal voice-mode MessageEvent fixture."""
        from unittest.mock import MagicMock
        from gateway.run import MessageEvent, MessageType
        from gateway.config import Platform

        event = MagicMock(spec=MessageEvent)
        event.message_type = MessageType.TEXT
        event.source.platform = Platform.DISCORD
        event.source.chat_id = "test_chat"
        return event

    def _make_runner(self, voice_mode="all"):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._voice_mode = {"discord:test_chat": voice_mode}
        return runner

    def test_returns_false_when_pipeline_tts_attempted(self, mock_event):
        """_should_send_voice_reply returns False when voice_pipeline_tts_attempted=True."""
        runner = self._make_runner()

        result = runner._should_send_voice_reply(
            mock_event,
            response="Hello world",
            agent_messages=[],
            voice_pipeline_tts_attempted=True,
            voice_pipeline_audio_sent=False,
        )
        assert result is False

    def test_returns_false_when_pipeline_audio_sent(self, mock_event):
        """_should_send_voice_reply returns False when voice_pipeline_audio_sent=True."""
        runner = self._make_runner()

        result = runner._should_send_voice_reply(
            mock_event,
            response="Hello world",
            agent_messages=[],
            voice_pipeline_audio_sent=True,
            voice_pipeline_tts_attempted=False,
        )
        assert result is False

    def test_returns_true_when_no_pipeline_flag(self, mock_event):
        """_should_send_voice_reply returns True when neither pipeline flag is set."""
        runner = self._make_runner()

        result = runner._should_send_voice_reply(
            mock_event,
            response="Hello world",
            agent_messages=[],
        )
        assert result is True

    def test_returns_false_for_empty_response(self, mock_event):
        """_should_send_voice_reply returns False for empty response."""
        runner = self._make_runner("off")

        result = runner._should_send_voice_reply(
            mock_event,
            response="",
            agent_messages=[],
        )
        assert result is False

    def test_returns_false_for_error_response(self, mock_event):
        """_should_send_voice_reply returns False for Error: prefixed response."""
        runner = self._make_runner("all")

        result = runner._should_send_voice_reply(
            mock_event,
            response="Error: something broke",
            agent_messages=[],
        )
        assert result is False

    def test_discord_voice_pipeline_final_only_env(self, monkeypatch):
        """Final-only Orpheus mode is enabled by explicit env flag, not by default."""
        runner = self._make_runner("all")

        monkeypatch.delenv("HERMES_DISCORD_VOICE_PIPELINE_FINAL_ONLY", raising=False)
        assert runner._discord_voice_pipeline_final_only() is False

        monkeypatch.setenv("HERMES_DISCORD_VOICE_PIPELINE_FINAL_ONLY", "true")
        assert runner._discord_voice_pipeline_final_only() is True


def test_text_chunker_default_numeric_comma_list_chunks_at_three_items():
    """Default numeric comma lists split after three spoken items."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=100, target_words=10)
    assert chunker.numeric_list_target == 3
    chunks = chunker.feed("One, two, three, four, five, six,")

    assert chunks == ["One, two, three,", "four, five, six,"]


def test_text_chunker_numeric_list_target_five_chunks_at_five_items():
    """Configurable numeric comma lists can split after five spoken items."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=100, target_words=10, numeric_list_target=5)
    assert chunker.numeric_list_target == 5
    chunks = chunker.feed("One, two, three, four, five, six, seven, eight, nine, ten,")

    assert chunks == ["One, two, three, four, five,", "six, seven, eight, nine, ten,"]


def test_text_chunker_numeric_list_target_zero_clamps_to_one_item_chunks():
    """Low numeric_list_target values clamp to one spoken item per chunk."""
    from gateway.voice_pipeline import TextChunker

    chunker = TextChunker(max_chars=100, target_words=10, numeric_list_target=0)
    assert chunker.numeric_list_target == 1
    chunks = chunker.feed("One, two, three,")

    assert chunks == ["One,", "two,", "three,"]


def test_env_helper_discord_streaming_tts_numeric_list_target(monkeypatch):
    """Test env helper for HERMES_DISCORD_VOICE_STREAMING_NUMERIC_LIST_TARGET with fallback and clamping."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)

    monkeypatch.delenv("HERMES_DISCORD_VOICE_STREAMING_NUMERIC_LIST_TARGET", raising=False)
    assert runner._discord_streaming_tts_numeric_list_target() == 3

    monkeypatch.setenv("HERMES_DISCORD_VOICE_STREAMING_NUMERIC_LIST_TARGET", "5")
    assert runner._discord_streaming_tts_numeric_list_target() == 5

    monkeypatch.setenv("HERMES_DISCORD_VOICE_STREAMING_NUMERIC_LIST_TARGET", "invalid")
    assert runner._discord_streaming_tts_numeric_list_target() == 3

    monkeypatch.setenv("HERMES_DISCORD_VOICE_STREAMING_NUMERIC_LIST_TARGET", "0")
    assert runner._discord_streaming_tts_numeric_list_target() == 1


def test_env_helper_discord_streaming_tts_min_chunk_words(monkeypatch):
    """Test env helper for HERMES_DISCORD_VOICE_STREAMING_MIN_CHUNK_WORDS with fallback and clamping."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)

    monkeypatch.delenv("HERMES_DISCORD_VOICE_STREAMING_MIN_CHUNK_WORDS", raising=False)
    assert runner._discord_streaming_tts_min_chunk_words() == 0

    monkeypatch.setenv("HERMES_DISCORD_VOICE_STREAMING_MIN_CHUNK_WORDS", "12")
    assert runner._discord_streaming_tts_min_chunk_words() == 12

    monkeypatch.setenv("HERMES_DISCORD_VOICE_STREAMING_MIN_CHUNK_WORDS", "invalid")
    assert runner._discord_streaming_tts_min_chunk_words() == 0

    monkeypatch.setenv("HERMES_DISCORD_VOICE_STREAMING_MIN_CHUNK_WORDS", "100")
    assert runner._discord_streaming_tts_min_chunk_words() == 80


def test_numeric_pcm_stream_default_cap_stops_single_number_loops(monkeypatch, caplog):
    """Single count chunks should not play multi-second Orpheus numeric loops."""
    import logging

    from gateway.voice_pipeline import DiscordVoicePipeline

    monkeypatch.delenv("HERMES_ORPHEUS_NUMERIC_PCM_MAX_BYTES", raising=False)
    pipeline = DiscordVoicePipeline.__new__(DiscordVoicePipeline)
    caplog.set_level(logging.WARNING)

    chunks = [b"a" * 65536, b"b" * 65536, b"c" * 65536]
    limited = list(pipeline._limit_numeric_pcm_stream(chunks, 94, "ninety four,"))

    assert sum(len(chunk) for chunk in limited) == 120000
    assert any("Truncated Orpheus numeric PCM chunk 94 at 120000 bytes" in rec.message for rec in caplog.records)


def test_numeric_pcm_stream_env_override_still_supported(monkeypatch):
    from gateway.voice_pipeline import DiscordVoicePipeline

    monkeypatch.setenv("HERMES_ORPHEUS_NUMERIC_PCM_MAX_BYTES", "200000")
    pipeline = DiscordVoicePipeline.__new__(DiscordVoicePipeline)

    chunks = [b"a" * 65536, b"b" * 65536, b"c" * 65536, b"d" * 65536]
    limited = list(pipeline._limit_numeric_pcm_stream(chunks, 94, "ninety four,"))

    assert sum(len(chunk) for chunk in limited) == 200000


# ==============================================================================
# H4: structured debug log behind HERMES_DISCORD_VOICE_DEBUG=1
# ==============================================================================


@pytest.mark.asyncio
async def test_voice_pipeline_debug_env_unset(caplog):
    """When HERMES_DISCORD_VOICE_DEBUG is unset, no voice_tts_turn record is emitted."""
    import logging
    caplog.set_level(logging.INFO)
    from gateway.voice_pipeline import DiscordVoicePipeline

    mock_adapter = MagicMock()
    mock_adapter.play_in_voice_channel = AsyncMock(return_value=True)

    mock_result = json.dumps({"success": True, "file_path": "/tmp/fake_audio.mp3"})

    with patch("tools.tts_tool.text_to_speech_tool", return_value=mock_result), \
         patch("os.path.isfile", return_value=True), \
         patch("os.unlink"):
        pipeline = DiscordVoicePipeline(
            adapter=mock_adapter,
            guild_id=12345,
            max_chars=50,
        )
        pipeline.start()
        pipeline.feed("Hello world! ")
        pipeline.flush()
        await pipeline.close()

    voice_tts_turn_records = [r for r in caplog.records if "voice_tts_turn" in r.getMessage()]
    assert len(voice_tts_turn_records) == 0


@pytest.mark.asyncio
async def test_voice_pipeline_debug_env_set(monkeypatch, caplog):
    """With HERMES_DISCORD_VOICE_DEBUG=1, close() emits a valid JSON record with required keys."""
    import logging
    caplog.set_level(logging.INFO)
    from gateway.voice_pipeline import DiscordVoicePipeline

    monkeypatch.setenv("HERMES_DISCORD_VOICE_DEBUG", "1")

    mock_adapter = MagicMock()
    mock_adapter.play_in_voice_channel = AsyncMock(return_value=True)

    mock_result = json.dumps({"success": True, "file_path": "/tmp/fake_audio.mp3"})

    with patch("tools.tts_tool.text_to_speech_tool", return_value=mock_result), \
         patch("os.path.isfile", return_value=True), \
         patch("os.unlink"):
        pipeline = DiscordVoicePipeline(
            adapter=mock_adapter,
            guild_id=12345,
            max_chars=50,
            target_words=20,
        )
        pipeline.start()

        # Feed chunks that produce multiple queued items
        pipeline.feed("First chunk!")
        pipeline.feed("Second test chunk. ")
        pipeline.feed_chunks(["Third queued chunk."])
        pipeline.flush()

        await pipeline.close()

    # Collect voice_tts_turn log records
    records = [
        r for r in caplog.records
        if "voice_tts_turn" in r.getMessage()
    ]
    assert len(records) == 1, f"Expected exactly 1 voice_tts_turn record, got {len(records)}"

    msg = records[0].getMessage()
    assert msg.startswith("voice_tts_turn=")

    record = json.loads(msg[len("voice_tts_turn="):])

    # Verify all required keys
    required_keys = {
        "turn_id", "guild_id", "provider", "endpoint", "playback_mode",
        "target_words", "numeric_list_target", "min_chunk_words", "max_chars",
        "attempted_any", "delivered_any", "chunk_count", "chunks",
        "pipeline_flags", "elapsed_ms",
    }
    assert required_keys.issubset(record.keys()), f"Missing keys: {required_keys - record.keys()}"

    # chunk_count must match chunks length
    assert record["chunk_count"] == len(record["chunks"])
    assert record["chunk_count"] > 0

    # Verify chunk structure
    for chunk in record["chunks"]:
        assert "idx" in chunk
        assert "text" in chunk
        assert "words" in chunk
        assert "chars" in chunk

    # pipeline_flags has required sub-keys
    assert "final_only" in record["pipeline_flags"]
    assert "streaming_tts_enabled" in record["pipeline_flags"]

    # Type checks
    assert record["guild_id"] == 12345
    assert record["playback_mode"] in ("pcm", "file")
    assert isinstance(record["elapsed_ms"], (int, float))
