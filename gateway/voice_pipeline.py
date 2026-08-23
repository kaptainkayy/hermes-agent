"""Live voice phrase chunking and low-latency pipeline helpers for streaming Discord VC TTS."""

import asyncio
import logging
import os
import tempfile
import json
import wave
from typing import Any, Callable, Iterable, Optional, List
import re

logger = logging.getLogger(__name__)


_ORDINAL_WORDS = {
    "1": "First",
    "2": "Second",
    "3": "Third",
    "4": "Fourth",
    "5": "Fifth",
    "6": "Sixth",
    "7": "Seventh",
    "8": "Eighth",
    "9": "Ninth",
    "10": "Tenth",
}

_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand",
}


_VOICE_CONTROL_MESSAGE_RE = re.compile(
    r"^\s*operation\s+interrupted(?:"
    r"\s*:\s*(?:waiting\s+for\s+model\s+response|handling\s+api\s+error|retrying\s+api\s+call\s+after\s+error)\b.*"
    r"|\s+during\s+retry\s*\(.*\)"
    r")\s*$",
    re.IGNORECASE | re.DOTALL,
)


def is_voice_control_message(text: str) -> bool:
    """Return True for internal gateway/control messages that TTS must not speak."""
    body = " ".join(str(text or "").strip().split())
    if not body:
        return False
    return bool(_VOICE_CONTROL_MESSAGE_RE.match(body))


def normalize_voice_text(text: str) -> str:
    """Return text cleaned for spoken TTS while preserving user-facing warmth/content."""
    try:
        from tools.tts_tool import _strip_markdown_for_tts
        text = _strip_markdown_for_tts(text)
    except Exception:
        text = str(text or "")

    # Streaming chunks can split balanced markdown tokens across chunk
    # boundaries, leaving fragments like ``**1.`` or ``Heading**`` for TTS.
    text = re.sub(r"(?<!\w)(?:\*\*|__)(?=\w|\d)", "", text)
    text = re.sub(r"(?<=\w|\d)(?:\*\*|__)(?!\w)", "", text)
    text = re.sub(r"(?<!\w)[`*_]+(?=\d{1,2}[.)])", "", text)

    if is_voice_control_message(text):
        logger.info("Suppressing internal control message from voice TTS: %r", str(text)[:160])
        return ""

    # Never speak transcript/session labels leaked by continuation handling.
    # In live Discord voice a length-limited turn can resume with artifacts such
    # as ``Assistant (me):``; Orpheus then literally says the label, which sounds
    # like a cutoff/chunking bug. Strip the label while keeping surrounding text.
    text = re.sub(r"(?i)assistant\s*\([^)]*\)\s*:\s*", " ", text)
    text = re.sub(r"(?im)^\s*(assistant|user|system)\s*:\s*", "", text)

    text = re.sub(
        r"(?m)^\s*(\d{1,2})[.)]\s+",
        lambda match: f"{_ORDINAL_WORDS.get(match.group(1), match.group(1))}, ",
        text,
    )
    text = text.replace("&", "and")
    text = re.sub(r",(?=\S)", ", ", text)
    text = re.sub(r"\b(\d{1,3})\s*°\s*F\b", lambda m: f"{_number_to_words(int(m.group(1)))} degrees Fahrenheit", text)
    text = _dehyphenate_number_words(text)
    text = re.sub(r"\b([A-Za-z]+)_([A-Za-z]+)\b", r"\1 \2", text)
    text = re.sub(r"\b(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth), ([^:]{1,60}):", lambda m: f"{m.group(1)}, {m.group(2).lower()}:", text)
    text = _repair_voice_text_corruption(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_voice_response_text(text: str, *, max_words: Optional[int] = None, max_sentences: Optional[int] = None) -> str:
    """Return a bounded, speech-first version of a final response for live TTS.

    This is intentionally a playback guardrail, not a replacement for prompting:
    the model should still answer briefly in voice mode, but TTS should not read
    a long wall of text when the model drifts.  Trimming happens only at sentence
    boundaries when possible so the spoken answer remains professional.
    """
    text = normalize_voice_text(text)
    if not text:
        return ""

    if max_words is None:
        raw_words = os.getenv("HERMES_DISCORD_VOICE_TTS_MAX_WORDS", "45")
        try:
            max_words = int(raw_words)
        except (TypeError, ValueError):
            max_words = 45
    if max_sentences is None:
        raw_sentences = os.getenv("HERMES_DISCORD_VOICE_TTS_MAX_SENTENCES", "3")
        try:
            max_sentences = int(raw_sentences)
        except (TypeError, ValueError):
            max_sentences = 3

    if max_words <= 0 and max_sentences <= 0:
        return text

    sentences = [part.strip() for part in re.findall(r"[^.!?]+[.!?]|[^.!?]+$", text) if part.strip()]
    if not sentences:
        sentences = [text]

    selected: List[str] = []
    word_count = 0
    for sentence in sentences:
        sentence_words = TextChunker._word_spans(sentence) if "TextChunker" in globals() else list(re.finditer(r"\b[\w'-]+\b", sentence))
        next_count = word_count + len(sentence_words)
        if selected and max_sentences > 0 and len(selected) >= max_sentences:
            break
        if selected and max_words > 0 and next_count > max_words:
            break
        if not selected and max_words > 0 and len(sentence_words) > max_words:
            words = sentence_words[:max_words]
            trimmed = sentence[: words[-1].end()].rstrip(" ,;:") if words else sentence
            selected.append(trimmed + ("." if not re.search(r"[.!?]$", trimmed) else ""))
            break
        selected.append(sentence)
        word_count = next_count

    return " ".join(selected).strip() or text


def _repair_voice_text_corruption(text: str) -> str:
    """Repair small stream-splice artifacts before TTS speaks them."""
    text = str(text or "")
    # Observed in live logs as "waitor just make you", caused by a missing
    # boundary space between "wait" and "or". Keep this conservative.
    text = re.sub(r"\b(wait)or\b", r"\1 or", text, flags=re.IGNORECASE)
    return _collapse_adjacent_repeated_word_runs(text)


def _clean_spoken_chunk_edges(chunk: str) -> str:
    """Remove punctuation-only leading fragments before a TTS chunk is spoken."""
    cleaned = re.sub(r"^[\s,.;:!?—–-]+", "", str(chunk or "")).strip()
    if cleaned != chunk:
        logger.warning("Cleaned leading punctuation from voice TTS chunk: before=%r after=%r", chunk, cleaned)
    return cleaned


def _collapse_adjacent_repeated_word_runs(text: str) -> str:
    """Collapse adjacent repeated 2-8 word runs while preserving simple text."""
    parts = text.split()
    if len(parts) < 4:
        return text

    def key(token: str) -> str:
        return re.sub(r"^\W+|\W+$", "", token).lower()

    i = 0
    while i < len(parts):
        collapsed = False
        max_run = min(8, (len(parts) - i) // 2)
        for run in range(max_run, 1, -1):
            left = [key(token) for token in parts[i : i + run]]
            right = [key(token) for token in parts[i + run : i + (2 * run)]]
            if left and left == right and all(left):
                del parts[i + run : i + (2 * run)]
                collapsed = True
                break
        if not collapsed:
            i += 1
    return " ".join(parts)


def _dehyphenate_number_words(text: str) -> str:
    """Avoid Orpheus drift on hyphenated spoken numbers like forty-six."""
    def replace(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        if left.lower() in _NUMBER_WORDS and right.lower() in _NUMBER_WORDS:
            return f"{left} {right}"
        return match.group(0)

    return re.sub(r"\b([A-Za-z]+)-([A-Za-z]+)\b", replace, text)


def _number_to_words(value: int) -> str:
    small = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen",
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    if 0 <= value < 20:
        return small[value]
    if value < 100:
        ten, one = divmod(value, 10)
        return tens[ten] if one == 0 else f"{tens[ten]}-{small[one]}"
    return str(value)


def _verbalize_digit_count_chunk(text: str) -> str:
    """Convert digit-only count chunks to words before Orpheus synthesis."""
    words = TextChunker._word_spans(text) if "TextChunker" in globals() else list(re.finditer(r"\b\d{1,3}\b", text))
    if not words:
        return text
    values: List[int] = []
    for word in words:
        token = word.group(0)
        if not re.fullmatch(r"\d{1,3}", token):
            return text
        value = int(token)
        if not 1 <= value <= 100:
            return text
        values.append(value)

    if len(values) != len(words):
        return text
    return re.sub(r"\b\d{1,3}\b", lambda match: _number_to_words(int(match.group(0))), text)


class TextChunker:
    """Split incremental streaming text deltas into natural phrase chunks.

    Buffers text until punctuation indicating a pause is found or until max_chars
    is exceeded.
    """

    def __init__(self, max_chars: int = 100, target_words: int = 0, numeric_list_target: int = 3, min_chunk_words: int = 0):
        self.max_chars = max_chars
        self.target_words = max(0, target_words)
        self.numeric_list_target = max(1, numeric_list_target)
        self.min_chunk_words = max(0, min_chunk_words)
        self.buffer = ""
        self._pending_prefix = ""

    @staticmethod
    def _word_spans(text: str) -> List[re.Match[str]]:
        return list(re.finditer(r"\b\d+(?:\.\d+)+\b|\b[\w'-]+\b", text))

    @staticmethod
    def _is_number_word(word: str) -> bool:
        if re.fullmatch(r"\d{1,3}", word):
            return True
        parts = [part for part in re.split(r"[-']+", word.lower()) if part]
        return bool(parts) and all(part in _NUMBER_WORDS for part in parts)

    @classmethod
    def _is_number_phrase(cls, text: str) -> bool:
        words = cls._word_spans(text)
        return bool(words) and all(cls._is_number_word(word.group(0)) for word in words)

    def _numeric_list_target(self, words: List[re.Match[str]]) -> Optional[int]:
        """Use tiny chunks for comma-separated number recitations.

        Orpheus can over-generate or drift on repetitive counting lists. Keeping
        number-list chunks to roughly five spoken numbers makes each synthesis
        request short enough to finish deterministically without creating a
        robotic one-number-per-chunk queue.
        """
        target_items = self.numeric_list_target
        if len(words) < target_items:
            return None
        sample = words[: min(len(words), 10)]
        numeric = sum(1 for word in sample if self._is_number_word(word.group(0)))
        if numeric < target_items or numeric / len(sample) < 0.8:
            return None
        if self.buffer.count(",") < target_items - 1:
            if not all(re.fullmatch(r"\d{1,3}", word.group(0)) for word in sample):
                return None
        return target_items

    def _numeric_list_split(self, target_items: int) -> Optional[int]:
        """Return a split point after target_items comma-separated number phrases.

        Counting lists contain multi-word numbers such as ``forty six`` and
        ``one hundred``. Splitting by raw word spans can cut inside those
        numbers and leave Orpheus with orphan chunks like ``hundred.``. Treat
        each comma-delimited number phrase as one spoken item instead.
        """
        numeric_items = 0
        for match in re.finditer(r"[^,]+(?:,|$)", self.buffer):
            raw_item = match.group(0).strip()
            if not raw_item:
                continue
            complete = raw_item.endswith(",") or bool(re.search(r"[.!?;:]$", raw_item))
            phrase = raw_item.rstrip(",.!?;:").strip()
            if not self._is_number_phrase(phrase):
                return None
            numeric_items += 1
            if numeric_items == target_items:
                return match.end() if complete else None
        return None

    def _numeric_word_split(self, target_items: int) -> Optional[int]:
        """Return a split point for whitespace-separated digit count output.

        If STT/provider routing misses the deterministic count path, some
        models answer with ``1 2 3 ...``. Keep those chunks bounded too, instead
        of letting the final sentence chunk contain dozens of digits.
        """
        words = self._word_spans(self.buffer)
        sample = words[: min(len(words), 10)]
        if sample and all(re.fullmatch(r"\d{1,3}", word.group(0)) for word in sample):
            # Fallback LLM digit streams like ``1 2 3 4 ...`` are harder for
            # Orpheus than comma-delimited deterministic count chunks. Keep
            # them extra short after verbalization so chunks such as
            # ``thirteen fourteen fifteen`` do not exceed the live UAT budget
            # or trigger count-loop synthesis.
            target_items = min(target_items, 2)
        if len(words) < target_items:
            return None
        if not sample or not all(self._is_number_word(word.group(0)) for word in sample):
            return None
        end = words[target_items - 1].end()
        match = re.match(r"\s*[,\.\!\?;:]?\s*", self.buffer[end:])
        return end + (match.end() if match else 0)

    def _short_comma_list_split(self, target_items: int = 6) -> Optional[int]:
        """Return a split point for long comma-separated name/list recitations."""
        if self.buffer.count(",") < target_items - 1:
            return None

        list_items = 0
        for match in re.finditer(r"[^,]+(?:,|$)", self.buffer):
            raw_item = match.group(0).strip()
            if not raw_item:
                continue
            complete = raw_item.endswith(",") or bool(re.search(r"[.!?;:]$", raw_item))
            phrase = raw_item.rstrip(",.!?;:").strip()
            if not phrase or len(self._word_spans(phrase)) > 4 or re.search(r"[.!?;:]", phrase):
                return None
            list_items += 1
            if list_items == target_items:
                return match.end() if complete else None
        return None

    def _target_word_split(self) -> Optional[int]:
        """Return a natural split point once the buffer reaches target_words.

        The low-latency Discord voice path gets text incrementally. If we only
        split by characters, the pipeline may cut mid-sentence into short pieces
        like "session to pull the latest headlines for you," which resets
        Orpheus prosody and makes a few words sound different. A word target
        keeps chunks in the 4-6s spoken range while preferring comma/sentence
        boundaries when they are available.
        """
        words = self._word_spans(self.buffer)
        numeric_target = self._numeric_list_target(words)
        if numeric_target is not None:
            numeric_split = self._numeric_list_split(numeric_target)
            if numeric_split is not None:
                return numeric_split
            numeric_split = self._numeric_word_split(numeric_target)
            if numeric_split is not None:
                return numeric_split

        comma_list_split = self._short_comma_list_split()
        if comma_list_split is not None:
            return comma_list_split

        target_words = self.target_words
        if target_words <= 0:
            return None
        if len(self.buffer) < self.max_chars:
            return None
        if len(words) < target_words:
            return None

        if target_words <= 5:
            end = words[target_words - 1].end()
            match = re.match(r"\s*,\s*", self.buffer[end:])
            return end + (match.end() if match else 0)

        min_word_index = max(1, target_words // 2)
        min_pos = words[min_word_index - 1].end() if len(words) >= min_word_index else 0
        target_pos = words[min(target_words, len(words)) - 1].end()
        search_window = self.buffer[: max(target_pos, min(len(self.buffer), self.max_chars))]
        natural_matches = list(re.finditer(r"([!?;:,]|\.(?!\d))(?:\s|$)", search_window))
        for match in reversed(natural_matches):
            if match.end() >= min_pos:
                return match.end()

        # No natural pause found within the window. Returning ``target_pos``
        # raw can leave the buffer remainder starting with a non-space
        # character (e.g. "Nevada,") which then joins to the next streamed
        # delta as "Nevada,New Hampshire". Always advance past a trailing
        # separator + whitespace before yielding, and refuse to split if
        # ``target_pos`` lands inside a word with no nearby whitespace --
        # waiting one more delta is preferable to a mid-word cut.
        sep_match = re.match(r"[,;:]?\s+", self.buffer[target_pos:])
        if sep_match:
            return target_pos + sep_match.end()
        if target_pos >= len(self.buffer):
            return None  # word end at buffer end -- wait for next delta
        return None

    def _split_exempt_from_min_words(self, split_idx: int) -> bool:
        """Return whether a split has its own chunk-sizing rule."""
        words = self._word_spans(self.buffer)
        numeric_target = self._numeric_list_target(words)
        if numeric_target is not None:
            for candidate in (
                self._numeric_list_split(numeric_target),
                self._numeric_word_split(numeric_target),
            ):
                if candidate == split_idx:
                    return True
        return self._short_comma_list_split() == split_idx

    def _below_min_chunk_words(self, chunk: str) -> bool:
        """Return whether a mid-stream chunk should wait for more words."""
        return self.min_chunk_words > 0 and len(self._word_spans(chunk)) < self.min_chunk_words

    @staticmethod
    def _ends_with_partial_word(chunk: str) -> bool:
        """Return whether a non-final chunk likely ends inside a streamed word."""
        if re.search(r"[.!?:;,]$", chunk):
            return False
        words = TextChunker._word_spans(chunk)
        if not words:
            return False
        last = words[-1].group(0)
        return bool(re.fullmatch(r"[a-z]", last))

    def _queueable_chunk(self, chunk: str, *, final: bool = False) -> Optional[str]:
        """Return a spoken chunk or hold tiny structure fragments for the next chunk."""
        chunk = normalize_voice_text(chunk)
        if not chunk:
            return None

        marker = re.fullmatch(r"(\d{1,2})[.)]", chunk)
        if marker:
            self._pending_prefix = f"{_ORDINAL_WORDS.get(marker.group(1), marker.group(1))},"
            return None

        words = self._word_spans(chunk)
        if not words and re.fullmatch(r"[.!?:;,\s]+", chunk):
            return None
        if chunk.endswith(":") and len(words) <= 4 and not final:
            prefix = f"{self._pending_prefix} " if self._pending_prefix else ""
            self._pending_prefix = f"{prefix}{chunk}".strip()
            return None

        if self._pending_prefix:
            if re.fullmatch(r"[\W_]+", self._pending_prefix):
                logger.warning("Dropping punctuation-only pending voice prefix: %r", self._pending_prefix)
                self._pending_prefix = ""
                return self._queueable_chunk(chunk, final=final)
            if self._pending_prefix.endswith(",") and chunk:
                chunk = chunk[:1].lower() + chunk[1:]
            chunk = f"{self._pending_prefix} {chunk}".strip()
            self._pending_prefix = ""

        if final and len(words) <= 2 and not re.search(r"[.!?:;,]$", chunk):
            chunk = f"{chunk}."

        chunk = _verbalize_digit_count_chunk(chunk)
        return _clean_spoken_chunk_edges(normalize_voice_text(chunk))

    def feed(self, text: str) -> List[str]:
        """Feed a new delta segment of text and return any ready chunks."""
        if self._ends_with_partial_word(self.buffer.rstrip()) and str(text or "")[:1].islower():
            self.buffer = self.buffer.rstrip()
        self.buffer += text
        if is_voice_control_message(self.buffer):
            logger.info("Dropping buffered internal control message from voice TTS: %r", self.buffer[:160])
            self.buffer = ""
            return []
        chunks: List[str] = []

        while True:
            # Look for natural sentence boundaries: . ! ? ; :
            matches = list(re.finditer(r'([!?;:]|\.(?!\d))(?:\s|$)', self.buffer))
            match = matches[0] if matches else None
            if self.min_chunk_words > 0:
                for candidate in matches:
                    candidate_text = self.buffer[: candidate.end()].strip()
                    if not self._below_min_chunk_words(candidate_text):
                        match = candidate
                        break
                else:
                    match = None
            target_split = self._target_word_split()
            if target_split is not None and (match is None or target_split < match.end()):
                chunk = self.buffer[:target_split].strip()
                if self._ends_with_partial_word(chunk):
                    break
                if self._below_min_chunk_words(chunk) and not self._split_exempt_from_min_words(target_split):
                    break
                self.buffer = self.buffer[target_split:]
                chunk = self._queueable_chunk(chunk)
                if chunk:
                    chunks.append(chunk)
            elif match:
                end_idx = match.end()
                chunk = self.buffer[:end_idx].strip()
                if self._below_min_chunk_words(chunk):
                    break
                self.buffer = self.buffer[end_idx:]
                chunk = self._queueable_chunk(chunk)
                if chunk:
                    chunks.append(chunk)
            elif len(self.buffer) >= self.max_chars:
                # Exceeded max chars, split at last whitespace in window.
                # If no whitespace exists inside ``max_chars`` we previously
                # clamped at ``max_chars`` -- but that slices mid-word for
                # long enumerations ("...Kentucky, Lou|isiana...") and ships
                # half-tokens to Orpheus. Prefer to wait for whitespace to
                # arrive in the next delta; only force-split when the buffer
                # grows past a hard 3x ceiling on pathological no-whitespace
                # input (e.g. a giant URL).
                split_idx = self.buffer.rfind(" ", 0, self.max_chars)
                if split_idx <= 0:
                    if len(self.buffer) < self.max_chars * 3:
                        break
                    far_split = self.buffer.find(" ", self.max_chars)
                    split_idx = far_split if far_split > 0 else len(self.buffer)
                chunk = self.buffer[:split_idx].strip()
                if self._ends_with_partial_word(chunk):
                    break
                if self._below_min_chunk_words(chunk):
                    break
                self.buffer = self.buffer[split_idx:]
                chunk = self._queueable_chunk(chunk)
                if chunk:
                    chunks.append(chunk)
            else:
                break
        return chunks

    def flush(self) -> List[str]:
        """Flush the remaining buffer contents as a single chunk."""
        chunk = self.buffer.strip()
        self.buffer = ""
        chunk = self._queueable_chunk(chunk, final=True) if chunk else None
        if chunk:
            return [chunk]
        if self._pending_prefix:
            chunk = self._pending_prefix
            self._pending_prefix = ""
            if not re.search(r"[.!?:;]$", chunk):
                chunk = f"{chunk}."
            chunk = _verbalize_digit_count_chunk(chunk)
            chunk = re.sub(r",\s*([.!?])", r"\1", chunk)
            return [_clean_spoken_chunk_edges(normalize_voice_text(chunk))]
        return []


class DiscordVoicePipeline:
    """Pipelines streaming text deltas from LLM callback to low-latency sequential TTS playback.

    Synthesis of later chunks runs in background threads while the current chunk
    is playing.
    """

    _turn_counter = 0

    def __init__(
        self,
        adapter: Any,
        guild_id: int,
        max_chars: int = 100,
        target_words: int = 0,
        tts_provider_override: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        numeric_list_target: int = 3,
        min_chunk_words: int = 0,
    ):
        self.adapter = adapter
        self.guild_id = guild_id
        self.max_chars = max_chars
        self.target_words = max(0, target_words)
        self.numeric_list_target = max(1, numeric_list_target)
        self.min_chunk_words = max(0, min_chunk_words)
        self.tts_provider_override = tts_provider_override

        self._chunker = TextChunker(
            max_chars=max_chars,
            target_words=target_words,
            numeric_list_target=numeric_list_target,
            min_chunk_words=min_chunk_words,
        )
        self._loop = loop or asyncio.get_running_loop()

        self._text_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[Optional[Any]] = asyncio.Queue()

        self._synthesis_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._started = False
        self._closed = False
        self._generated_paths: List[str] = []
        self._debug_paths: List[str] = []
        self._delivered_any = False
        self._attempted_any = False
        self._started_at: Optional[float] = None
        self._queued_chunks: List[dict] = []
        type(self)._turn_counter += 1
        self._turn_id: str = f"{id(self)}-{type(self)._turn_counter}"

    def start(self):
        """Start the background synthesis and playback worker tasks."""
        if self._started:
            return
        self._started = True
        self._started_at = self._loop.time()

        def _start_tasks():
            self._synthesis_task = self._loop.create_task(self._synthesis_worker())
            self._playback_task = self._loop.create_task(self._playback_worker())

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._loop:
            _start_tasks()
        else:
            self._loop.call_soon_threadsafe(_start_tasks)

    def feed(self, text: str):
        """Feed an incremental text delta segment from LLM response (thread-safe)."""
        if self._closed:
            return
        chunks = self._chunker.feed(text)
        if chunks:
            self._attempted_any = True
        for chunk in chunks:
            words = len(TextChunker._word_spans(chunk))
            self._queued_chunks.append({
                "idx": len(self._queued_chunks) + 1,
                "text": chunk,
                "words": words,
                "chars": len(chunk),
            })
            self._loop.call_soon_threadsafe(self._text_queue.put_nowait, chunk)

    def feed_chunks(self, chunks: Iterable[str]):
        """Queue already chunked spoken text directly, bypassing TextChunker."""
        if self._closed:
            return
        queued = False
        for raw_chunk in chunks:
            chunk = normalize_voice_text(str(raw_chunk or "").strip())
            chunk = _verbalize_digit_count_chunk(chunk)
            chunk = _clean_spoken_chunk_edges(chunk)
            if not chunk:
                continue
            queued = True
            words = len(TextChunker._word_spans(chunk))
            self._queued_chunks.append({
                "idx": len(self._queued_chunks) + 1,
                "text": chunk,
                "words": words,
                "chars": len(chunk),
            })
            self._loop.call_soon_threadsafe(self._text_queue.put_nowait, chunk)
        if queued:
            self._attempted_any = True

    def flush(self):
        """Signal completion and flush any remaining tail to playback (thread-safe)."""
        if self._closed:
            return
        chunks = self._chunker.flush()
        if chunks:
            self._attempted_any = True
        for chunk in chunks:
            words = len(TextChunker._word_spans(chunk))
            self._queued_chunks.append({
                "idx": len(self._queued_chunks) + 1,
                "text": chunk,
                "words": words,
                "chars": len(chunk),
            })
            self._loop.call_soon_threadsafe(self._text_queue.put_nowait, chunk)
        self._loop.call_soon_threadsafe(self._text_queue.put_nowait, None)

    @property
    def delivered_any(self) -> bool:
        """Return True if any audio chunk has been successfully generated and queued/played."""
        return self._delivered_any

    @property
    def attempted_any(self) -> bool:
        """Return True if any text chunk was queued for TTS synthesis this turn."""
        return self._attempted_any

    @property
    def active_for_turn(self) -> bool:
        """Return True if the pipeline was started and has not been closed (active for this turn)."""
        return self._started and not self._closed

    def _should_use_orpheus_pcm_streaming(self) -> bool:
        """Return True when the pipeline should bypass file TTS for Orpheus PCM."""
        has_explicit_pcm_method = (
            "play_pcm_stream_in_voice_channel" in getattr(self.adapter, "__dict__", {})
            or hasattr(type(self.adapter), "play_pcm_stream_in_voice_channel")
        )
        if not has_explicit_pcm_method:
            return False
        provider = (self.tts_provider_override or "").strip().lower()
        if not provider:
            try:
                from tools.tts_tool import _get_provider, _load_tts_config
                provider = _get_provider(_load_tts_config())
            except Exception:
                provider = ""
        return provider == "orpheus"

    async def _synthesis_worker(self):
        from tools.tts_tool import text_to_speech_tool

        use_pcm_streaming = self._should_use_orpheus_pcm_streaming()
        idx = 0
        while True:
            chunk = await self._text_queue.get()
            if chunk is None:
                await self._audio_queue.put(None)
                break

            idx += 1
            words = len(TextChunker._word_spans(chunk))
            elapsed = (self._loop.time() - self._started_at) if self._started_at is not None else 0.0
            if use_pcm_streaming:
                self._delivered_any = True
                await self._audio_queue.put(("pcm", idx, chunk))
                logger.info(
                    "Pipeline queued direct PCM stream for chunk %d words=%d chars=%d elapsed=%.2fs text=%r",
                    idx,
                    words,
                    len(chunk),
                    elapsed,
                    chunk[:160],
                )
                continue

            audio_path = os.path.join(
                tempfile.gettempdir(),
                "hermes_voice",
                f"pipeline_stream_{id(self)}_{idx}.mp3",
            )
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            try:
                def run_tts():
                    previous_override = os.environ.get("HERMES_TTS_PROVIDER_OVERRIDE")
                    if self.tts_provider_override:
                        os.environ["HERMES_TTS_PROVIDER_OVERRIDE"] = self.tts_provider_override
                    try:
                        return text_to_speech_tool(text=chunk, output_path=audio_path)
                    finally:
                        if previous_override is None:
                            os.environ.pop("HERMES_TTS_PROVIDER_OVERRIDE", None)
                        else:
                            os.environ["HERMES_TTS_PROVIDER_OVERRIDE"] = previous_override

                result_json = await asyncio.to_thread(run_tts)
                result = json.loads(result_json)
                actual_path = result.get("file_path", audio_path)
                if result.get("success") and os.path.isfile(actual_path):
                    self._generated_paths.append(actual_path)
                    self._delivered_any = True
                    await self._audio_queue.put(actual_path)
                else:
                    logger.warning("Pipeline TTS synthesis failed for chunk '%s': %s", chunk, result.get("error"))
            except Exception as e:
                logger.error("Pipeline TTS synthesis exception: %s", e, exc_info=True)

    async def _playback_worker(self):
        from tools.tts_tool import iter_orpheus_pcm_chunks

        receivers = getattr(self.adapter, "_voice_receivers", {})
        receiver = receivers.get(self.guild_id) if isinstance(receivers, dict) else None
        if receiver and hasattr(receiver, "reset_barge_in"):
            receiver.reset_barge_in()

        while True:
            audio_item = await self._audio_queue.get()
            if audio_item is None:
                break

            if receiver and getattr(receiver, "_barge_in_triggered", False):
                logger.info("Pipeline playback skipped rest of queue due to barge-in")
                break

            try:
                if isinstance(audio_item, tuple) and audio_item and audio_item[0] == "pcm":
                    if len(audio_item) == 3:
                        _, chunk_idx, chunk_text = audio_item
                    else:
                        chunk_idx, chunk_text = 0, audio_item[1]
                    await self.adapter.play_pcm_stream_in_voice_channel(
                        self.guild_id,
                        self._capture_pcm_debug_stream(
                            self._limit_pcm_stream(
                                self._limit_numeric_pcm_stream(
                                    iter_orpheus_pcm_chunks(chunk_text),
                                    int(chunk_idx),
                                    str(chunk_text),
                                ),
                                int(chunk_idx),
                                str(chunk_text),
                            ),
                            int(chunk_idx),
                            str(chunk_text),
                        ),
                    )
                else:
                    await self.adapter.play_in_voice_channel(self.guild_id, audio_item)
            except Exception as e:
                logger.error("Pipeline playback failed for %r: %s", audio_item, e)

    def _pcm_debug_enabled(self) -> bool:
        raw = os.getenv("HERMES_ORPHEUS_PCM_DEBUG", "false")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _capture_pcm_debug_stream(self, chunks: Iterable[bytes], idx: int, text: str) -> Iterable[bytes]:
        """Optionally tee raw Orpheus PCM to WAV/text files while streaming."""
        if not self._pcm_debug_enabled():
            yield from chunks
            return

        debug_dir = os.getenv(
            "HERMES_ORPHEUS_PCM_DEBUG_DIR",
            os.path.join(tempfile.gettempdir(), "hermes_voice", "orpheus_pcm_debug"),
        )
        os.makedirs(debug_dir, exist_ok=True)
        base = f"guild_{self.guild_id}_pipeline_{id(self)}_chunk_{idx:02d}"
        wav_path = os.path.join(debug_dir, f"{base}.wav")
        txt_path = os.path.join(debug_dir, f"{base}.txt")
        self._debug_paths.extend([wav_path, txt_path])
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        try:
            with wave.open(wav_path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                total = 0
                for chunk in chunks:
                    if chunk:
                        wav.writeframes(chunk)
                        total += len(chunk)
                    yield chunk
            logger.info("Captured Orpheus PCM debug chunk %d bytes=%d path=%s text=%r", idx, total, wav_path, text[:160])
        except Exception as exc:
            logger.warning("Failed to capture Orpheus PCM debug chunk %d: %s", idx, exc)
            yield from chunks

    def _limit_pcm_stream(self, chunks: Iterable[bytes], idx: int, text: str) -> Iterable[bytes]:
        """Apply a general Orpheus PCM byte cap to prevent runaway synthesis."""
        raw_limit = os.getenv("HERMES_ORPHEUS_PCM_MAX_BYTES", "921600")
        try:
            max_bytes = int(raw_limit)
        except (TypeError, ValueError):
            max_bytes = 921600
        if max_bytes <= 0:
            yield from chunks
            return

        total = 0
        for chunk in chunks:
            if not chunk:
                yield chunk
                continue
            remaining = max_bytes - total
            if remaining <= 0:
                logger.warning(
                    "Stopped Orpheus PCM chunk %d at %d bytes to prevent runaway synthesis text=%r",
                    idx,
                    total,
                    text[:160],
                )
                break
            if len(chunk) > remaining:
                yield chunk[:remaining]
                total += remaining
                logger.warning(
                    "Truncated Orpheus PCM chunk %d at %d bytes to prevent runaway synthesis text=%r",
                    idx,
                    total,
                    text[:160],
                )
                break
            yield chunk
            total += len(chunk)

    def _limit_numeric_pcm_stream(self, chunks: Iterable[bytes], idx: int, text: str) -> Iterable[bytes]:
        """Cap numeric-count PCM chunks so Orpheus loops cannot repeat forever."""
        words = TextChunker._word_spans(text)
        is_numeric = bool(words) and all(TextChunker._is_number_word(word.group(0)) for word in words)
        if not is_numeric:
            yield from chunks
            return

        raw_limit = os.getenv("HERMES_ORPHEUS_NUMERIC_PCM_MAX_BYTES", "120000")
        try:
            max_bytes = int(raw_limit)
        except (TypeError, ValueError):
            max_bytes = 120000
        if max_bytes <= 0:
            yield from chunks
            return

        total = 0
        for chunk in chunks:
            if not chunk:
                yield chunk
                continue
            remaining = max_bytes - total
            if remaining <= 0:
                logger.warning(
                    "Stopped Orpheus numeric PCM chunk %d after %d bytes to prevent count-loop text=%r",
                    idx,
                    total,
                    text[:160],
                )
                break
            if len(chunk) > remaining:
                yield chunk[:remaining]
                total += remaining
                logger.warning(
                    "Truncated Orpheus numeric PCM chunk %d at %d bytes to prevent count-loop text=%r",
                    idx,
                    total,
                    text[:160],
                )
                break
            yield chunk
            total += len(chunk)

    async def close(self):
        """Wait for worker tasks to complete and cleanup temporary files."""
        self._closed = True
        if self._synthesis_task:
            try:
                await self._synthesis_task
            except Exception:
                pass
        if self._playback_task:
            try:
                await self._playback_task
            except Exception:
                pass

        # Emit structured debug record per turn when enabled
        if os.getenv("HERMES_DISCORD_VOICE_DEBUG", "") == "1":
            provider = (self.tts_provider_override or "").strip().lower()
            endpoint = ""
            tts_config = {}
            if not provider:
                try:
                    from tools.tts_tool import _get_provider, _load_tts_config
                    tts_config = _load_tts_config()
                    provider = _get_provider(tts_config)
                except Exception:
                    provider = provider or ""
            else:
                try:
                    from tools.tts_tool import _load_tts_config
                    tts_config = _load_tts_config()
                except Exception:
                    tts_config = {}
            if provider:
                try:
                    from tools.tts_tool import _get_named_provider_config
                    provider_config = tts_config.get(provider, {}) if isinstance(tts_config, dict) else {}
                    if not isinstance(provider_config, dict):
                        provider_config = {}
                    if not provider_config:
                        provider_config = _get_named_provider_config(tts_config, provider)
                    endpoint = provider_config.get("base_url", "")
                except Exception:
                    endpoint = ""

            playback_mode = "pcm" if self._should_use_orpheus_pcm_streaming() else "file"

            elapsed = (self._loop.time() - self._started_at) if self._started_at is not None else 0.0

            record = {
                "turn_id": self._turn_id,
                "guild_id": self.guild_id,
                "provider": provider,
                "endpoint": endpoint,
                "playback_mode": playback_mode,
                "target_words": self.target_words,
                "numeric_list_target": self.numeric_list_target,
                "min_chunk_words": self.min_chunk_words,
                "max_chars": self.max_chars,
                "attempted_any": self._attempted_any,
                "delivered_any": self._delivered_any,
                "chunk_count": len(self._queued_chunks),
                "chunks": list(self._queued_chunks),
                "pipeline_flags": {
                    "final_only": os.getenv("HERMES_DISCORD_VOICE_PIPELINE_FINAL_ONLY", "false").strip().lower()
                    in {"1", "true", "yes", "on"},
                    "streaming_tts_enabled": os.getenv("HERMES_DISCORD_VOICE_STREAMING_TTS", "false").strip().lower()
                    in {"1", "true", "yes", "on"},
                },
                "elapsed_ms": round(elapsed * 1000, 1),
            }
            logger.info("voice_tts_turn=%s", json.dumps(record, ensure_ascii=False))

        for path in self._generated_paths:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass
        self._generated_paths.clear()
