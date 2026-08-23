#!/usr/bin/env python3
"""Generate a combined Orpheus voice sampler audio file."""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import tts_tool  # noqa: E402

VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe", "julia"]
BASE_URL = os.environ.get("HERMES_ORPHEUS_BASE_URL", "http://127.0.0.1:5005/v1")
MODEL = os.environ.get("HERMES_ORPHEUS_MODEL", "orpheus")
API_KEY = os.environ.get("HERMES_ORPHEUS_API_KEY", "not-needed")

SAMPLE_TEXT = (
    "This is {voice}, one of the Persephone voices. "
    "I can be used for the Hermes Discord bot voice stack."
)

def main() -> int:
    hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
    out_dir = hermes_home / "cache" / "audio" / f"orpheus_voice_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    failures = []

    for voice in VOICES:
        output = out_dir / f"orpheus_{voice}.mp3"
        config = {
            "provider": "orpheus",
            "orpheus": {
                "base_url": BASE_URL,
                "api_key": API_KEY,
                "model": MODEL,
                "voice": voice,
                "speed": 1.0,
            },
        }
        old_loader = tts_tool._load_tts_config
        tts_tool._load_tts_config = lambda cfg=config: cfg
        try:
            result = json.loads(tts_tool.text_to_speech_tool(SAMPLE_TEXT.format(voice=voice.title()), output_path=str(output)))
            if result.get("success") and Path(result.get("file_path", output)).exists():
                generated.append((voice, Path(result.get("file_path", output))))
                print(f"OK {voice}: {result.get('file_path', output)}")
            else:
                failures.append((voice, result.get("error", "unknown error")))
                print(f"FAIL {voice}: {result}")
        except Exception as exc:
            failures.append((voice, repr(exc)))
            print(f"FAIL {voice}: {exc!r}")
        finally:
            tts_tool._load_tts_config = old_loader

    if not generated:
        print("No samples generated", file=sys.stderr)
        return 1

    silence = out_dir / "silence_450ms.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "0.45",
        "-q:a", "9", "-acodec", "libmp3lame", str(silence)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    concat_list = out_dir / "concat.txt"
    with concat_list.open("w") as f:
        for _voice, path in generated:
            f.write(f"file '{path.as_posix()}'\n")
            f.write(f"file '{silence.as_posix()}'\n")

    combined = out_dir / "orpheus_all_voices_sampler.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy", str(combined)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "voices_requested": VOICES,
        "generated": [{"voice": v, "path": str(p)} for v, p in generated],
        "failures": [{"voice": v, "error": e} for v, e in failures],
        "combined": str(combined),
        "base_url": BASE_URL,
        "model": MODEL,
    }, indent=2))

    print("COMBINED=" + str(combined))
    print("MANIFEST=" + str(manifest))
    if failures:
        print("FAILURES=" + json.dumps(failures))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
