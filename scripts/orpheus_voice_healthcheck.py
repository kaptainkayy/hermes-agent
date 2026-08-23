#!/usr/bin/env python3
"""Deterministic healthcheck for the Persephone Discord voice profile."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_PROFILE_HOME = Path("/home/kayai3/.hermes/profiles/orpheusab")
AGENT_ROOT = Path("/home/kayai3/.hermes/hermes-agent")
VENV_PYTHON = AGENT_ROOT / "venv/bin/python"
SMOKE_SCRIPT = Path(
    "/home/kayai3/AiSetupResearch/.opencode/skills/discord-tts-live-smoke/scripts/automated_tts_smoke.py"
)
SERVICE = "hermes-gateway-orpheusab.service"
TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
MEMORY_RE = re.compile(r"\[MEMORY\]\s+rss=(\d+)MB.*?threads=(\d+).*?uptime=(\d+)s")
CHUNK_RE = re.compile(r"Pipeline queued direct PCM stream for chunk (\d+)\b")
READY_RE = re.compile(r"response ready: .*?time=([0-9.]+)s.*?response=(\d+) chars")


@dataclass
class LogLine:
    path: str
    timestamp: datetime | None
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Persephone Discord voice health.")
    parser.add_argument("--profile-home", type=Path, default=DEFAULT_PROFILE_HOME)
    parser.add_argument("--hours", type=float, default=6)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--strict-exit", action="store_true", help="Exit non-zero when status is critical.")
    parser.add_argument("--asr-case", action="append", dest="asr_cases")
    parser.add_argument("--asr-model", default="base")
    parser.add_argument("--asr-min-similarity", type=float, default=0.72)
    return parser.parse_args()


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    return datetime.strptime(f"{match.group(1)}.{match.group(2)}", "%Y-%m-%d %H:%M:%S.%f")


def read_recent_logs(profile_home: Path, hours: float) -> list[LogLine]:
    cutoff = datetime.now() - timedelta(hours=hours)
    results: list[LogLine] = []
    for name in ("gateway.log", "agent.log", "errors.log"):
        path = profile_home / "logs" / name
        if not path.exists():
            continue
        last_in_window = False
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            ts = parse_timestamp(raw)
            if ts is not None:
                last_in_window = ts >= cutoff
            if last_in_window:
                results.append(LogLine(name, ts, raw))
    return results


def check_service() -> dict[str, Any]:
    command = [
        "systemctl",
        "--user",
        "show",
        SERVICE,
        "--property=ActiveState,SubState,MainPID,MemoryCurrent,MemoryPeak,ExecMainStartTimestamp",
    ]
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
    except Exception as exc:  # pragma: no cover - defensive for non-systemd hosts
        return {"ok": False, "error": str(exc), "properties": {}}
    props: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            props[key] = value
    return {
        "ok": proc.returncode == 0 and props.get("ActiveState") == "active",
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
        "properties": props,
    }


def check_http(url: str) -> dict[str, Any]:
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - localhost healthcheck
            return {"ok": response.status == 200, "status": response.status}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def summarize_logs(lines: list[LogLine]) -> dict[str, Any]:
    text_lines = [line.text for line in lines]
    joined = "\n".join(text_lines)
    counts = {
        "accepted_transcripts": sum("Voice input from user" in line for line in text_lines),
        "barge_in": sum("barge-in" in line.lower() for line in text_lines),
        "whisper_hallucination": joined.count("whisper_hallucination"),
        "high_no_speech_probability": joined.count("high_no_speech_probability"),
        "missing_temp_wav": sum("vc_listen_" in line and ("No such file" in line or "FileNotFound" in line) for line in text_lines),
        "pcm_timeout": joined.count("PCM streaming playback timed out"),
        "tts_synthesis_failed": joined.count("Pipeline TTS synthesis failed"),
        "playback_failed": joined.count("Pipeline playback failed"),
        "pcm_truncated": joined.count("Truncated Orpheus PCM chunk"),
        "pcm_stopped": joined.count("Stopped Orpheus PCM chunk"),
    }
    return {
        "counts": counts,
        "memory": summarize_memory(lines),
        "long_responses": summarize_long_responses(lines),
    }


def summarize_memory(lines: list[LogLine]) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for line in lines:
        match = MEMORY_RE.search(line.text)
        if match:
            points.append(
                {
                    "timestamp": line.timestamp.isoformat(sep=" ") if line.timestamp else None,
                    "rss_mb": int(match.group(1)),
                    "threads": int(match.group(2)),
                    "uptime_s": int(match.group(3)),
                }
            )
    if not points:
        return {"points": 0}
    first = points[0]["rss_mb"]
    last = points[-1]["rss_mb"]
    delta = last - first
    elapsed_hours = max((len(points) - 1) * 5 / 60, 1 / 60)
    return {
        "points": len(points),
        "first_mb": first,
        "last_mb": last,
        "min_mb": min(point["rss_mb"] for point in points),
        "max_mb": max(point["rss_mb"] for point in points),
        "delta_mb": delta,
        "mb_per_hour": round(delta / elapsed_hours, 1),
        "threads_min": min(point["threads"] for point in points),
        "threads_max": max(point["threads"] for point in points),
    }


def summarize_long_responses(lines: list[LogLine]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    chunk_max = 0
    chunk_started: datetime | None = None
    for line in lines:
        if line.path != "gateway.log":
            continue
        chunk_match = CHUNK_RE.search(line.text)
        if chunk_match:
            chunk_max = max(chunk_max, int(chunk_match.group(1)))
            chunk_started = chunk_started or line.timestamp
            continue
        ready_match = READY_RE.search(line.text)
        if ready_match:
            seconds = float(ready_match.group(1))
            chars = int(ready_match.group(2))
            severity = "ok"
            reasons = []
            if chunk_max > 10 or seconds > 35:
                severity = "critical"
            elif chunk_max > 6 or seconds > 20 or chars > 500:
                severity = "warning"
            if chunk_max > 6:
                reasons.append(f"chunks={chunk_max}")
            if seconds > 20:
                reasons.append(f"seconds={seconds:.1f}")
            if chars > 500:
                reasons.append(f"chars={chars}")
            if severity != "ok":
                responses.append(
                    {
                        "timestamp": (chunk_started or line.timestamp).isoformat(sep=" ") if (chunk_started or line.timestamp) else None,
                        "chunks": chunk_max,
                        "seconds": seconds,
                        "chars": chars,
                        "severity": severity,
                        "reasons": reasons,
                    }
                )
            chunk_max = 0
            chunk_started = None
    return responses


def service_memory_mb(service: dict[str, Any]) -> int:
    value = service.get("properties", {}).get("MemoryCurrent")
    try:
        return int(value) // (1024 * 1024)
    except (TypeError, ValueError):
        return 0


def run_asr_smoke(args: argparse.Namespace) -> dict[str, Any]:
    cases = args.asr_cases or ["orchestrator_approval_gate", "orchestrator_repetitive_confirmations"]
    command = [
        str(VENV_PYTHON),
        str(SMOKE_SCRIPT),
        "--mode",
        "tts-only",
        "--voice-chunks",
        "--gateway-caps",
        "--asr-listen",
        "--asr-model",
        args.asr_model,
        "--asr-min-similarity",
        str(args.asr_min_similarity),
        "--json",
    ]
    for case in cases:
        command.extend(["--case", case])
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=600, check=False)
    except Exception as exc:
        return {"status": "critical", "error": str(exc), "cases": cases}
    parsed: Any = None
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        pass
    result = {
        "status": "healthy" if proc.returncode == 0 else "critical",
        "returncode": proc.returncode,
        "cases": cases,
        "stdout_excerpt": proc.stdout[-1200:],
        "stderr_excerpt": proc.stderr[-1200:],
        "summary": parsed,
    }
    similarities: list[float] = []
    failures: list[str] = []
    output_dir = None
    if isinstance(parsed, dict):
        output_dir = parsed.get("output_dir") or parsed.get("run_dir")
        for item in parsed.get("results", []):
            if isinstance(item, dict):
                if item.get("asr_similarity") is not None:
                    similarities.append(float(item.get("asr_similarity") or 0))
                failures.extend(str(failure) for failure in item.get("failures", []))
    if similarities:
        result["min_similarity"] = min(similarities)
    if failures:
        result["failures"] = failures
        result["status"] = "critical" if proc.returncode else "warning"
    if output_dir:
        result["output_dir"] = output_dir
    return result


def overall_status(service: dict[str, Any], endpoints: dict[str, Any], logs: dict[str, Any], asr: dict[str, Any] | None) -> str:
    counts = logs["counts"]
    memory = logs["memory"]
    current_memory_mb = memory.get("last_mb", 0) or service_memory_mb(service)
    long_responses = logs["long_responses"]
    critical = (
        not service.get("ok")
        or any(not endpoint.get("ok") for endpoint in endpoints.values())
        or counts["pcm_timeout"]
        or counts["tts_synthesis_failed"]
        or counts["playback_failed"]
        or any(item["severity"] == "critical" for item in long_responses)
        or (asr is not None and asr.get("status") == "critical")
        or abs(memory.get("mb_per_hour", 0)) > 750
        or current_memory_mb > 5000
    )
    if critical:
        return "critical"
    warning = (
        counts["missing_temp_wav"]
        or counts["pcm_truncated"]
        or counts["pcm_stopped"]
        or long_responses
        or current_memory_mb > 3500
        or abs(memory.get("mb_per_hour", 0)) > 250
        or (asr is not None and asr.get("status") == "warning")
    )
    return "warning" if warning else "healthy"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    lines = read_recent_logs(args.profile_home, args.hours)
    service = check_service()
    endpoints = {
        "orpheus_tts": check_http("http://127.0.0.1:5005/"),
        "llama_health": check_http("http://127.0.0.1:5006/health"),
    }
    logs = summarize_logs(lines)
    asr = None if args.skip_asr else run_asr_smoke(args)
    status = overall_status(service, endpoints, logs, asr)
    return {
        "status": status,
        "profile_home": str(args.profile_home),
        "hours": args.hours,
        "service": service,
        "endpoints": endpoints,
        "logs": logs,
        "asr_smoke": asr,
    }


def markdown(report: dict[str, Any]) -> str:
    props = report["service"].get("properties", {})
    endpoints = report["endpoints"]
    counts = report["logs"]["counts"]
    memory = report["logs"]["memory"]
    long_responses = report["logs"]["long_responses"]
    asr = report.get("asr_smoke")
    lines = [f"# Persephone/Orpheus Healthcheck: {report['status'].upper()}", ""]
    lines.extend(
        [
            "## Service",
            f"- Gateway: {props.get('ActiveState', 'unknown')}/{props.get('SubState', 'unknown')} PID {props.get('MainPID', 'unknown')}",
            f"- Orpheus TTS: HTTP {endpoints['orpheus_tts'].get('status', endpoints['orpheus_tts'].get('error', 'unknown'))}",
            f"- llama.cpp: HTTP {endpoints['llama_health'].get('status', endpoints['llama_health'].get('error', 'unknown'))}",
            "",
            "## ASR Smoke",
        ]
    )
    if asr is None:
        lines.append("- Skipped")
    else:
        lines.append(f"- Status: {asr.get('status')}")
        if asr.get("min_similarity") is not None:
            lines.append(f"- Min similarity: {asr['min_similarity']:.2f}")
        if asr.get("output_dir"):
            lines.append(f"- Artifacts: {asr['output_dir']}")
        if asr.get("failures"):
            lines.append(f"- Failures: {len(asr['failures'])}")
    lines.extend(
        [
            "",
            "## Voice Runtime",
            f"- Accepted transcripts: {counts['accepted_transcripts']}",
            f"- Barge-in events: {counts['barge_in']}",
            f"- Whisper rejections: {counts['whisper_hallucination'] + counts['high_no_speech_probability']}",
            f"- Missing temp WAV errors: {counts['missing_temp_wav']}",
            f"- Long responses: {len(long_responses)}",
        ]
    )
    if long_responses:
        worst = max(long_responses, key=lambda item: (item["severity"] == "critical", item["seconds"], item["chunks"]))
        lines.append(f"- Worst response: {worst['chunks']} chunks, {worst['seconds']:.1f}s, {worst['chars']} chars")
    lines.extend(
        [
            "",
            "## Memory",
            f"- RSS: {memory.get('last_mb', 'unknown')}MB (min {memory.get('min_mb', 'unknown')}, max {memory.get('max_mb', 'unknown')})",
            f"- Trend: {memory.get('delta_mb', 'unknown')}MB, {memory.get('mb_per_hour', 'unknown')}MB/hour",
            "",
            "## Errors",
            f"- Critical TTS/PCM failures: {counts['pcm_timeout'] + counts['tts_synthesis_failed'] + counts['playback_failed']}",
            f"- PCM truncation/stop warnings: {counts['pcm_truncated'] + counts['pcm_stopped']}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(markdown(report))
    return 2 if args.strict_exit and report["status"] == "critical" else 0


if __name__ == "__main__":
    raise SystemExit(main())
