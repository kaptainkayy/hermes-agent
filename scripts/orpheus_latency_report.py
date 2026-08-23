#!/usr/bin/env python3
"""
Orpheus Latency Report — parse agent/gateway logs and report voice turn timings.

Usage:
    python scripts/orpheus_latency_report.py [--agent-log PATH] [--gateway-log PATH] [--lines N]

Default log paths (in order):
    $HERMES_HOME/profiles/orpheusab/logs/agent.log
    $HERMES_HOME/logs/agent.log
    ~/.hermes/profiles/orpheusab/logs/agent.log
    ~/.hermes/logs/agent.log

Same pattern for gateway.log.

Output:
    Summary table with median stage timings and duplicate-TTS detection.
"""

import argparse
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# Log line patterns — adapted from Hermes logging conventions
RE_STT_DURATION = re.compile(r"STT.*?(?:took|duration|complete|in)\s+([\d.]+)\s*s", re.IGNORECASE)
RE_LLM_START = re.compile(r"(?:LLM|model|inference|generate).*?(?:start|begin|call)", re.IGNORECASE)
RE_LLM_END = re.compile(r"(?:LLM|model|inference|generate).*?(?:end|complete|done|finished|took)\s+([\d.]+)\s*s", re.IGNORECASE)
RE_LLM_TOKENS = re.compile(r"(?:output|generated|completion)\s*tokens?.*?(\d+)", re.IGNORECASE)
RE_PIPELINE_TTS_START = re.compile(r"Pipeline TTS synthesis.*?chunk", re.IGNORECASE)
RE_PIPELINE_STREAM_FILE = re.compile(r"pipeline_stream_")
RE_DISCORD_PLAYBACK = re.compile(r"(?:play_in_voice_channel|playback|playing)\s*(?:audio|file|started|complete)?", re.IGNORECASE)
RE_FINAL_RESPONSE = re.compile(r"final_response|response-ready|result_holder\[0\]", re.IGNORECASE)
RE_DUPLICATE_TTS = re.compile(r"(?:audio_cache/tts_|duplicate.*TTS|full.response.*TTS)", re.IGNORECASE)
RE_TURN_BOUNDARY = re.compile(r"(?:New turn|Processing|=== Turn|\d{2}:\d{2}:\d{2}.*voice)", re.IGNORECASE)
RE_TIMESTAMP = re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,\.]\d+)")
RE_PROVIDER_LINE = re.compile(r"provider:\s*(\w+)", re.IGNORECASE)


def resolve_paths(agent_log=None, gateway_log=None, lines=500):
    """Resolve log file paths from args, env, or defaults."""
    hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

    if agent_log:
        agent_path = agent_log if Path(agent_log).is_file() else None
    else:
        agent_path = None
        for c in [
            str(hermes_home / "profiles" / "orpheusab" / "logs" / "agent.log"),
            str(hermes_home / "logs" / "agent.log"),
            str(Path.home() / ".hermes" / "profiles" / "orpheusab" / "logs" / "agent.log"),
            str(Path.home() / ".hermes" / "logs" / "agent.log"),
        ]:
            if Path(c).is_file():
                agent_path = c
                break

    if gateway_log:
        gateway_path = gateway_log if Path(gateway_log).is_file() else None
    else:
        gateway_path = None
        for c in [
            str(hermes_home / "profiles" / "orpheusab" / "logs" / "gateway.log"),
            str(hermes_home / "logs" / "gateway.log"),
            str(Path.home() / ".hermes" / "profiles" / "orpheusab" / "logs" / "gateway.log"),
            str(Path.home() / ".hermes" / "logs" / "gateway.log"),
        ]:
            if Path(c).is_file():
                gateway_path = c
                break

    return agent_path, gateway_path


def tail_lines(path, n):
    """Read the last n lines of a file efficiently."""
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    return lines[-n:]


def parse_timings(agent_lines, gateway_lines):
    """Parse log lines and extract per-turn timing data."""
    turns = []
    current_turn = defaultdict(list)
    all_lines = list(agent_lines) + list(gateway_lines)
    all_lines.sort(key=lambda x: x)  # rough sort by timestamp prefix

    for line in all_lines:
        line = line.strip()

        if RE_TURN_BOUNDARY.search(line) and any(t in current_turn for t in ["timestamps"]):
            if current_turn:
                turns.append(dict(current_turn))
            current_turn = defaultdict(list)

        timestamp_match = RE_TIMESTAMP.search(line)
        if timestamp_match:
            current_turn.setdefault("timestamps", []).append(timestamp_match.group(1))

        if RE_STT_DURATION.search(line):
            m = RE_STT_DURATION.search(line)
            if m:
                try:
                    current_turn.setdefault("stt_durations", []).append(float(m.group(1)))
                except ValueError:
                    pass

        if RE_LLM_END.search(line):
            m = RE_LLM_END.search(line)
            if m:
                try:
                    current_turn.setdefault("llm_durations", []).append(float(m.group(1)))
                except ValueError:
                    pass

        if RE_LLM_TOKENS.search(line):
            m = RE_LLM_TOKENS.search(line)
            if m:
                current_turn.setdefault("output_tokens", []).append(int(m.group(1)))

        if RE_PIPELINE_TTS_START.search(line):
            current_turn["pipeline_tts_started"] = True

        if RE_PIPELINE_STREAM_FILE.search(line):
            current_turn["pipeline_stream_saved"] = True

        if RE_DISCORD_PLAYBACK.search(line):
            current_turn["discord_playback"] = True

        if RE_FINAL_RESPONSE.search(line):
            current_turn["final_response"] = True

        if RE_DUPLICATE_TTS.search(line):
            current_turn["duplicate_tts"] = True

        if RE_PROVIDER_LINE.search(line):
            m = RE_PROVIDER_LINE.search(line)
            if m:
                current_turn["provider"] = m.group(1)

    if current_turn:
        turns.append(dict(current_turn))

    return turns


def format_duration(seconds):
    """Format a duration in seconds for display."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"


def report(turns):
    """Generate latency report from parsed turn data."""
    if not turns:
        print("No voice turns found in the log range.")
        return

    # Collect timing data across turns
    stt_times = []
    llm_times = []
    pipeline_tts_turns = 0
    duplicate_tts_turns = 0
    total_turns = len(turns)

    for turn in turns:
        if turn.get("stt_durations"):
            stt_times.extend(turn["stt_durations"])
        if turn.get("llm_durations"):
            llm_times.extend(turn["llm_durations"])
        if turn.get("pipeline_tts_started"):
            pipeline_tts_turns += 1
        if turn.get("duplicate_tts"):
            duplicate_tts_turns += 1

    print("=" * 60)
    print("  Orpheus Voice Latency Report")
    print("=" * 60)
    print(f"  Turns analyzed:     {total_turns}")
    if stt_times:
        print(f"  STT median:          {format_duration(statistics.median(stt_times))}")
    if llm_times:
        print(f"  LLM TTS median:      {format_duration(statistics.median(llm_times))}")
    print(f"  Pipeline TTS active: {pipeline_tts_turns}/{total_turns}")
    print(f"  Duplicate TTS:       {'YES' if duplicate_tts_turns > 0 else 'no'}")
    print()

    # Summary stats
    if stt_times:
        print(f"  STT warm median:     {format_duration(statistics.median(stt_times))}")
    else:
        print("  STT warm median:     N/A")
    if llm_times:
        print(f"  LLM median:          {format_duration(statistics.median(llm_times))}")
    else:
        print("  LLM median:          N/A")

    # Bottleneck detection
    bottlenecks = []
    if stt_times and llm_times:
        median_stt = statistics.median(stt_times)
        median_llm = statistics.median(llm_times)
        if median_stt > median_llm and median_stt > 1.0:
            bottlenecks.append("STT cold start")
        if median_llm > median_stt and median_llm > 1.5:
            bottlenecks.append("LLM generation")
    if duplicate_tts_turns > 0:
        bottlenecks.append("duplicate full-response TTS")
    if pipeline_tts_turns < total_turns:
        bottlenecks.append("pipeline not active for all turns")

    if bottlenecks:
        print(f"  Top bottlenecks:     {' / '.join(bottlenecks)}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Orpheus Voice Latency Report")
    parser.add_argument("--agent-log", help="Path to agent.log")
    parser.add_argument("--gateway-log", help="Path to gateway.log")
    parser.add_argument("--lines", type=int, default=500, help="Number of recent log lines to analyze (default: 500)")
    args = parser.parse_args()

    agent_path, gateway_path = resolve_paths(args.agent_log, args.gateway_log)

    agent_lines = []
    gateway_lines = []

    if agent_path:
        print(f"  Agent log:   {agent_path}")
        agent_lines = tail_lines(agent_path, args.lines)
    else:
        print("  Agent log:   (not found)")

    if gateway_path:
        print(f"  Gateway log: {gateway_path}")
        gateway_lines = tail_lines(gateway_path, args.lines)
    else:
        print("  Gateway log: (not found)")

    if not agent_lines and not gateway_lines:
        print("No log files found. Specify --agent-log and/or --gateway-log.")
        return

    turns = parse_timings(agent_lines, gateway_lines)
    report(turns)


if __name__ == "__main__":
    main()
