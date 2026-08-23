#!/usr/bin/env python3
"""Minimal Telnyx Call Control bridge for Hermes live-call plumbing tests.

MVP behavior:
- Accept Telnyx webhooks on /telnyx/webhook.
- On call.initiated, answer the call and request media streaming to /telnyx/media.
- On call.answered, speak a short test prompt so the caller can verify audio.
- Log all events to ~/.hermes/logs/telnyx-bridge.log without printing secrets.

This is not yet the full Hermes coach loop; it is the smallest useful bridge to
prove Telnyx routing + Call Control API + media WebSocket are working.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

HERMES_HOME = Path(os.getenv("HERMES_HOME", "/home/kayai3/.hermes"))
ENV_PATH = HERMES_HOME / ".env"
LOG_PATH = HERMES_HOME / "logs" / "telnyx-bridge.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)],
)
log = logging.getLogger("telnyx-bridge")


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env_file()

TELNYX_API_KEY = os.getenv("TELNYX_API_KEY", "").strip()
TELNYX_MEDIA_WS_URL = os.getenv(
    "TELNYX_MEDIA_WS_URL",
    "wss://kayai3-ubuntu.gecko-decibel.ts.net/telnyx/media",
).strip()
TELNYX_WEBHOOK_URL = os.getenv(
    "TELNYX_WEBHOOK_URL",
    "https://kayai3-ubuntu.gecko-decibel.ts.net/telnyx/webhook",
).strip()
TELNYX_API_BASE = os.getenv("TELNYX_API_BASE", "https://api.telnyx.com/v2").rstrip("/")
TELNYX_SPEAK_VOICE = os.getenv("TELNYX_SPEAK_VOICE", "Telnyx.KateLight.en-US").strip()
TELNYX_ALLOWED_CALLERS = {
    x.strip() for x in os.getenv("TELNYX_ALLOWED_CALLERS", "").split(",") if x.strip()
}

answered_calls: set[str] = set()
spoken_calls: set[str] = set()


def event_summary(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data") or {}
    inner = data.get("payload") or data
    return {
        "event_type": data.get("event_type"),
        "id": data.get("id"),
        "call_control_id": inner.get("call_control_id"),
        "call_leg_id": inner.get("call_leg_id"),
        "from": inner.get("from"),
        "to": inner.get("to"),
        "direction": inner.get("direction"),
    }


def extract_payload(body: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    data = body.get("data") or {}
    event_type = data.get("event_type", "")
    payload = data.get("payload") or data
    call_control_id = payload.get("call_control_id")
    return event_type, payload, call_control_id


async def telnyx_api(action_path: str, payload: dict[str, Any]) -> tuple[int, str]:
    if not TELNYX_API_KEY:
        log.error("TELNYX_API_KEY is missing; cannot call Telnyx API action=%s", action_path)
        return 0, "missing TELNYX_API_KEY"
    url = f"{TELNYX_API_BASE}{action_path}"
    headers = {
        "Authorization": f"Bearer {TELNYX_API_KEY}",
        "Content-Type": "application/json",
    }
    timeout = ClientTimeout(total=15)
    async with ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            text = await resp.text()
            if resp.status >= 300:
                log.error("Telnyx API error action=%s status=%s body=%s", action_path, resp.status, text[:2000])
            else:
                log.info("Telnyx API ok action=%s status=%s body=%s", action_path, resp.status, text[:1000])
            return resp.status, text


async def answer_call(call_control_id: str) -> None:
    if call_control_id in answered_calls:
        return
    answered_calls.add(call_control_id)
    payload = {
        "command_id": f"hermes-answer-{uuid.uuid4()}",
        "stream_url": TELNYX_MEDIA_WS_URL,
        "stream_track": "both_tracks",
        "stream_codec": "PCMU",
        "stream_bidirectional_mode": "rtp",
        "stream_bidirectional_codec": "PCMU",
        "stream_bidirectional_target_legs": "self",
        "send_silence_when_idle": True,
        "webhook_url": TELNYX_WEBHOOK_URL,
        "webhook_url_method": "POST",
    }
    log.info("Answering call_control_id=%s media=%s", call_control_id, TELNYX_MEDIA_WS_URL)
    await telnyx_api(f"/calls/{call_control_id}/actions/answer", payload)


async def speak_test_prompt(call_control_id: str) -> None:
    if call_control_id in spoken_calls:
        return
    spoken_calls.add(call_control_id)
    payload = {
        "command_id": f"hermes-speak-{uuid.uuid4()}",
        "payload": "Hermes bridge connected. This is a first call control test. The coaching brain is not attached yet.",
        "payload_type": "text",
        "service_level": "basic",
        "voice": TELNYX_SPEAK_VOICE,
    }
    log.info("Speaking test prompt call_control_id=%s voice=%s", call_control_id, TELNYX_SPEAK_VOICE)
    await telnyx_api(f"/calls/{call_control_id}/actions/speak", payload)


async def health(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "service": "telnyx-bridge",
            "api_key_present": bool(TELNYX_API_KEY),
            "media_ws_url": TELNYX_MEDIA_WS_URL,
        }
    )


async def telnyx_webhook(request: web.Request) -> web.Response:
    raw = await request.read()
    headers = {
        "telnyx-signature-ed25519": bool(request.headers.get("telnyx-signature-ed25519")),
        "telnyx-timestamp": bool(request.headers.get("telnyx-timestamp")),
        "content-type": request.headers.get("content-type", ""),
    }

    if request.method == "HEAD":
        return web.Response(status=200)
    if request.method == "OPTIONS":
        return web.Response(status=204, headers={"Allow": "GET,POST,HEAD,OPTIONS"})
    if request.method == "GET":
        return web.json_response({"ok": True, "service": "telnyx-bridge", "method": "GET"})

    try:
        body = json.loads(raw.decode("utf-8")) if raw else {}
    except Exception:
        body = {"_raw": raw[:1000].decode("utf-8", "replace")}

    summary = event_summary(body if isinstance(body, dict) else {})
    log.info("%s /telnyx/webhook headers=%s summary=%s", request.method, headers, summary)

    if isinstance(body, dict):
        event_type, payload, call_control_id = extract_payload(body)
        caller = str(payload.get("from") or "")
        if TELNYX_ALLOWED_CALLERS and caller and caller not in TELNYX_ALLOWED_CALLERS:
            log.warning("Ignoring non-allowlisted caller=%s call_control_id=%s", caller, call_control_id)
        elif call_control_id:
            if event_type == "call.initiated":
                asyncio.create_task(answer_call(call_control_id))
            elif event_type == "call.answered":
                asyncio.create_task(speak_test_prompt(call_control_id))
            elif event_type in {"call.hangup", "call.finalized"}:
                log.info("Call ended event=%s call_control_id=%s", event_type, call_control_id)
        else:
            log.info("No call_control_id in event_type=%s", event_type)

    return web.json_response(
        {
            "ok": True,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "service": "telnyx-bridge",
        }
    )


async def telnyx_media(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    peer = request.remote
    log.info("WS /telnyx/media connected peer=%s", peer)
    frame_count = 0
    try:
        async for msg in ws:
            frame_count += 1
            if frame_count <= 5 or frame_count % 100 == 0:
                if msg.type == web.WSMsgType.TEXT:
                    log.info("media text frame #%s: %s", frame_count, msg.data[:500])
                elif msg.type == web.WSMsgType.BINARY:
                    log.info("media binary frame #%s: %d bytes", frame_count, len(msg.data))
                elif msg.type == web.WSMsgType.ERROR:
                    log.warning("media ws error: %s", ws.exception())
    finally:
        log.info("WS /telnyx/media disconnected peer=%s frames=%s", peer, frame_count)
    return ws


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_route("GET", "/telnyx/webhook", telnyx_webhook)
    app.router.add_route("POST", "/telnyx/webhook", telnyx_webhook)
    app.router.add_route("OPTIONS", "/telnyx/webhook", telnyx_webhook)
    app.router.add_get("/telnyx/media", telnyx_media)
    return app


if __name__ == "__main__":
    host = os.getenv("TELNYX_BRIDGE_HOST", "127.0.0.1")
    port = int(os.getenv("TELNYX_BRIDGE_PORT", "8091"))
    log.info("Starting Telnyx bridge on http://%s:%s api_key_present=%s", host, port, bool(TELNYX_API_KEY))
    web.run_app(make_app(), host=host, port=port)
