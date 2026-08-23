#!/usr/bin/env python3
"""Temporary Telnyx webhook placeholder.

This is only to let Telnyx save/validate the webhook while the real
streaming voice bridge is being built.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

from aiohttp import web

LOG_PATH = os.getenv(
    "TELNYX_PLACEHOLDER_LOG",
    "/home/kayai3/.hermes/logs/telnyx-placeholder.log",
)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH),
    ],
)
log = logging.getLogger("telnyx-placeholder")


async def health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "telnyx-placeholder"})


async def telnyx_webhook(request: web.Request) -> web.Response:
    raw = await request.read()
    headers = {
        "telnyx-signature-ed25519": bool(request.headers.get("telnyx-signature-ed25519")),
        "telnyx-timestamp": request.headers.get("telnyx-timestamp", ""),
        "content-type": request.headers.get("content-type", ""),
    }
    body_preview = raw[:1000].decode("utf-8", "replace")
    log.info("%s /telnyx/webhook headers=%s body=%s", request.method, headers, body_preview)

    # Telnyx UI/testers may probe the webhook with GET/HEAD/OPTIONS before real
    # call events arrive via POST. Accept them during placeholder setup so the
    # portal does not show 405 Method Not Allowed.
    if request.method == "HEAD":
        return web.Response(status=200)
    if request.method == "OPTIONS":
        return web.Response(status=204, headers={"Allow": "GET,POST,HEAD,OPTIONS"})

    # Acknowledge quickly. The real bridge will verify signatures, inspect event
    # types, answer calls, and start media streaming.
    return web.json_response(
        {
            "ok": True,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "placeholder": True,
            "method": request.method,
        }
    )


async def telnyx_media(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    peer = request.remote
    log.info("WS /telnyx/media connected peer=%s", peer)
    await ws.send_str(json.dumps({"event": "placeholder_connected"}))
    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            log.info("media text frame: %s", msg.data[:500])
        elif msg.type == web.WSMsgType.BINARY:
            log.info("media binary frame: %d bytes", len(msg.data))
        elif msg.type == web.WSMsgType.ERROR:
            log.warning("media ws error: %s", ws.exception())
    log.info("WS /telnyx/media disconnected peer=%s", peer)
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
    log.info("Starting Telnyx placeholder on http://%s:%s", host, port)
    web.run_app(make_app(), host=host, port=port)
