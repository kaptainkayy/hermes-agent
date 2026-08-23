#!/usr/bin/env python3
"""Safely store Telnyx secrets in ~/.hermes/.env without echoing them."""

from __future__ import annotations

import getpass
from pathlib import Path

ENV_PATH = Path('/home/kayai3/.hermes/.env')
ENV_PATH.parent.mkdir(parents=True, exist_ok=True)


def upsert_env(key: str, value: str) -> None:
    lines = ENV_PATH.read_text(errors='ignore').splitlines() if ENV_PATH.exists() else []
    out = []
    done = False
    for line in lines:
        if line.startswith(key + '='):
            out.append(f'{key}={value}')
            done = True
        else:
            out.append(line)
    if not done:
        if out and out[-1].strip():
            out.append('')
        out.append(f'{key}={value}')
    ENV_PATH.write_text('\n'.join(out) + '\n')


def main() -> None:
    api_key = getpass.getpass('Paste TELNYX_API_KEY (hidden): ').strip()
    if not api_key:
        raise SystemExit('No TELNYX_API_KEY entered; not changing file.')
    upsert_env('TELNYX_API_KEY', api_key)

    public_key = getpass.getpass('Paste TELNYX_PUBLIC_KEY if you have it, or press Enter to skip (hidden): ').strip()
    if public_key:
        upsert_env('TELNYX_PUBLIC_KEY', public_key)

    print(f'Saved Telnyx secret(s) to {ENV_PATH}')
    text = ENV_PATH.read_text(errors='ignore')
    for k in ['TELNYX_API_KEY', 'TELNYX_PUBLIC_KEY']:
        val = ''
        for line in text.splitlines():
            if line.startswith(k + '='):
                val = line.split('=', 1)[1].strip()
        print(f'{k}: {"set len=" + str(len(val)) if val else "missing/empty"}')


if __name__ == '__main__':
    main()
