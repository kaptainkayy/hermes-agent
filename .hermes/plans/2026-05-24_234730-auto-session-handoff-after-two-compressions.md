# Auto session handoff after two compressions

## Goal

When a logical conversation has been compressed twice, Hermes must automatically start a fresh session, alert the user that it is doing so, and inject the summarized context into the new session so work continues without the user manually running `/new`.

This is a core Hermes runtime feature, not a normal user-authored skill: a skill document can guide behavior after it is loaded, but it cannot reliably rotate the CLI/gateway session, update the SQLite session lineage, reset counters, and inject context across all platforms by itself.

## Current context

Relevant existing behavior:

- `agent/conversation_compression.py::compress_context()` performs the compression call.
- It already rotates `agent.session_id` after every successful compression and creates a new SQLite session row with `parent_session_id=old_session_id` and `boundary_reason="compression"` notifications.
- It preserves summarized context by returning `compressed` messages and rebuilding the system prompt.
- It currently only warns after repeated compression:
  - lines around 445-452: if `agent.context_compressor.compression_count >= 2`, it prints: `Session compressed N times — accuracy may degrade. Consider /new to start fresh.`
- `run_agent.py::reset_session_state()` resets token/cost/session counters and calls `context_compressor.on_session_reset()`, which resets `compression_count`.

Key issue:

- Compression-driven session ID rotation is not the same as a user-visible fresh `/new` handoff. It keeps the same logical compressed chain and currently continues accumulating compression degradation.
- The required behavior is stronger: after the second successful compression, perform an automatic handoff that behaves like a fresh session but starts with the latest summarized context injected.

## Proposed behavior

After a successful compression, if the compressor reports that this was the second compression in the current logical chain:

1. Emit an explicit lifecycle/status alert to the user before continuing:
   - Example: `⚠ Context has been compressed twice. I’m automatically starting a fresh session and carrying forward a summarized handoff so accuracy does not degrade.`
2. End the current session with a distinct reason, e.g. `auto_handoff_after_compression`.
3. Create a new session ID and SQLite session row whose `parent_session_id` points to the prior session.
4. Reset per-session counters and reset `context_compressor.compression_count` to zero.
5. Inject the compressed summary into the new session’s in-memory message list as the initial carried-forward context.
6. Update the system prompt for the new session.
7. Notify context engines and memory providers with a new boundary reason that is distinct from ordinary compression:
   - `boundary_reason="auto_handoff_after_compression"`
   - memory switch `reason="auto_handoff_after_compression"`, `reset=False` unless we discover a provider requires `reset=True`.
8. Continue the active user request in the new session without requiring the user to resend anything.

## Implementation plan

### 1. Add config, with safe default enabled

File: `hermes_cli/config.py`

Add a config option under `compression`, for example:

```yaml
compression:
  auto_new_session_after_compressions: 2
```

Use `2` as the default to satisfy the requirement. Allow `0` or `null` to disable only if we want an escape hatch for developers/tests.

Also consider a name that communicates the behavior clearly:

- `compression.auto_handoff_after_count: 2`
- or `compression.max_compressions_per_session: 2`

Preferred: `compression.max_compressions_per_session: 2` because the condition reads naturally: once the current session reaches the max, force a fresh handoff.

### 2. Factor session rotation into helper functions

File: `agent/conversation_compression.py`

Current `compress_context()` contains inline session rotation logic. To avoid duplicating fragile behavior, extract helpers:

- `_rotate_session_for_compression(agent, new_system_prompt) -> old_session_id | None`
- `_rotate_session_for_auto_handoff(agent, new_system_prompt) -> old_session_id | None`
- or a generic `_rotate_session(agent, new_system_prompt, boundary_reason, reset_counters=False)`

The helper should handle:

- old title lookup and title propagation
- `commit_memory_session(messages)` before ending the old session
- `agent._session_db.end_session(old_id, reason)`
- new session ID creation
- `os.environ["HERMES_SESSION_ID"]` update
- gateway `_SESSION_ID` contextvar update
- `agent._session_db.create_session(...)`
- `agent._session_db.update_system_prompt(...)`
- `agent._last_flushed_db_idx = 0`
- context engine `on_session_start(...)`
- memory manager `on_session_switch(...)`

### 3. Add an auto-handoff path after successful compression

File: `agent/conversation_compression.py`

Replace the current repeated-compression warning block with forced handoff logic:

Pseudo-flow:

```python
_cc = agent.context_compressor.compression_count
_limit = getattr(agent, "compression_max_compressions_per_session", 2)
if _limit and _cc >= _limit:
    agent._emit_status("⚠ Context has been compressed twice. Starting a fresh session and carrying forward a summarized handoff...")

    # Important: compressed already contains the newly-generated summary.
    handoff_messages = _normalize_handoff_messages(compressed)

    # End old compressed chain and start a true fresh continuation.
    _auto_handoff_session(agent, handoff_messages, new_system_prompt, old_messages=messages)

    # Reset compression/token counters after handoff, but preserve the handoff messages.
    agent.reset_session_state()
    agent.context_compressor.last_prompt_tokens = estimate_request_tokens_rough(...)
```

Important ordering:

- Do not call `reset_session_state()` before checking `_cc`; it would erase the trigger.
- Do not discard `compressed`; that is the summarized context that must be injected.
- After handoff, reset compression count so the next two compressions in the new session are counted independently.

### 4. Define the injected summarized context format

The existing `compressed` message list should be preserved, but we should make the handoff explicit so the new session has a clean anchor.

Preferred injected shape:

```python
[
  {
    "role": "user",
    "content": "[Automatic session handoff after two context compressions]\nThe prior session was summarized below. Treat this as authoritative carried-forward context, but do not mention it unless useful.\n\n<summary>..."
  }
]
```

However, if `ContextCompressor.compress()` already returns protected head/tail messages plus a summary message, replacing it with a single message might lose recent tail turns. Safer first implementation:

- Keep `compressed` as returned.
- Optionally prepend a short user-role marker explaining that this is an automatic handoff.
- Ensure role alternation remains valid.

Need to inspect actual compressed message shape in `agent/context_compressor.py` before finalizing. Tests should lock this down.

### 5. Make alert visible in CLI and gateway

Files:

- `agent/conversation_compression.py`
- possibly gateway status rendering code if lifecycle statuses are filtered

Use `agent._emit_status(...)` or `agent._emit_warning(...)`, not only `_vprint(...)`, because the current repeated-compression warning is CLI-only. The requirement says alert the user; gateway users must see it too.

The alert should be emitted exactly once per handoff.

### 6. Update session DB lineage semantics

Files likely involved:

- `hermes_state.py`
- `agent/conversation_compression.py`
- tests under `tests/run_agent/`

Use a distinct end reason:

- ordinary compression split: `compression`
- forced handoff: `auto_handoff_after_compression`

This preserves auditability in session history and lets future UI/session search show why the boundary happened.

Open question:

- Should the auto-handoff create one new session after the existing compression-created continuation session, or should it replace the ordinary compression rotation when `_cc >= 2`?

Preferred:

- On the second compression, use the handoff boundary instead of ordinary `compression` rotation, not an extra third session row. That avoids confusing lineage and duplicate title numbering.

### 7. Tests

Add/modify tests in:

- `tests/run_agent/test_compression_boundary.py`
- `tests/run_agent/test_compression_persistence.py`
- possibly new `tests/run_agent/test_auto_handoff_after_compression.py`

Test cases:

1. First successful compression:
   - compresses normally
   - rotates session with reason `compression`
   - does not reset `compression_count` to zero if existing behavior depends on counting logical compression chain
   - does not emit auto-handoff alert

2. Second successful compression:
   - emits user-visible lifecycle/warning status
   - creates a new session row with parent pointing to the previous session
   - uses end reason `auto_handoff_after_compression`
   - resets `context_compressor.compression_count` to 0 after handoff
   - keeps summarized context in returned messages
   - updates `HERMES_SESSION_ID`
   - updates system prompt in DB

3. Gateway visibility:
   - status callback receives the handoff alert

4. Compression abort path:
   - if compression aborts and returns unchanged messages, no handoff occurs even if count was near threshold

5. Config override:
   - threshold defaults to 2
   - disabled/changed value behaves correctly if we add an escape hatch

6. Plugin context engine compatibility:
   - `on_session_start()` receives `boundary_reason="auto_handoff_after_compression"`
   - existing `boundary_reason="compression"` tests still pass for normal compression

### 8. Docs

Update:

- `website/docs/developer-guide/context-compression-and-caching.md`
- possibly user-facing compression docs/config docs
- `hermes-agent` skill if it documents compression behavior/config keys

Document:

- why Hermes auto-hands off after two compressions
- alert text users should expect
- config key, if configurable
- how lineage appears in session history

## Risks and mitigations

1. Role alternation breakage
   - Mitigation: tests validate returned message sequence after prepending any handoff marker.

2. Losing recent tail context
   - Mitigation: preserve `compressed` output rather than replacing it with a single summary unless tests prove the summary contains all tail context.

3. Double session rotation on second compression
   - Mitigation: refactor rotation so second compression chooses `compression` OR `auto_handoff_after_compression`, not both.

4. Plugin context engines may treat the new boundary incorrectly
   - Mitigation: pass distinct boundary reason and add fallback to old compression behavior if a plugin rejects unknown kwargs.

5. Memory providers may interpret reset incorrectly
   - Mitigation: use `reset=False` initially because the logical task continues; only session counters/compression count reset.

## Verification commands

From repo root:

```bash
python -m pytest tests/run_agent/test_compression_boundary.py -q -o 'addopts='
python -m pytest tests/run_agent/test_compression_persistence.py -q -o 'addopts='
python -m pytest tests/run_agent/test_auto_handoff_after_compression.py -q -o 'addopts='
```

Then run a focused manual smoke test by lowering the compression threshold in a temp `HERMES_HOME` and sending enough turns to trigger two compressions. Verify the user sees the auto-handoff alert and `/status` or session DB lineage shows a fresh post-handoff session.

## Summary

The current code already detects `compression_count >= 2`, but only prints a CLI warning suggesting `/new`. The implementation should replace that warning with a forced, user-visible auto-handoff path that starts a fresh session, resets compression counters, and carries forward the just-created compressed context as the new session’s starting context.
