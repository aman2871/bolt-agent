# Bolt — what to fix, in order

Written 2026-09-01, after an audit that found five latent bugs by probing rather
than waiting for them. Ordered by what blocks the next thing, not by size.

## 0. Before anything else

- [ ] **Commit the code.** 0 tracked files, ~11 uncommitted paths. Everything
      built so far exists only as working-tree files on one laptop.
- [ ] **Tests around the access layer, the agent loop and proposals.** The
      current 8 tests cover a toy `budget.py` and nothing else. Every bug found
      in the audit — shared SQLite connection, unguarded `apply`, orphaned
      tasks — would have been caught by tests that take an hour to write.
      This is the prerequisite for changing permissions safely.

## 1. Permission model (the thing that makes it shareable)

Today access is static: `BOLT_EXTRA_ROOTS` is set once and agents see everything
under it. For anyone but the author, that is "I gave this thing my home folder".

- [ ] Ask per **project**, once — not per file. Per-file prompts train people to
      click allow.
- [ ] Persist grants with scope and expiry; `workspace.roots()` and `resolve()`
      are already the single choke point, so this is a store plus a prompt.
- [ ] Show what was done with the access afterwards. `bolt checkpoints` is
      already half of this.
- [ ] Read-only vs read-write as separate grants.

## 2. Product readiness

- [ ] **One-command install.** Today: a venv, a compose file, two LaunchAgents,
      a 141 MB whisper model and a 310 MB Kokoro model. Needs to be `brew
      install bolt` or a signed `.dmg`, with model downloads asked for, not
      silent.
- [ ] **The index must be invisible.** Nobody will run `bolt index --rebuild`.
      Build on first grant, refresh incrementally on file change.
- [ ] **A real migration system.** `_migrate()` is one `ALTER TABLE`. Schema
      changes will run unattended on other people's data.
- [ ] **`bolt diagnose`.** Local-first means the evidence lives on their laptop.
      Package logs, DB stats and versions into something sendable.
- [ ] Decide the shape: local-first, bring-your-own-Ollama. That removes
      multi-tenancy, credential custody and inference margin as problems —
      they were only problems for a hosted product.

## 3. Correctness still outstanding

- [ ] **Chat proposals bypass the gate.** `bolt code` runs snapshot → tests →
      review. A chat proposal shows a diff and applies it, with neither. These
      two paths should converge.
- [ ] **`run_shell` writes are invisible.** Undo covers `write_file` and gate
      applies. An agent using `sed -i` is neither proposed nor undoable.
- [ ] **The index goes stale** the moment you edit anything.
- [ ] **`bolt ask` does not stream** — only the dashboard does.
- [ ] Corrections (thumbs-down notes) are API-only, with no way to review or
      clear them from the dashboard.

## 4. Video

No model reachable from Ollama accepts video: `video_url` is rejected outright
by both vision models. Frames plus an audio transcript is not a workaround, it
*is* the approach — but the current version is far weaker than it needs to be.

- [x] **Raise 8 frames to 24–32.** Done — 24 default, 16 per window. Measured: 32 frames costs ~7,000 prompt
      tokens (~218 each) and the model counts them back correctly. Eight frames
      over a two-minute video is one frame per 15 seconds, which is why it
      describes rather than follows what happens.
- [x] **Timestamp every frame** ("frame 7 — 00:42") so the model can place
      events in time instead of guessing from order.
- [x] **Use whisper's segments, not one blob**, so speech is aligned to the
      frames it accompanies.
- [x] **Chunk long videos** into windows and summarise per window; a single
      pass cannot cover an hour.
- [ ] Optional: scene-change detection already exists — surface *why* a frame
      was chosen so the model knows it is a cut, not a sample.

## 5. Accepted limits

Worth stating rather than quietly carrying:

- Capability probes are a floor, not a ranking: results vary run to run, and one
  probe is weak evidence.
- Corrections are prompt-level guidance, not training.
- Stopping a reply cancels between steps, so a slow tool call finishes first.
- Retries cost a full task each; the three-strike cap is what bounds it.
