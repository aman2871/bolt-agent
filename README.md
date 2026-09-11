# Bolt

A voice- and text-driven multi-agent system. One manager agent decomposes a
request and delegates it to named specialists; every code change is executed in
a sandbox and reviewed before you see it.

Everything runs on Ollama's OpenAI-compatible API. Speech never leaves the machine.

## Quick start

```bash
.venv/bin/bolt agents                    # the roster
.venv/bin/bolt ask "..."                 # ask the manager (it delegates)
.venv/bin/bolt ask -a scout "..."        # address one specialist
.venv/bin/bolt code "fix X" --test "python3 -m pytest -q"
.venv/bin/bolt serve                     # dashboard on http://127.0.0.1:8080
.venv/bin/bolt listen                    # voice loop
.venv/bin/bolt eval baseline             # score the agents
.venv/bin/bolt tasks / bolt logs <id>    # history and full transcripts
```

## Running it: Docker

```bash
docker compose up -d                 # api + redis, on http://127.0.0.1:8090
docker compose logs -f api
scripts/install-autostart.sh         # bring it up at every login
```

Port 8090, not 8080 — 8080 on this machine is already taken by the `iiitd-web`
nginx container.

Two things stay on the host by design. **Ollama**, because it holds your cloud
credentials and, for local models, needs the GPU — containers reach it at
`host.docker.internal`. And **voice**, because a Linux container on macOS has no
microphone or speaker, so `bolt listen` is a host-only command.

Mounting `/var/run/docker.sock` lets the code gate use the host daemon, so
running in Docker upgrades the sandbox from the subprocess tier to the real one
(no network, capped memory) automatically — `/api/health` reports which tier is
live.

Bus workers are off by default:

```bash
BOLT_BUS=redis docker compose --profile workers up -d
```

### Surviving a reboot

`restart: unless-stopped` only helps while the Docker daemon is running, and on
macOS that daemon is a desktop app that may not be up at login. So
`scripts/install-autostart.sh` installs a LaunchAgent that waits for the daemon
(launching Docker Desktop if needed), then brings the stack up, and retries
every 10 minutes in case login raced it.

```bash
launchctl print gui/$(id -u)/com.bolt.stack     # status
tail -f ~/Library/Logs/bolt-autostart.log       # what it did
scripts/install-autostart.sh --uninstall        # remove it
```

### Surviving a network change

Moving between networks — office wifi to home wifi, sleep/wake, VPN up or down —
leaves pooled sockets half-open: the peer is unreachable but the OS keeps the
socket in ESTABLISHED, so a read blocks until TCP keepalive gives up, which by
default takes hours. A process that looks alive but answers nothing is almost
always this.

Bolt guards against it in three places:

- **HTTP** — pooled connections expire after 30s idle (`BOLT_LLM_KEEPALIVE`), and
  any transport-level failure drops the whole pool before retrying, so the retry
  cannot inherit the dead socket.
- **Redis** — `socket_keepalive`, a 5s connect timeout and a 30s
  `health_check_interval`, which pings an idle pooled connection rather than
  discovering it is dead mid-read.
- **Containers** — a `HEALTHCHECK` plus `restart: unless-stopped`, so a wedged
  process gets replaced rather than sitting there.

## Models: automatic and manual

Each agent's model is resolved per task, by three levels of authority:

1. **A pin** — set in the dashboard or `bolt pin <agent> <model>`.
2. **An explicit model** in `agents.yaml`.
3. **The router**, when the agent's model is `auto` (the default for most).

The router is rule-based, not another model call: it runs on every task, so it
has to be free and predictable. A task needing vision is restricted to models
that actually accept images; the rest are scored on tier fit (how much model the
prompt warrants), how many of the agent's `prefers` strengths the model has, and
cost. A pin that cannot see images is overridden for image tasks rather than
failing silently.

```bash
bolt models              # catalogue, capabilities, who uses what
bolt models --probe      # send a real image to every model and record the answer
bolt agents              # what each agent resolves to, and why
bolt pin forge kimi-k3:cloud
bolt pin forge auto      # release
```

### Capabilities are measured, not declared

`models.yaml` only declares defaults. `bolt models --probe` runs a real request
per capability and stores what actually happened, and observation always beats
the config. The probes are small and mechanically checkable — a probe nobody can
check is not worth running:

| probe | what it checks |
|---|---|
| `speed` | measured output tokens/sec |
| `tools` | emits a tool call with the right name and arguments |
| `structured_output` | returns strict, parseable JSON in the requested shape |
| `instruction_following` | obeys an exact output format |
| `reasoning` / `reasoning_hard` | a lowest-common-multiple problem; a constraint puzzle with one brute-force-verified answer |
| `code` / `code_hard` | generated code is executed against hidden tests, including edge cases |
| `long_context` | finds one planted fact in a ~27k-token document |
| `vision` | reads a specific code out of an image, not merely accepting the image |

**What the probes actually found here:** all six models passed every probe except
vision. Only two capabilities separate them at all:

| model | tok/s | $/Mtok | ctx | vision |
|---|---|---|---|---|
| `deepseek-v4-flash` | 92 | 0.20 | 128k | no |
| `glm-5.2` | 91 | 0.60 | 128k | no |
| `deepseek-v4-pro` | 68 | 0.80 | 128k | no |
| `kimi-k2.7-code` | 57 | 1.00 | 256k | **yes** |
| `kimi-k3` | 53 | 1.00 | 256k | **yes** |
| `nemotron-3-super` | 38 | 0.40 | 128k | no |

That result shaped the router: the `strengths` in `models.yaml` are unverified
claims, so they only break ties, while probed capabilities are hard filters. It
also means measured speed is kept deliberately separate from `tier` — being slow
is not evidence of being clever, and nothing here suggests the slower models are
better. Treat the uniform results as a floor, not a ranking: these probes are
small, and a harder real workload may still separate the models.

## Iris: images and video

```bash
bolt look media/screenshot.png "what is the error count?"
bolt look media/clip.mp4 "what changes over the clip?"
bolt ask -a iris "check every screenshot in media/ for error dialogs"
```

No model reachable from Ollama accepts video — a `video_url` block is rejected
outright. So `watch_video` reduces a clip to something a vision model can read,
as a timeline rather than a pile of stills:

- **24 frames** by default, sampled across the clip (scene changes preferred,
  even spacing as a fallback for footage with no hard cuts). Measured: 32 frames
  in one message costs ~7,000 prompt tokens and the model counts them back
  correctly. Eight — the old default — is one frame per fifteen seconds on a
  two-minute video, which is why it used to describe a video instead of
  following it.
- **Every frame carries its timestamp**, and the speech from that stretch is
  attached to it, so the model reads "Frame 7 — 0:42, heard around here: …"
  rather than guessing which words belong to which picture.
- **Frames are chosen by what happens in them.** A cheap decimated scan (raw
  greyscale piped from ffmpeg, no model, ~0.2s for five minutes) measures how
  much each frame differs from the last, and the busiest moments are picked.
  Selection is relative to the footage's own baseline, because movement in a
  dark CCTV frame is noise in a bright screen recording. Any unused budget
  covers the quiet stretches, so a static video is still described.
- **Long videos are read in windows** of two minutes, 16 frames each, and the
  per-window observations are combined. One pass cannot cover an hour at any
  frame budget.

Why it matters: on a five-minute clip containing three *two-second* events,
clock sampling landed inside none of them and reported "exactly one thing
happens in the entire video". Activity sampling found all three by name, with
their start and end times. `BOLT_MOTION_SAMPLING=0` returns to clock sampling.

Accuracy is bounded by how densely the scan runs and how many frames each window
gets — not by the timestamps, which are exact. `BOLT_SCAN_FPS_MIN` sets the
shortest event that can be noticed at all.

## Chat

The dashboard opens on a conversation view: a thread of messages, a sidebar of
past conversations, and a composer with a microphone.

Turns are stored, and the previous ones are replayed to the model, so follow-ups
work — "how many agents are there?" then "which of them can see images?" resolves
without repeating yourself. Spoken and typed turns share one thread, so you can
start by voice and follow up by typing.

`conversations` and `turns` are separate from the raw `messages` table: the
former is what a person reads back, the latter is the full model transcript
including tool calls, which the tasks view still shows.

While you hold the mic, a nebula animation drives off real microphone amplitude
(a Web Audio `AnalyserNode`), so it breathes with your voice rather than
animating blindly. It also lights up, dimmer, when the daemon is listening or
speaking.

### Streaming

Replies stream as they are generated. Only the agent you are talking to streams —
delegated agents run at depth > 0 and their text interleaved into the reply would
be unreadable, so their progress shows as tool events instead. Their thinking is
surfaced separately: these models emit `reasoning` alongside `content`, and the
former drives the progress line rather than the answer.

Streamed responses report no usage unless `stream_options.include_usage` is set,
which would have silently zeroed every token and cost figure — so it is set.

### Proposed changes

In chat, an agent writing a file **proposes** the change instead of making it.
The reply is followed by a diff per file with **apply** and **discard**:

```
bolt/util/budget.py   modified  +5 -0     [apply] [discard]
```

Nothing touches disk until you apply. The agent is told the write is pending, so
it does not claim to have finished. This is the counterpart to the code gate,
which protects `bolt code` but never covered chat — where writes previously went
straight to the real workspace.

Inside the gate's snapshot, writes still happen directly: that copy is already
isolated from anything real.

### Why a long search used to blow the budget

Every step of an agent resends the whole conversation, so a tool-heavy run costs
roughly the square of its context. One search that ended at a 27k-token context
had billed 212k tokens across twelve steps — and its last six steps each cost
~25k while producing under 150 tokens of output.

Old tool results are the bulk of that, and are rarely needed verbatim once the
model has acted on them. So above `BOLT_COMPACT_ABOVE_CHARS` (24k), tool results
older than the last `BOLT_KEEP_TOOL_RESULTS` (3) are trimmed to a few hundred
characters. System, user and assistant turns are never touched — losing those
changes what is being asked.

The same question that failed at 212k now answers in 4.6k. A deliberately broad
one ("compare OTP handling across every project") triggered compaction eleven
times and saved 1.6 million characters in a single run.

The budget counts **tokens billed**, not context size, and the error now says so
rather than just reporting a number.

### Undo

Applying a change is not a one-way door. The previous contents are kept, so any
change Bolt wrote can be put back:

```bash
bolt checkpoints        # what has been written and can still be undone
bolt undo               # the most recent change
bolt undo p_a1b2c3      # a specific one
```

In the dashboard the **apply** button becomes **undo** once a change lands.
Changes from the code gate are recorded the same way, so `bolt code --apply` is
undoable too.

Undo refuses when the file has been edited since Bolt wrote it — restoring the
old contents would silently discard that work. `bolt checkpoints` marks those
rows `edited since`, and `--force` (or confirming in the dashboard) overrides.

The honest limit: this covers changes made through `write_file` and the gate. An
agent that edits a file through `run_shell` — `sed -i` and the like — is not
recorded, and cannot be undone this way.

### Replying, copying and rating

Every reply carries four controls on hover:

- **copy** the reply to the clipboard
- **👍 / 👎** rate it
- **↩ reply** to that specific message

Reply quotes the message you picked into the prompt, so "explain that in simpler
terms" means the message you clicked rather than whatever happened to be last.
Verified: replying to an older message about `bus.py` answered about `bus.py`,
not about the `router.py` turn in between.

A thumbs-down does not just record a complaint — it **reworks the answer**. The
agent is given the original question and every rejected attempt (with whatever
you said was wrong) and told to work out what was actually being asked, rather
than restate the same thing differently. The new attempt is appended and labelled
`attempt 2`, so you can compare them.

After three rejected attempts it stops guessing and asks you to explain what you
are after, because a model that has missed three times is usually missing context
rather than effort. `BOLT_MAX_RETRIES` changes the limit.

The note you leave is also kept as a **correction** and folded into that agent's
prompt on later turns:

```
Corrections from earlier feedback, apply them:
- Always name the exact file path and line numbers you used.
```

Be clear about what this is: prompt-level guidance, not training. Nothing about
the model changes — later turns simply carry your correction in their
instructions. The five most recent per agent are used, so a stale one eventually
falls out, and you can drop them all:

```bash
curl localhost:8090/api/feedback                          # ratings and corrections
curl -X DELETE localhost:8090/api/feedback/lessons/scout  # forget scout's corrections
```

### Stopping and editing

The send button becomes a stop button while a reply is being generated. Stopping
tells the server first and aborts the request second: aborting only the browser
request would leave the agent running, still spending tokens and still calling
tools. The turn is kept and marked `(stopped)` so you can edit or retry it.
`bolt hush` and the ✋ button stop a reply in progress as well as any speech.

Hovering your most recent message shows a ✎. It rewinds the conversation —
dropping that turn and everything after it, on the server as well as on screen —
and puts the text back in the composer to change and resend. Only the latest
message is editable: rewinding further would discard replies that later turns
were built on.

### Appearance

The gear in the header opens **Appearance**: a mode (System / Light / Dark),
**sixteen themes**, and **37 chat backgrounds** — or any image from your machine.
Choices are remembered per browser.

Palettes are derived from a few seeds rather than written out: sixteen themes,
two modes and eleven tokens each is far too many values to keep in tune by hand.

Backgrounds come in two kinds.

**37 generated ones** — washes, patterns and tinted geometry, written as CSS
rather than shipped as files. They take their colour from the active theme, so
they re-tint when you switch rather than clashing.

**42 photographs** — earth and sky, architecture, code and desks, quiet life —
fetched once and served locally:

```bash
bolt wallpapers --fetch    # ~17MB, one time
bolt wallpapers            # what you have, and where each came from
```

They are downloaded rather than hotlinked. Hotlinking means a request to a third
party every time the dashboard loads, which tells them your IP and breaks when
you are offline — the opposite of the point of running this on your own machine.

Source is Unsplash, whose licence allows free use, commercial or not, without
permission or attribution. Attribution is appreciated though, so
`backgrounds/manifest.json` records the photographer's page and the licence for
every image, and `bolt wallpapers` prints them.

A photograph has bright regions wherever it likes, so a scrim sits over it and
the welcome copy sits on a frosted panel. A glow in the background colour was the
first attempt: it works on a dark theme and actively harms a light one, where a
wide near-white blur bleeds over dark glyphs and greys them out. A panel behaves
the same way in both modes and lets the ordinary text tokens do their job.

A custom image is downscaled to 1600px and stored in the browser; a photograph
straight off disk is several megabytes, which will not fit in local storage and
is wasted on a background. A scrim sits over it so message text stays legible.

Themes are applied as CSS custom properties from JavaScript rather than as CSS
rules. Every palette needs a light set and a dark set, and mode can follow the
system, so pure CSS would mean three near-identical rules per palette to say the
same thing. `color-scheme` is set alongside, so native controls — scrollbars, the
agent dropdown — follow the theme instead of the OS.

The waveform takes its hues from the palette too, with a darker line on light
grounds: one lightness cannot serve both. Following the system means following
it as it changes, so a `prefers-color-scheme` listener re-applies on the fly.

### The listening modal

Pressing the mic opens a small dialog — a waveform reacting to your voice, and
Cancel / Done (Escape / Enter). It is not only decoration: the dimmed backdrop
means the composer cannot be reached while it is open, so a half-typed message
cannot collide with what you are saying.

The waveform is drawn from the microphone's **actual samples**, not just its
loudness — so the shape of what you say is visible, which reads as listening
rather than as a meter. Three layered lines give it depth, and each is tapered
towards the ends so it never meets the canvas edge.

It rises quickly and falls slowly, which is what makes it feel like a beat
rather than a swell. With no audio it drifts on a gentle sine, so the idle
state still looks alive.

### Handing Bolt a file

Drag a video, image or file onto the chat — or paste a screenshot, or use the 📎
button. It uploads, appears as a chip above the composer, and the path is
included with your message.

Uploads are saved into `uploads/<conversation>/` inside the workspace rather than
held in memory, because every tool addresses files by path: once saved,
`watch_video`, `look_at_image` and `read_file` work on it unchanged.

Filenames are sanitised to a bare name — `../../etc/passwd` becomes `passwd` —
and the write is re-checked against the workspace guard afterwards. Types are
limited to media, text and code; size to `BOLT_MAX_UPLOAD_MB` (500 by default).

`uploads/` is gitignored, and nothing prunes it — worth a sweep if you drop a lot
of footage.

### Folders of timestamped images

Cameras rarely hand you a video — they drop one image a minute into a folder.
`watch_image_sequence` reads such a folder as a timeline, taking each image's
moment from its filename (`IPC-CAM01_20240516-0912.jpg`) and falling back to the
file's modification time. Images are spread evenly across the whole sequence,
because the end of a sequence is usually the point.

```
"how many buses are in the bay over time?"
 → 0 from 09:00, 2 from 09:12, 5 from 09:27 — two changes, with the times
```

`region="x1,y1,x2,y2"` crops both this and `watch_video` to part of the frame.
That matters more than it sounds: a timestamp banner changes every second and
swamps the change measure, so the interesting part never gets picked.

## The dashboard

`bolt serve` puts everything on http://127.0.0.1:8080. Four views:

- **overview** — ask box, live trace of every agent step and tool call as it
  happens, and a per-agent table (running / worker / idle, calls, tokens, p50,
  p95, retries, success rate, estimated cost).
- **tasks** — every request, click one for its delegation tree, the full
  transcript, and any diff it produced. Tree nodes are clickable, so you can
  drill from a manager request into what each specialist actually did.
- **code gate** — run a change and watch snapshot → tests → review → retry,
  with a per-attempt table and the option to apply an approved diff.
- **models** — the catalogue with probed capabilities, who currently uses what,
  and a button to re-probe vision.
- **evals** — the scoreboard by agent, model and run label, plus every
  individual eval run and why it failed.

Each agent row on the overview carries a model dropdown: leave it on `auto —
<model>` to let the router choose per task, or pick a model to pin it.

The header strip is live system state: websocket, Redis, which sandbox tier is
actually in use, and how many workers are alive.

## How it fits together

```
 voice ──┐                          ┌── scout      research
 text  ──┼─► gateway ─► MANAGER ────┼── forge      code
 (CLI,   │   (adapters)    │        ├── sentinel   review
  HTTP)  │                 │        ├── archivist  notes
 voice ◄─┘          Redis Streams   └── ops        shell
                           │
                  SQLite: tasks, runs, messages, artifacts, evals
                           │
                  Sandbox: snapshot + tests, Docker when available
```

Agents never call each other directly. An agent that needs help calls the
`delegate` tool, and the manager decides — that single choke point is what
enforces the depth limit, the roster permissions and the token budget.

## How voice works

`bolt listen` runs on the host only — a Linux container on macOS has no
microphone or speaker.

1. **Capture** (`voice/audio.py`) — 16 kHz mono in 30 ms frames. The room's noise
   floor is measured for 0.6 s at startup; speech starts above 3x that floor and
   ends after 800 ms below 1.8x. Under 300 ms is discarded as a cough.
   (`webrtcvad` is unusable on modern setuptools — it imports the removed
   `pkg_resources` — so this is a plain energy VAD.)
2. **Transcribe** (`voice/stt.py`) — whisper.cpp with `ggml-base.en`, ~275 ms for
   a short utterance.
3. **Route** (`voice/wake.py`) — see below.
4. **Run** — the resolved agent goes through the normal loop, so voice requests
   delegate, use tools and route models exactly like typed ones.
5. **Speak** (`voice/tts.py`) — markdown, links and code fences are stripped,
   then Kokoro synthesises sentence by sentence and plays each as it is made, so
   speech starts before the whole reply exists (~1.1 s to first sentence).
6. **Barge-in** — a watcher thread listens during playback at a higher threshold
   (4x floor, so the assistant does not interrupt itself) and stops playback
   after ~90 ms of speech, then drains the mic so its own tail is not heard as
   your next utterance.

Terminal needs microphone access under System Settings → Privacy & Security →
Microphone.

## Talking to it

Three ways, all using the same local whisper model — recorded speech never
leaves the machine.

**The dashboard mic button.** Click *speak*, talk, click *stop*. The browser
records; the server transcribes and routes it. *Speak replies* plays the answer
back through Kokoro. Works in the container too — `whisper-cli` is built into
the image for exactly this.

**A one-off from the terminal:**

```bash
bolt listen            # runs until you stop it
```

**Listening only while the dashboard is open.** This is the default: the voice
daemon runs from login but holds no microphone until a browser tab is watching,
and releases it the moment the last one closes — the system microphone indicator
goes out with it. Ignoring the audio is not enough; an open input stream is a
live capture, so the stream is closed and PortAudio is terminated outright. The tab proves it is alive with a heartbeat every
10s, so a browser that dies without closing its socket goes stale within 30s
rather than leaving the microphone on.

Two processes share one room, so they coordinate through a small state file
(`~/.bolt/voice-control.json`, mounted into the container):

- Pressing the dashboard mic **pauses the daemon**, otherwise both pipelines hear
  the same sentence and answer it twice, talking over each other.
- **stop talking** in the dashboard, `bolt hush`, or closing the tab stops
  whatever is speaking — including the daemon, which speaks through the laptop
  and cannot hear the browser.
- Talking over a reply stops it and **keeps what you said**, so you can interrupt
  and give the next command in one breath.

```bash
bolt voice      # what the voice system is doing right now
bolt hush       # stop talking, wherever it is talking from
bolt listen --always   # ignore the tab gate and listen permanently
```

**Always listening, from login:**

```bash
scripts/install-autostart.sh --listen
tail -f ~/Library/Logs/bolt-listen.log
scripts/install-autostart.sh --listen --uninstall   # just the mic
```

That installs `com.bolt.listen`, which keeps the voice loop running and restarts
it when audio devices come and go (headphones connecting, sleep/wake). It only
acts on speech addressed to it, so leaving it on is not the same as recording
everything you say — unaddressed utterances are transcribed locally, matched
against the wake names, and dropped.

macOS will ask for microphone permission the first time. If no prompt appears,
run `bolt listen` once in Terminal to trigger it, or add the app by hand under
System Settings → Privacy & Security → Microphone.

## Addressing agents by voice

Say the system name, an agent name, or both:

- "Hey Bolt, what changed in the config?" → manager
- "Forge, fix the parser in budget.py" → forge
- "Bolt, Scout, find the bus module" → scout
- "Bolt bro, let's work on the bus module" → manager (filler after the name is dropped)

Unaddressed speech is ignored, so the loop can stay on. Whisper's usual
manglings ("Hayhive", "Sentinal", "Ford") are matched fuzzily.

Set `BOLT_REQUIRE_ADDRESS=0` to answer everything it hears.

## The code gate

`bolt code` never edits your workspace directly:

1. The workspace is copied to `.worktrees/<task_id>` with a hash manifest.
2. Forge works only inside that copy.
3. The test command runs in the sandbox — Docker with no network and capped
   memory when the daemon is up, otherwise a plain subprocess (the result says
   which, so the two are never confused).
4. Sentinel reviews the diff together with the test output.
5. Tests failing or review rejecting sends it back to Forge with the evidence,
   up to `BOLT_GATE_ATTEMPTS` times.
6. Only then is it offered, and only `--apply` copies it back.

The manifest is what makes step 6 safe: only files the agent actually changed
are copied back, so edits made in your workspace meanwhile are never clobbered.

## Working across other projects

Agents can reach directories beyond the workspace. In Docker these are bind
mounts, which share the host's files rather than copying them — nothing is
duplicated, and edits land on both sides immediately:

```yaml
environment:
  BOLT_EXTRA_ROOTS: /projects/Chartr
volumes:
  - ${CHARTR_DIR:-${HOME}/Projects/Chartr}:/projects/Chartr
```

On the host, set the same variable — the LaunchAgent installer picks up a sibling
`~/Projects/Chartr` automatically:

```bash
BOLT_EXTRA_ROOTS=~/Projects/Chartr bolt ask "what framework does chartr-website use?"
```

A symlink does not work for the container: a symlink stores a path, and a host
path like `/Users/you/Projects/Chartr` does not exist inside the container, so it
dangles. Bind mounts are the equivalent that survives the filesystem boundary.

Agents discover what they can reach with `list_roots`. Relative paths resolve
against the workspace; anything in another root needs an absolute path.

### AWS

The container has AWS CLI v2 and boto3, with `~/.aws` mounted read-only, so
agents can query real infrastructure:

```bash
bolt ask -a ops "how many S3 buckets does the chartr profile have?"
```

Access is **read-only**. The shell tool allows verbs that only observe —
`describe-*`, `list-*`, `get-*`, `s3 ls`, `sts get-caller-identity` — and refuses
everything else, so an agent acting on a misread instruction cannot change real
infrastructure:

```
ERROR: 'aws s3 rb' is not a read-only command, so it is refused.
```

Set `BOLT_AWS_WRITE=1` to lift that. `aws sts get-session-token` stays blocked
either way, since it mints credentials that would outlive the guard.

There is no `[default]` profile with keys in this setup, so `AWS_PROFILE` is left
unset and agents pass `--profile` explicitly; `aws configure list-profiles` shows
what is available. Profiles carrying an `aws_session_token` expire, and agents
will report an auth error rather than anything subtler when they do.

The file tools refuse `~/.aws` outright — only the `aws` CLI reads it.

### Reusing code across projects

Grep does not scale to 95 repositories — an agent either misses the file or
burns thousands of tokens crawling directories. So everything reachable is
indexed:

```bash
bolt index --rebuild     # 19,690 files, 59,974 symbols, 96 projects in ~8s
bolt index               # what is indexed
bolt find "razorpay signature verify"
```

Agents get three tools from it — `search_code`, `find_symbol`, `list_projects` —
and Forge and Scout are told to look before writing anything non-trivial, and to
say which project they took an approach from.

```
"Have we built SMS sending anywhere already?"
 → chartr-messaging-api: four providers behind one MessageApi interface,
   chosen by SMS_API_PROVIDER, plus a MessagingServiceWrapper you can copy.
```

SQLite FTS5, not embeddings: it needs no model, indexes in seconds, and ranks
exact identifiers — which is most of what code search is. FTS5 requires *every*
term by default, which returns nothing for a phrase like "jwt decode token
middleware", so the search tries all terms first for precision and any term
second for recall.

Results carry **absolute** paths. Shortened ones read better but do not resolve —
relative paths resolve against the workspace, and these files live under other
roots. Host and container keep separate indexes for the same reason: they see the
same files at different paths.

The index skips credential files, `node_modules`, build output and minified
bundles, and never leaves your machine.

### Credentials are refused by name

Agents send file contents to a cloud model, and a real project tree carries a lot
that nobody meant to share — the Chartr mount alone has 417 `.env` files and 265
key files. So `read_file`, `grep`, `write_file` and `run_shell` all refuse paths
matching `BOLT_SECRET_PATTERNS` (`.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa*`,
`*credentials*.json`, …). `*.sample`, `*.example` and `*.template` are exempt,
since templates are usually the useful thing to read.

This matches names, not contents — it is a guard, not a scanner. Set
`BOLT_ALLOW_SECRETS=1` to turn it off.

## Configuration

Agents live in `bolt/config/agents.yaml` — model, prompt, tools, budgets and
delegation permissions. Adding an agent is a config edit, not new code.

| Variable | Default | Purpose |
|---|---|---|
| `BOLT_OLLAMA_URL` | `http://localhost:11434/v1` | Ollama endpoint |
| `BOLT_LLM_CONCURRENCY` | `4` | In-flight requests, capped for cloud rate limits |
| `BOLT_BUS` | `inproc` | `redis` routes delegation through the bus |
| `BOLT_MAX_DEPTH` | `3` | Delegation depth limit |
| `BOLT_GATE_ATTEMPTS` | `3` | Forge tries before the gate gives up |
| `BOLT_WORKSPACE` | this repo | What agents may read and write |
| `BOLT_REQUIRE_ADDRESS` | `1` | Ignore speech not addressed to Bolt |
| `BOLT_VIDEO_FRAMES` | `8` | Frames sampled per video |
| `BOLT_SCENE_THRESHOLD` | `0.3` | ffmpeg scene-change sensitivity |

## Running on the bus

In-process delegation is the default and needs nothing running. For durable,
restartable, multi-machine delegation:

```bash
brew services start redis
bolt worker scout &   # one per agent you want available
BOLT_BUS=redis bolt ask "..."
```

Un-acked work is reclaimed by another worker after `BOLT_RECLAIM_IDLE_MS`, so a
worker that dies mid-task does not lose it.

## Requirements

- Ollama with the cloud models in `agents.yaml`
- Redis, only for `BOLT_BUS=redis`
- Docker, only for the stronger sandbox tier
- Voice: `brew install whisper-cpp`, plus `models/ggml-base.en.bin`,
  `models/kokoro-v1.0.onnx` and `models/voices-v1.0.bin`

The microphone needs Terminal to be granted mic access in
System Settings → Privacy & Security → Microphone.
