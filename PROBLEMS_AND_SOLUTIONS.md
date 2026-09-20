# Problems and Solutions

Real problems hit while building this project, in chronological order by milestone.
Git history here is coarse (the whole project landed in three squashed commits, so
there's no commit-by-commit trail of individual bugs) — these entries are reconstructed
from commit messages, code comments, test fixtures written specifically to prevent
regressions, and direct recollection of what happened.

---

## 1. Duplicate sources in the Writer's final report (Milestone 7)

**Problem.** After wiring the full graph together and running the system on a real
question ("What are the effects of social media on mental health?"), the generated
report's "Sources" section at the bottom listed the same URL two or three times
instead of once each.

**Root cause.** This was *not* a bug in the Researcher. The same source URL
legitimately appearing multiple times in `state.findings` is correct behavior — the
Researcher records one finding entry per sub-question it supports, so if one article
was genuinely relevant to two different sub-questions, that URL correctly appears
twice in `findings`. The actual bug was narrower: the Writer's prompt just handed the
LLM the raw findings and asked it to write a report including a Sources list, trusting
the LLM to deduplicate URLs on its own while compiling that list from the findings —
which it did not reliably do.

**Diagnosis.** Manual review of the generated report — reading the output revealed
the same link listed multiple times in what should have been a clean, unique
reference list.

**Fix.** `writer.py`'s `_build_report_prompt()` computes a deduplicated,
order-preserving list of source URLs in Python via `dict.fromkeys(finding.source_url
for finding in state.findings)` *before* building the prompt, and passes that as an
explicit, authoritative "use exactly this list" instruction
(`WRITER_SYSTEM_PROMPT`: *"use exactly that list for the Sources section, rather than
compiling it yourself from the findings, since the same source may appear against
multiple findings"*) — rather than trusting the LLM to deduplicate from raw findings,
which proved unreliable. `test_generate_report_deduplicates_sources_in_prompt` locks
this in.

**Prevention.** Deduplication (or any other structural guarantee about output shape)
should be computed in code and handed to the LLM as authoritative input, never left as
an implicit instruction for the LLM to get right on its own when a prompt says "compile
a list from X."

---

## 2. OpenRouter rate-limited during normal development (pre–Milestone 9)

**Problem.** During ordinary development/testing (not a formal load test), the
OpenRouter model in use got rate-limited mid-session, with no fallback path — the app
had no way to make progress except waiting out the rate limit.

**Root cause.** `LLMClient` had exactly one provider wired in. Any rate limit on that
provider was a hard stop for the whole app, whether it happened during a two-minute
dev session or a real user's request.

**Diagnosis.** Direct observation — a request failed with a 429 from OpenRouter during
normal use.

**Fix.** This is what motivated building the Groq fallback into `llm_client.py` at
all: on a `RateLimitError` from OpenRouter, `_create_completion()` now retries the same
request against a second, independently-configured Groq client (if `GROQ_API_KEY` is
set), rather than failing immediately.

**Prevention.** Any external dependency an app calls synchronously in its critical
path is a single point of failure until it has at least one fallback — worth building
in before it's needed in production, not just after the first time it bites.

---

## 3. The Groq fallback model didn't exist on Groq's actual lineup (same area)

**Problem.** While building the Groq fallback above, the first model name tried on
Groq (`llama-3.3-70b-versatile`) was rejected outright.

**Root cause.** The model string was assumed to be available on Groq without checking
Groq's actual served-model list at the time — OpenRouter and Groq don't share a model
namespace, so a model name that's valid on one provider isn't guaranteed to mean
anything on the other.

**Diagnosis.** Groq's API returned a `404 model_not_found` error on the very first
fallback call.

**Fix.** Switched `FALLBACK_MODEL` to `openai/gpt-oss-120b`, chosen deliberately (not
just as "a model that exists") for explicit structured-output support, since
`generate_structured()` — used by the Planner, Researcher's extraction step, and
Supervisor — depends on it.

**Prevention.** Verify a model identifier actually exists on the specific provider
being called (via that provider's own model-listing endpoint or docs) before wiring it
into a fallback path that, by definition, only gets exercised under failure — a
fallback that itself fails on first use is worse than no fallback, since it fails
silently until the primary path breaks.

---

## 4. Orphaned database row on a lost idempotency race (Milestone 10 follow-up)

**Problem.** Two requests with the same `idempotency_key` arriving close together
could both create a `research_runs` row before either checked Redis, leaving one row
permanently stuck at `status="pending"` with no run ever executing it.

**Root cause.** The original flow was: check Redis for an existing run → if none,
create a run row → then claim the idempotency key in Redis. Between "create a run row"
and "claim the key," a second concurrent request could also miss the Redis check (it
hasn't been claimed yet) and create its *own* row. Whichever request lost the
subsequent `SET NX` race had already created a real row in Postgres with nothing left
to clean it up.

**Diagnosis.** Reasoned through directly from the two-step (check-then-act) shape of
the idempotency logic — a classic TOCTOU (time-of-check to time-of-use) race — rather
than caught after an actual concurrent-request incident.

**Fix.** After losing the `SET NX EX` race, the request now explicitly deletes the run
row it had already created (`run_store.delete_run()`) and re-fetches whichever
`run_id` actually won the race from Redis, returning that one instead. Covered by
`test_losing_idempotency_race_deletes_the_orphaned_run`, which simulates the exact
sequence a lost race produces (initial `GET` misses, `SET NX` always fails, follow-up
`GET` returns the winner's `run_id`).

**Prevention.** Any "check external state, then act" sequence across two systems
(Redis + Postgres here) needs an explicit plan for "what if I lose a race I didn't
know I was in" — don't assume the check-then-act window is too short to matter.

---

## 5. Real `GROQ_API_KEY` in `.env` silently doubling test setup cost (Milestone 12)

**Problem.** The unit test suite got noticeably slower (~57.92s, consistently) right
around when Groq-fallback support and its tests were added.

**Root cause.** `pydantic-settings` reads `.env` directly as a fallback source below
OS environment variables — so any test file whose fixture didn't explicitly neutralize
`GROQ_API_KEY` picked up the real key from `.env`. That made `LLMClient.__init__`
construct a *second* real `AsyncOpenAI` client (the Groq fallback client) on every
single `LLMClient()` instantiation across the suite, and each `AsyncOpenAI()`
construction cost ~0.7–0.8s.

**Diagnosis.** Noticed the suite's runtime had grown disproportionately to the number
of new tests added, then traced it to `LLMClient.__init__`'s conditional fallback-client
construction being tied to `settings.GROQ_API_KEY` truthiness, and confirmed most test
fixtures used `monkeypatch.delenv("GROQ_API_KEY", raising=False)` — which does nothing
against a `.env`-sourced value, since deleting the *environment* variable just lets
pydantic-settings fall through to `.env` anyway.

**Fix.** Every test file's settings fixture now does
`monkeypatch.setenv("GROQ_API_KEY", "")` (an explicit empty string, not deletion) —
this outranks `.env` in precedence and is still falsy, so `LLMClient()` builds exactly
one real client in tests regardless of what's in the local `.env`. This was later
consolidated into a single shared `tests/unit/conftest.py` fixture (Milestone 13) so
the fix — and the reasoning behind it — lives in one place instead of being duplicated
(and potentially drifting) across seven files.

**Prevention.** `monkeypatch.delenv` only removes an *environment* variable; it cannot
"unset" a value a settings library reads from a file underneath the environment.
When a config value has a file-based fallback source, tests must override it with an
explicit falsy value, not deletion.

---

## 6. Coverage tool silently under-reporting due to Python 3.13 + background threads (Milestone 13)

**Problem.** Running `pytest --cov=app` reported ~70% coverage on `app/api/routes.py`,
with large stretches of route-handling code — including code exercised by passing,
assertion-checked tests — marked as "missing."

**Root cause.** Starlette's `TestClient` (used by every API test) runs the ASGI app's
async request handling inside a background thread via `anyio`'s blocking portal.
`coverage.py`'s default C-based tracer, on this Python version, doesn't reliably follow
execution into that background thread — so code that was genuinely running (and being
asserted on) never got marked as covered, purely as an artifact of *how* it ran, not
*whether* it ran.

**Diagnosis.** The coverage numbers didn't match reality: tests that directly asserted
on behavior inside the "uncovered" line ranges were passing, which is only possible if
that code executed. Rather than trusting the tool's first output, tried
`concurrency = thread` in a `.coveragerc` (the standard fix for missed-thread coverage)
first — no change — then switched the tracer itself to Python 3.13's newer
`sys.monitoring`-based core (`core = sysmon`), which is process-wide by design and
correctly captured the background-thread execution. Coverage on `app/api/routes.py`
immediately jumped to the accurate number once the real gaps (an actually-unexercised
function, an actually-unexercised exception branch) were visible on their own.

**Fix.** `.coveragerc` sets `concurrency = thread` and `core = sysmon`. This is also
why the milestone's coverage-gap-closing pass found real gaps that had been masked
underneath the false ones the whole time.

**Prevention.** A coverage report that contradicts what you already know to be tested
(a passing, assertion-heavy test on a line marked "missing") is a signal to question
the tool's configuration before trusting its output — especially with threaded test
clients, which are a known blind spot for coverage tracers.

---

## 7. Docker itself could not be verified inside the assistant's own sandboxed environment (Milestone 14)

**Problem.** The Dockerfile, `.dockerignore`, and `docker-compose.yml` were written and
reviewed, but the AI assistant building them had no `docker` CLI available in its own
sandboxed working environment — `docker compose build`/`up` could not be run directly
as part of that milestone's own verification step.

**Root cause.** A sandboxed development environment doesn't necessarily include every
tool the project itself targets — containerization tooling in particular, since
running Docker-in-Docker or a full container runtime inside an already-sandboxed agent
environment isn't generally available.

**Diagnosis/workaround.** The closest available proxies were run instead: the exact
`uvicorn app.main:app --host 0.0.0.0 --port 8000` command the Dockerfile's `CMD` uses,
confirming `/docs` was reachable; `alembic upgrade head` against the real Supabase
`DATABASE_URL`, confirming it was a clean no-op; and a real `POST /research` request
through the locally-running app, confirming it could reach OpenRouter, Tavily, and
Supabase over the network. These proxies validate the *application* code path the
container would run, but not the container build/image itself (base image
correctness, non-root user permissions inside the container, `.dockerignore`
exclusions actually taking effect, etc.).

**Fix.** The actual `docker compose build && docker compose up` was run for real,
directly by the project owner, on their own machine with Docker installed — the first
true container-based confirmation of the whole setup. That real run is also what
surfaced problem #8 below, since it was a genuine end-to-end request against real
accumulated findings.

**Prevention.** Treat host-process proxies (same command, same code, no container) as
useful but incomplete evidence for a Dockerization milestone — get a real
container-runtime confirmation from an environment that actually has Docker before
considering it verified, not just a "should behave the same way" argument.

---

## 8. Supervisor prompt exceeded Groq's token-per-minute limit during real Docker testing (Milestone 14/15)

**Problem.** A real end-to-end research run — first observed via a local (non-container)
sanity-check request, then reproduced by the project owner's actual `docker compose up`
run — failed with: OpenRouter's configured model rate-limited, and the Groq fallback
*also* failed, with Groq returning a 413/token-rate-limit error (~9,875 tokens
requested against an 8,000-tokens-per-minute cap).

**Root cause.** Two independent issues stacked: (1) `DEFAULT_MODEL` was set to a
free-tier OpenRouter model prone to rate limiting under a full run's ~10-12 sequential
LLM calls, which triggered the Groq fallback path at all; and (2) the Supervisor's
evaluation prompt (`_build_evaluation_prompt()`) included full detail — claim,
`source_url`, and snippet — for every finding collected so far, so the prompt grew
linearly with findings and had reached ~9,875 tokens by the time enough research had
accumulated, exceeding Groq's real account-level TPM quota.

**Diagnosis.** The failure surfaced as a real `LLMCallError` with both providers'
actual error messages chained together (visible in the run's `error` field and in
`_execute_research_run`'s exception handling, which correctly recorded
`status="failed"` rather than crashing) — the Groq error message itself
(`Request too large ... tokens per minute (TPM): Limit 8000, Requested 9875`) pointed
directly at prompt size as the cause.

**Fix (Milestone 15).** `_build_evaluation_prompt()` now groups findings by
sub-question and sends, per group, just the sub-question text, a count of supporting
findings, and each claim truncated to ~100 characters — `source_url` and full snippets
are omitted entirely from this prompt, since the Supervisor's job is judging
sufficiency, not verifying citations (the Writer's prompt, which does need full detail
for accurate citations, was deliberately left unchanged). A log line now reports the
constructed prompt's approximate token count (character count ÷ 4) before every
Supervisor call, so a prompt growing back toward a provider's limit is visible in logs
before it causes a real failure again.

**Prevention.** Any prompt built from an accumulating, unbounded collection (findings,
messages, search results) needs an explicit size strategy — grouping, truncation, or a
hard cap — from the start, not just enough detail to look complete in a small test run.
Log the size, not just the content, so growth is visible before it becomes a real
failure. (The Writer's prompt has this same latent risk and is explicitly listed as a
known limitation in `README.md`, not yet fixed.)

---

## 9. Supabase's direct connection string is unreachable from inside Docker (Milestone 16)

**Problem.** After adding the Streamlit frontend and finally running the full stack
under real Docker (Docker itself wasn't available until this milestone — see problem
#7), submitting a question through the UI returned a 500 from `POST /research` every
time, even though the exact same `DATABASE_URL` worked fine for every non-Docker run
throughout the whole project.

**Root cause.** `DATABASE_URL` used Supabase's direct connection host
(`db.<project>.supabase.co`), which resolves to an IPv6 address. Docker Desktop's
default bridge network doesn't route IPv6 traffic out by default, so `asyncpg`'s
connection attempt from inside the `app` container failed outright — while the same
hostname resolved and connected fine from the host machine directly, which does have
working IPv6 routing. Same connection string, different network path, only one of them
broken.

**Diagnosis.** The app container's logs showed the real traceback ending in
`OSError: [Errno 101] Network is unreachable` inside asyncpg's connection setup — not a
timeout, not an auth failure, a routing-level failure, which pointed straight at
IPv6/network path rather than credentials or the database itself.

**Fix.** Switched `DATABASE_URL` to Supabase's Session Pooler connection string
(`aws-0-<region>.pooler.supabase.com`), which is IPv4-reachable. No code changes
required — purely a connection-string change, confirmed by rebuilding and confirming
`POST /research` succeeded through the container afterward. `.env.example` and
`README.md` now both call this out explicitly next to `DATABASE_URL` so it isn't
rediscovered blind.

**Prevention.** When a hosted service offers both a direct and a pooled/proxied
connection option, prefer the pooled one for anything that might run inside a
container or a restrictive network — direct connections to modern hosting providers
increasingly default to IPv6-only or IPv6-preferred, which is exactly the kind of thing
that works on a developer's host machine and silently fails in a container.

---

## 10. Non-root container user couldn't write to its own working directory (Milestone 16)

**Problem.** After fixing problem #9, the frontend container's logs showed a second,
unrelated error on every session event: `PermissionError: [Errno 13] Permission
denied: '/app/.streamlit'`, thrown when Streamlit tried to write its local
usage-telemetry ID file.

**Root cause.** `Dockerfile.streamlit`'s `WORKDIR /app` creates that directory while
still running as `root` (before the `USER appuser` switch later in the file). The
subsequent `COPY --chown=appuser:appuser streamlit_app.py .` only changes ownership of
the file it copies, not the pre-existing `/app` directory itself — so `/app` stayed
root-owned, and the non-root `appuser` couldn't create a new `.streamlit/` subdirectory
inside it at runtime.

**Diagnosis.** Read the container's own traceback directly — it named the exact path
(`/app/.streamlit`) and exact syscall failure (`PermissionError: [Errno 13]`), which is
unambiguous: a non-root user, an operation needing write access to a directory it
doesn't own.

**Fix.** Added `chown appuser:appuser /app` immediately after creating the user, in
both `Dockerfile.streamlit` and (preemptively, since it has the same latent gap even
though nothing had triggered it there yet) the main `Dockerfile` — before either
`COPY` step runs, so the working directory itself is owned by the user that will later
run as, not just the files copied into it.

**Prevention.** Creating a non-root user and copying files into a `WORKDIR` with
`--chown` is not the same as that user owning the directory itself — anything that
later needs to create a *new* file or subdirectory there (caches, telemetry, temp
files) needs the directory's own ownership fixed explicitly, not just the files known
about at build time.
