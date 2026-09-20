# Interview Prep: Multi-Agent Research Assistant

This document is self-contained — paste the whole thing into a new chat with no other
context and it can run a mock interview based on this project. It describes a real
system that was actually built (not a hypothetical), including the real bugs hit while
building it, so answers can be grounded in specifics rather than generalities.

Usage instructions for whoever is using this to run the mock interview are at the very
end of this document.

---

## Project description

### 30-second version

"I built a multi-agent research assistant: you give it a question, and it plans
sub-questions, searches the web for grounded evidence on each one, has a Supervisor
agent judge whether the evidence is actually sufficient before proceeding, and then
writes a cited Markdown report — only once that gate passes. It's built on LangGraph
for orchestration, FastAPI for the API, Postgres for persistence, and Redis for
request idempotency, with hard caps on iteration count and token usage so it can't
loop forever or run away on cost, and per-node execution tracing so every run's
behavior and cost is inspectable after the fact. It's fully containerized with Docker
Compose, including a small Streamlit UI on top of the API for actually trying it
without reaching for curl."

### 2-minute version

"The core idea is separating research into four distinct agent roles instead of one
big prompt: a Planner breaks the question into 3-5 focused sub-questions; a Researcher
sequentially searches the web (via Tavily) for each sub-question and extracts only
claims directly supported by search results, each claim paired with its source URL; a
Supervisor — a separate LLM call producing a structured boolean decision — judges
whether the collected findings are actually sufficient to write from, and if not,
routes back to the Researcher for another pass; and only once the Supervisor is
satisfied (or a hard iteration cap of 2 is hit) does a Writer agent turn everything
into a Markdown report with proper citations.

That loop is a LangGraph `StateGraph` with a real Pydantic state object threaded
through every node. Two independent safety mechanisms bound it: a hard 2-iteration cap
on the Supervisor loop, so an LLM that keeps saying "not sufficient" can't loop
forever, and a 50,000-token budget checked by the Researcher and Writer so a single
expensive iteration can't run away on cost either — if the budget trips, the Researcher
stops early and the Writer still produces a best-effort report that's honest about the
cutoff, rather than failing outright.

Around that graph sits a FastAPI service: `POST /research` kicks off a run as a
background task and returns immediately with a run ID; `GET /research/{id}` polls
status and result; `GET /research/{id}/trace` returns a full per-node execution trace
— tokens, latency, success/failure — for every node that ran. Postgres (hosted on
Supabase) is the system of record for runs, findings, reports, and traces; Redis
(hosted on Upstash) handles exactly one job, an atomic idempotency lock so a repeated
request with the same key doesn't start a duplicate run.

One real production concern that came up during development: the primary LLM
provider (OpenRouter) rate-limited under real use, so I built an automatic fallback to
Groq on rate-limit errors specifically — and later, a real Docker-based end-to-end test
found that Groq's own token-per-minute quota could also be exceeded by a Supervisor
prompt that grew unboundedly with accumulated findings, which I fixed by compacting
that specific prompt (grouped, truncated, no full snippets) rather than raising the
model's other complexity.

Getting the whole thing running in real Docker (not just code that should theoretically
work in a container) surfaced two more real bugs at the very end: Supabase's direct
Postgres connection string resolves to IPv6, which Docker's default network couldn't
route out over, so the app container failed with a raw 'network unreachable' error
even though the exact same connection string worked fine outside Docker — fixed by
switching to Supabase's IPv4-reachable connection pooler string. And the Streamlit
container's non-root user didn't actually own its own working directory, since
`COPY --chown` only covers the files it copies, not a directory `WORKDIR` already
created as root — it failed trying to write a local telemetry file at runtime, fixed
with an explicit `chown` on the directory itself."

---

## Likely interview questions and model answers

### Why multi-agent instead of a single LLM call?

A single prompt that both searches and writes has no independent check on whether it
actually found enough evidence before it starts drafting prose — "have I researched
enough" and "write this up well" are different skills, and conflating them means the
model can talk itself into writing a confident-sounding report on thin evidence with
nothing to catch it. Splitting the Supervisor out as a separate, structured-output
decision point makes "is this sufficient?" an explicit, testable gate instead of an
implicit assumption baked into one giant prompt. It also means each agent's prompt
stays focused and each step is independently testable — I can (and did) unit-test the
Planner's decomposition, the Researcher's extraction, the Supervisor's decision logic,
and the Writer's citation handling completely separately, with the LLM itself mocked
out.

### Why LangGraph specifically?

The control flow here isn't linear — there's a loop (Researcher ↔ Supervisor) with a
conditional exit — and LangGraph gives that a first-class `StateGraph` with typed state
and explicit conditional edges (`add_conditional_edges`), rather than hand-rolling a
`while` loop around LLM calls with manual state-passing. The state object
(`ResearchState`, a Pydantic model) flows through every node and accumulates fields
(findings, iteration count, token usage) as it goes, so the graph structure and the
state shape are both explicit and inspectable, not implicit in imperative code.

### Why a Supervisor pattern with a hard iteration cap of 2 specifically?

The Supervisor's judgment is itself an LLM call, and it fails open (defaults to
`research_complete=True`) if that call errors — which is good for availability but
means nothing *inside* the LLM's judgment guarantees termination. A broad or
effectively unanswerable question could get "not sufficient yet" from the Supervisor
indefinitely. The hard cap makes termination and cost bounded regardless of what the
LLM decides — worst case, exactly 2 research iterations run before the graph is forced
to the Writer. When that forced-completion happens, `research_incomplete=True` gets
set specifically so the final report can honestly disclose that some areas may be
under-researched, rather than silently presenting a capped-out run as if it were
naturally complete. Two was chosen as "enough to allow one genuine follow-up pass, not
so many that a bad first pass compounds cost" — it's a starting point, not something
rigorously tuned against real outcome data yet.

### Why both Redis AND PostgreSQL — why not just one?

They're solving different-shaped problems. Postgres is the system of record: durable,
relational, needs to survive restarts, and needs to be queryable (find all runs with
`status="failed"`, join findings to their run, etc.) — that's a fundamentally relational,
persistent-storage problem. The idempotency check is a different shape entirely: "has
anyone already claimed this specific key in the last 24 hours," an atomic
check-and-set on a single ephemeral flag with a TTL, not data anyone ever needs to
query relationally. Redis's `SET key value NX EX <ttl>` is exactly that operation,
atomically, in one call. Doing the equivalent in Postgres means a unique constraint
plus catching the resulting integrity error plus manually expiring old rows — more
ceremony for something that's inherently short-lived and non-relational.

### Why FastAPI?

The whole app is async end-to-end — the agent nodes make async LLM/HTTP calls, the DB
layer is SQLAlchemy's async engine, Redis is `redis.asyncio` — and FastAPI is async-native
rather than async-bolted-on, so route handlers, dependency injection (`Depends`), and
background tasks all compose naturally with that without event-loop friction. It also
generates OpenAPI docs automatically (`/docs`), which was genuinely useful for manually
exercising the API during development without hand-writing a client.

### Why Tavily specifically for search?

Tavily is built for exactly this use case — LLM agents that need grounded web
evidence — and returns normalized `title`/`url`/`content` results directly, rather than
raw HTML a scraper would need to clean up itself. That let `search_web()` stay a thin
wrapper: call Tavily, deduplicate by URL, return a plain list of dicts, with a single
`SearchToolError` type shielding callers from Tavily's own SDK-specific exceptions.

### How are infinite loops actually prevented — walk through the mechanism.

Two independent mechanisms, at two different layers, deliberately not relying on just
one:

1. **Iteration cap.** `supervisor_node` increments `state.research_iterations` every
   time it runs, and checks it against `MAX_RESEARCH_ITERATIONS` (2). If the cap is
   reached *and* the LLM still says "not sufficient," the code overrides the LLM's
   decision — forces `research_complete=True` and sets `research_incomplete=True` — so
   the graph routes to the Writer regardless of what the model wants. This is a pure
   Python state check in `route_after_supervisor()`, not an LLM judgment, so it's
   deterministic and can't be talked out of firing.
2. **Token budget.** Independently, `state.total_tokens_used` is checked against
   `MAX_TOKEN_BUDGET` (50,000) — the Researcher checks it before *each* sub-question
   (not just once per iteration) and can stop mid-loop even within a single
   supervisor-approved iteration, and the Writer checks it once at the start. This
   catches the case where a single iteration alone (many sub-questions, large search
   results) is already too expensive, which the iteration cap alone wouldn't catch.

Both are plain state checks, not further LLM calls, specifically so they can't fail
the same way the thing they're guarding against could fail.

### How are costs / token budgets controlled and enforced?

`LLMClient` accumulates `total_input_tokens`/`total_output_tokens` as instance
attributes on every call (`_accumulate_usage()`, reading `completion.usage` off the
raw SDK response) — this covers both `generate()` and `generate_structured()`, even
though `generate_structured()` only returns a parsed Pydantic model with no
token-carrying wrapper of its own, since these are the one place both call shapes'
usage is available uniformly. Each agent node creates its own short-lived `LLMClient`,
does its work, and adds that instance's totals into `state.total_tokens_used` before
finishing — so the token count is a genuine running total across the whole graph
execution, not an estimate. The Researcher and Writer both compare that running total
against `MAX_TOKEN_BUDGET` before doing more expensive work, as described above. All
of this is also persisted per-node in `ExecutionTrace` rows, so token cost is
auditable per node, per run, after the fact — not just enforced live.

### How does the system handle API failures and rate limits — tell the real story.

There are two layers here, and the real story involves the second one actually
failing too. `llm_client.py` wraps every OpenAI-SDK exception into a single
`LLMCallError` type so callers never handle provider-specific exceptions directly.
Network/timeout errors (`APIConnectionError`, `APITimeoutError`) get retried with
exponential backoff via a generic `with_retry()` decorator. A `RateLimitError`
specifically triggers a fallback: the same request gets replayed against a second,
independently-configured Groq client with a fixed fallback model
(`openai/gpt-oss-120b`, chosen because it supports structured outputs, which the
Planner/Researcher/Supervisor all depend on).

That fallback exists because it was needed for real — OpenRouter's model rate-limited
during normal development before any of this failure-handling existed, which is what
motivated building it. Building the fallback itself hit its own bug: the first model
name tried on Groq (`llama-3.3-70b-versatile`) didn't exist there and returned a `404
model_not_found` — OpenRouter and Groq don't share a model namespace.

And later, during real Docker-based end-to-end testing, the fallback *itself* failed
under real load: a Supervisor prompt that had grown to ~9,875 tokens (full claim text
+ source URLs + snippets for every accumulated finding) exceeded Groq's own
8,000-tokens-per-minute quota, so both the primary call and its fallback failed in the
same run. That failure was caught correctly (`_execute_research_run`'s exception
handling recorded `status="failed"` with the real chained error message rather than
crashing), but the underlying prompt-growth problem needed an actual fix: the
Supervisor's prompt was rewritten to send grouped, truncated, citation-free summaries
instead of full per-finding detail, plus a log line reporting the prompt's estimated
token count before every call so a similar growth is visible before it fails again.

### How are citations / source provenance validated and deduplicated?

Every `Finding` the Researcher extracts is explicitly tied to the exact search-result
URL it came from (`ExtractedFinding.source_url`), and the Researcher's own system
prompt instructs the LLM to extract *only* claims directly supported by the provided
search-result content — no outside knowledge, no fabrication — so provenance is
attached at extraction time, not reconstructed afterward. Deduplication happens at
report-writing time, in code, not left to the LLM: `_build_report_prompt()` computes
`dict.fromkeys(finding.source_url for finding in state.findings)` — an
order-preserving, deduplicated list — and hands that to the Writer as the
authoritative list to use for the Sources section, explicitly instructed not to
compile its own list from the (deliberately non-deduplicated) findings. That's a fix
for a real bug: earlier, the same URL could legitimately appear multiple times across
different sub-questions' findings (correct — one article can support two different
sub-questions), but the Writer's Sources list wasn't deduplicating that on its own,
so the same link showed up two or three times in the final report until the explicit,
code-computed list was introduced.

### How is execution traced?

Every agent node function times itself (`started_at`/`completed_at`), and on both
success and failure calls `record_node_trace()`, which writes one `ExecutionTrace` row
per node execution — `node_name`, `input_tokens`, `output_tokens`, `latency_ms`,
`status` (`success`/`failure`), and an `error` message if it failed — opening its own
short-lived DB session so tracing works even when nodes are called directly (e.g. in
tests) without a request-scoped session. Trace recording deliberately can never crash
a node or mask its real error: any exception while writing a trace is caught and
logged, never re-raised. `GET /research/{run_id}/trace` exposes the full ordered list
of these rows for a run, alongside the run's overall `total_tokens_used` and
`budget_exceeded` flag — so a run's full cost and timing breakdown, node by node, is
inspectable after the fact, and a failed run's trace shows exactly which node failed
and why.

### How would this scale under concurrent requests?

`POST /research` returns immediately and runs the actual graph execution as a FastAPI
`BackgroundTask`, so the API layer itself doesn't block per-request on LLM/search
latency — many requests can be in flight concurrently as background tasks within one
process. The idempotency mechanism (Redis `SET NX EX`) is what actually makes
concurrent *duplicate* requests safe: two requests with the same key racing each other
are handled deterministically (one wins, the other's orphaned DB row gets cleaned up
and both return the same `run_id`). Real scaling beyond a single process's background
tasks would mean moving execution to a proper task queue (Celery, arq, or similar)
with multiple worker processes pulling from it, since in-process `BackgroundTasks`
doesn't survive a process restart or scale beyond one machine's concurrency — that's a
genuine next step this project doesn't yet implement, not something it currently does.

### What happens if Tavily or an LLM provider goes down entirely (not just rate-limited)?

For the LLM providers: if OpenRouter is down and Groq isn't configured (or is also
down), `_create_completion()` ultimately raises `LLMCallError`, which propagates up
through the calling node. Only the Supervisor explicitly catches `LLMCallError` and
fails open (defaults to completing the research rather than blocking forever) — the
Planner, Researcher, and Writer let it propagate, which trips each node's own
`except Exception` trace-and-reraise handling, and `_execute_research_run` catches it
at the top level and marks the whole run `status="failed"` with the real error message
recorded — so the system degrades to a clearly-failed, inspectable run rather than
hanging or crashing the server. If Tavily is down, `search_web()` retries (it retries
on the bare `Exception` type, since Tavily's SDK has no shared exception base to be
more selective with) and then raises `SearchToolError`; the Researcher explicitly
catches that per sub-question and treats it as "no findings for this sub-question"
rather than failing the whole run — so a Tavily outage degrades findings quality for
that run rather than failing it outright, which is a deliberately different failure
mode than an LLM outage.

---

## System-design questions specific to this project's architecture

**Walk me through exactly what happens, system-by-system, from `POST /research` to a
completed run.** *(Use this to check you can trace: request → optional Redis
idempotency check → Postgres run-row creation → 202 response → background task starts
→ LangGraph `ainvoke` → planner_node → researcher_node (loop) → supervisor_node →
conditional routing → writer_node → final `run_store.update_run()` writing status,
findings, report, and token totals back to Postgres.)*

**Why is the graph invoked from inside a `BackgroundTasks` callback rather than
awaited directly in the route handler?** *(So `POST /research` can return immediately
with a `run_id` instead of blocking the HTTP response for however long the full
research run takes — the client polls `GET /research/{id}` for status instead.)*

**The Researcher processes sub-questions sequentially. How would you make that
concurrent, and what would break if you just wrapped it in `asyncio.gather`
naively?** *(Token-budget checking currently happens *between* sub-questions, reading
a shared running total off `llm_client.total_input_tokens` — running sub-questions
concurrently with separate `LLMClient` instances would need the budget check
redesigned around a shared, atomically-updated counter rather than a simple
before-each-iteration comparison, since concurrent tasks could all pass the check
before any of them updates the total.)*

**The Writer's prompt still includes full finding detail per finding, unlike the
now-fixed Supervisor prompt. Design a fix for when that becomes a problem.**
*(Options to discuss: truncate snippets specifically while keeping full claims and
source_urls intact for citation accuracy; chunk findings and summarize in batches;
switch to a model with a larger context window for the Writer step specifically,
since it's a single call, not a per-sub-question loop like the Researcher.)*

**How would you add authentication/multi-tenancy to this API?** *(There's currently no
user/auth concept at all — every run is globally visible by `run_id`. Discuss adding a
user/API-key concept, scoping `research_runs` rows to an owner, and what changes in the
idempotency key's Redis namespace to avoid cross-tenant collisions.)*

**How would you support streaming the report back as it's written, instead of
polling?** *(Discuss Server-Sent Events or WebSockets from the API layer, and that
the Writer currently makes one non-streaming `generate()` call — using OpenAI SDK
streaming there and forwarding tokens as they arrive would be the actual code change,
plus what happens to `total_tokens_used` accounting mid-stream.)*

**How would you migrate off Supabase/Upstash to self-hosted Postgres/Redis?**
*(Nothing in the app talks to Supabase or Upstash specifically — `DATABASE_URL` and
`REDIS_URL` are generic connection strings read via `pydantic-settings`, and
`_to_async_url()` just rewrites a plain `postgresql://` scheme to
`postgresql+asyncpg://`. Migration is a connection-string change, not a code change —
that's the point of hosting them externally in the first place.)*

---

## Debugging-style questions (based on real issues from this project)

**"Our test suite got noticeably slower after adding a fallback-provider feature, with
no obvious new slow tests. How would you find out why?"** *(Real answer: this actually
happened. Root cause was `pydantic-settings` reading a real API key out of `.env` as a
fallback source below environment variables, so `monkeypatch.delenv()` in test
fixtures didn't actually neutralize it — the real key made every `LLMClient()`
construct a second real SDK client. Diagnosis approach to describe: notice the
slowdown correlates with a specific feature's tests, inspect what that feature's
`__init__` does differently, check whether "unset" env vars in tests are actually
being read from a file underneath the environment.)*

**"A coverage report shows 0% coverage on a function you know is being exercised by a
passing test that asserts on its behavior. What do you check?"** *(Real answer: this
happened with `pytest-cov` and Starlette's `TestClient`, which runs async route
handling in a background thread that the default coverage tracer doesn't reliably
follow on Python 3.13. Diagnosis approach: trust the passing test over the coverage
tool first, then check whether the "uncovered" code runs in a thread, subprocess, or
other non-main execution context the tracer might not be configured to follow — try
`concurrency = thread` first, then a different tracer core if that doesn't fix it.)*

**"A background job that's supposed to be idempotent under a given key created two
database rows for what should have been one request."** *(Real answer: this happened
with the idempotency-key flow here. Root cause was a check-then-act race — checking
Redis for an existing key, then creating a DB row, then claiming the key — with a gap
between the check and the claim that a concurrent second request could land in.
Diagnosis approach: look for any place code checks external state and then acts on it
in two separate steps; that gap is where races live, regardless of how narrow it looks.
Fix approach: after losing the race, explicitly clean up the row you already created
and defer to whichever request actually won.)*

**"A fallback path you built for reliability failed on its very first real use, with a
'model not found' style error from the fallback provider."** *(Real answer: this
happened with the Groq fallback here — the first model name tried
(`llama-3.3-70b-versatile`) didn't exist on Groq specifically, and returned a 404.
Root cause: assuming a model name valid on one provider is valid on another, without
checking. Diagnosis approach: read the fallback provider's actual returned error
message and status code directly rather than assuming the fallback logic itself is
broken — a 404 on the model name is a different bug than a fallback that never
triggers at all.)*

**"A request pipeline works fine in every test and small manual run, but fails for
real once enough data has accumulated within a single run."** *(Real answer: this is
exactly the Supervisor/Groq TPM failure here — the Supervisor's prompt grew linearly
with accumulated findings and only exceeded a real provider quota once enough research
had actually accumulated within one run, which small tests with 1-2 findings never
reproduce. Diagnosis approach: for anything built from an accumulating, unbounded
collection, ask "what does this look like at 10x the size a normal test uses" before
assuming a small-scale passing test means the design is safe; log the actual size of
the constructed artifact (prompt length, token estimate), not just its content.)*

**"A database connection string works perfectly from every developer's machine but
fails with a low-level network error the moment the exact same app runs inside a
Docker container."** *(Real answer: this happened with Supabase's direct connection
host here — it resolves to an IPv6 address, and Docker Desktop's default network
doesn't route IPv6 traffic out, so `asyncpg` failed with `OSError: [Errno 101] Network
is unreachable` from inside the container while the identical connection string worked
fine on the host. Diagnosis approach: when the exact same config works in one
environment and fails in another with a low-level OS/network error rather than an
auth or timeout error, suspect the network path itself, not the credentials or the
remote service — check whether the failing environment has a different IP protocol
story (IPv4-only container network vs. dual-stack host). Fix approach: prefer a
hosted service's pooled/proxied connection option over its direct one when the client
might run somewhere with restricted networking.)*

**"A container running as a non-root user gets a permission error trying to write a
file inside its own working directory — a directory the Dockerfile's `COPY --chown`
was supposed to hand to that exact user."** *(Real answer: this happened with the
Streamlit container here — Streamlit tried to write a local telemetry ID file and got
`PermissionError: [Errno 13] Permission denied`. Root cause: `WORKDIR` creates that
directory while the Dockerfile is still running as root, before the user switch, and
`COPY --chown=user:group` only changes ownership of the files it copies *into* that
directory — not the pre-existing directory entry itself. Diagnosis approach: read the
exact path in the traceback and ask "was this path created before or after the user
was switched, and by which instruction" — ownership of a file and ownership of the
directory it lives in are two separate things. Fix approach: `chown` the directory
itself, explicitly, right after creating the user and before switching to it.)*

---

## Instructions for the interviewer (paste this whole document, then say this to Claude)

Ask me questions from this document one at a time, don't reveal the model answer until
I respond, then give feedback on my answer before moving to the next question. Mix in
the system-design and debugging-style questions, not just the model-answer ones — for
those, let me attempt a design/diagnosis first before comparing it to the real answer
documented here.
