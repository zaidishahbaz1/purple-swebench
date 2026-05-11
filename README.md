# purple-swebench

A purple (attacker-side) agent for the [AgentBeats](https://agentbeats.org) **Coding Agent** track, evaluated on [SWE-bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os) — 731 real-world Python issues from open-source repositories. Given a problem statement and the buggy repository as a Docker image, the agent must produce a unified diff that, when applied, makes the hidden test suite pass.

## Abstract

This agent is a direct application of **Recursive Language Models (RLM)** — the inference strategy introduced by Zhang, Khattab, and Kraska (MIT CSAIL, 2025) in [_Recursive Language Models_](https://arxiv.org/abs/2512.24601). RLM lets a root LM decompose long-context work by handing chunks to a recursive `llm_query()` call inside a Python interpreter, instead of cramming everything into one window. We adopt that pattern wholesale and wrap it around the SWE-bench Pro single-shot patch-generation protocol.

Our root agent (`gpt-4o`) runs a ReAct loop with three tools:

- **`bash`** — execute commands inside a sidecar container spawned from the task's Docker image, giving the model a real shell over the buggy repo (read files, grep, run git, try edits, validate with `git apply --check`).
- **`repl`** — a persistent Python REPL where intermediate state (file contents, search results, candidate patches) is stashed in a `context` variable that survives across turns. This is the RLM scratchpad: bulky intermediate state lives in the interpreter rather than the root LM's window.
- **`final`** — emit the final unified diff as the task artifact.

Inside the REPL the model can call **`llm_query(prompt, content)`** — this is the RLM recursive call. It dispatches to a cheaper sub-LLM (`gpt-4o-mini`) and lets the root model hand off large-context grunt work — "find the function that does X in this 5K-line file", "summarize this stack trace" — without paying the token cost of pulling that content into its own window. Recursion is bounded (`MAX_REPL=20`, `MAX_BASH=30`, `MAX_INNER_STEPS=30`) and every sub-call is logged.

The result is an agent that behaves like a small engineering team: a planner with bash access, a scratchpad for state, and a junior assistant on call for bulk reads.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  A2A server (a2a-python)                                     │
│  ── receives { problem_statement, docker_image, ... }        │
└──────────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────┐
                        │  Root LLM        │   gpt-4o
                        │  (ReAct loop)    │   tool_choice=required
                        └─────┬────────────┘
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
   ┌─────────┐          ┌──────────┐           ┌──────────┐
   │  bash   │          │   repl   │           │  final   │
   │ docker  │          │ persist. │           │  emit    │
   │  exec   │          │  python  │           │  patch   │
   └────┬────┘          └─────┬────┘           └──────────┘
        │                     │
        ▼                     ▼
   sidecar             llm_query() ──► gpt-4o-mini
   container                          (sub-LLM for
   (SWE-bench Pro                      bulk reads)
    image)
```

Each task spins up a sidecar container from the task's image (`docker run --entrypoint /bin/sh ... sleep N`), the agent works on the repo via `docker exec`, and the container is torn down at the end of the task. The framework's `docker.sock` is mounted via Amber's `experimental_features: ["docker"]` mechanism.

## Project structure

```
src/
├─ server.py        # A2A server + agent card
├─ executor.py      # A2A request handling
├─ agent.py         # RLM-style ReAct loop (root agent + REPL + sub-LLM)
└─ messenger.py     # A2A messaging utilities
scripts/
├─ local_smoke.py   # Direct-import smoke test (full flow, real LLM calls)
└─ docker_smoke.py  # Docker-only smoke test (no LLM, verifies sidecar exec)
amber-manifest.json5
Dockerfile
```

## Running locally

```bash
# Verify docker sidecar exec works (no LLM, $0)
uv run scripts/docker_smoke.py

# Full end-to-end on one task (~$1-3 in OpenAI credits)
export OPENAI_API_KEY=sk-...
uv run scripts/local_smoke.py
```

## Submission

Deployed via Amber manifest (`amber-manifest.json5`) and submitted through the [SWE-bench Pro leaderboard](https://github.com/RDI-Foundation/swe-bench-leaderboard) Quick Submit form. The Amber image is built from `Dockerfile` and pushed to `ghcr.io/zaidishahbaz1/purple-swebench:latest` by GitHub Actions on push to `main`.

## Citation

This work is a direct application of Recursive Language Models. If you build on this agent, please cite the original paper:

```bibtex
@article{zhang2025recursive,
  title   = {Recursive Language Models},
  author  = {Zhang, Alex and Khattab, Omar and Kraska, Tim},
  journal = {arXiv preprint arXiv:2512.24601},
  year    = {2025},
  institution = {MIT CSAIL},
  url     = {https://arxiv.org/abs/2512.24601},
}
```

## Acknowledgments

Built on the [RDI-Foundation/agent-template](https://github.com/RDI-Foundation/agent-template). Evaluated on [SWE-bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os) by Scale AI. Architecture inspired by [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Khattab, Kraska — MIT CSAIL, 2025).
