"""
SWE-bench Pro Purple Agent — RLM-style ReAct loop.

Adapted from our Terminal Bench agent (Recursive Language Models, MIT CSAIL,
arXiv 2512.24601). Each task runs to completion in one A2A turn:

  1. Pull image_uri, start a sidecar docker container with the repo at /app
  2. Init persistent REPL: context = [] (transcript), llm_query(prompt) sub-LLM
  3. Root LM (gpt-4o) drives a free-form ReAct loop with three tools:
       - bash(cmd, timeout=30): docker exec inside the sidecar; output appended
                                to context, truncated preview returned to chat
       - repl(code):           persistent in-process Python over `context` and
                                `llm_query`; for slicing, summarizing, grepping
                                large stored content without burning prompt tokens
       - final(patch):         submit a unified diff. We git-apply --check it
                                first; if invalid the model gets the error and
                                can try again
  4. Up to 30 LLM steps per task. Patch returned via add_artifact as {"patch":...}

This sidesteps every previous failure mode:
  - No fixed file-pick (model can grep/find/cat freely)
  - Long contents (file dumps, pytest output) live in REPL `context`, not prompt
  - llm_query lets the model summarize/extract from any chunk on demand
  - final() validates before submitting; the model gets to revise on apply errors
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import random
import re
import subprocess
import sys
import traceback
import uuid
from typing import Any

import litellm
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from dotenv import load_dotenv

from messenger import Messenger

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
    force=True,
)
logger = logging.getLogger(__name__)

ROOT_MODEL = os.environ.get("ROOT_MODEL", "openai/gpt-4o")
SUB_MODEL = os.environ.get("SUB_MODEL", "openai/gpt-4o-mini")
ROOT_MAX_TOKENS = 4096
SUB_MAX_TOKENS = 4096

# Resource caps per task
MAX_INNER_STEPS = 30
MAX_BASH = 30
MAX_REPL = 20
MAX_LLM_QUERY = 15
MAX_FINAL_ATTEMPTS = 3
LLM_OBS_TRUNCATE = 6000          # chat history shows truncated bash/repl output
SUB_PROMPT_MAX_CHARS = 400_000   # max input to llm_query

# Container settings
CONTAINER_LIFETIME_SEC = 1800
EXEC_TIMEOUT = 60
DOCKER_PULL_TIMEOUT = 600
TEST_RUN_TIMEOUT = 240

# LLM retry
RETRY_MAX_ATTEMPTS = 10
RETRY_BACKOFF_CAP = 30


SYSTEM_PROMPT = """You are an expert software engineer fixing a bug in a real open-source repository. Your job: produce a unified-diff patch that makes the failing tests pass without breaking existing ones.

You are inside a sandbox that has the repository checked out at /app, at the exact base commit. You have THREE tools and must call exactly ONE per step:

1. bash(command, timeout=30)
   Runs a shell command inside the repository container (cwd is the repo root).
   Use it to: ls, find, grep, cat, head, tail, git log, run pytest, anything.
   Output (stdout/stderr/exit_code) is appended to the `context` Python list.
   In chat you only see a TRUNCATED preview (~6KB). Full output is in context[-1].
   Default timeout 30s, max 300s. Increase for builds or long test runs.

2. repl(code)
   Runs Python in a persistent in-process REPL. Available globals:
     - context: a list of every bash/repl/llm_query record from this task
     - llm_query(prompt: str) -> str: a fast helper LLM (gpt-4o-mini, ~400K chars input)
     - json, re modules
   Use it to: slice/grep over context, summarize large outputs via llm_query,
   compute, build buffers. NEVER use repl to run shell commands — use bash.

3. final(patch)
   Submit your unified-diff patch. We will run `git apply --check` first.
   If invalid, you'll see the error and you must revise (call final again with
   a fixed patch). On valid, the task is done.

WORKING MEMORY MODEL (RLM — Recursive Language Models):
The chat history accumulates fast on hard tasks. **Long file contents and test
outputs are kept in the `context` REPL list, NOT in your chat prompt.** Pull
slices via repl. Summarize via llm_query.

Concrete worked example. Suppose you've just done:
  bash("cat src/middleware/auth.py")     # 60KB file, you only saw 6KB truncated
  bash("python -m pytest tests/test_auth.py -v")   # 30KB pytest output

To dig into the truncated file content WITHOUT another bash call, use repl:
```python
# Pull the auth.py content out of context (the cat result lives at index -2)
auth_src = context[-2]['stdout']
# Find the buggy function by searching for keywords from the issue
import re
for m in re.finditer(r'def (validate_token|check_session)', auth_src):
    start = m.start()
    print(auth_src[max(0, start-100):start+800])
    print('---')
```

Or summarize a huge pytest dump that's overflowing chat:
```python
test_log = context[-1]['stdout']
summary = llm_query(
    "Identify the failing assertion and the relevant traceback frames "
    "(file:line, function name) from this pytest output:\\n\\n" + test_log
)
print(summary)
```

These repl calls cost you ZERO bash budget and don't add long content to your
chat history. Use them aggressively.

DEBUGGING WORKFLOW (the way humans solve these):
- Read the issue carefully. Identify symptom, expected behavior, key terms.
- bash: find the relevant code with grep/find on those terms.
- bash: read suspect files. The model has training priors on common repos.
- bash: locate and read the failing test files (look in tests/, test/, etc.).
- Form a hypothesis. Patch.
- final(patch). If apply fails, fix and retry.
- After a successful apply, you can ALSO bash: run the failing tests to verify.
  Then re-final with corrections if needed.

DISCIPLINE:
- One tool per response.
- Don't re-read the same file. Cache findings via repl variables.
- Patches must be valid unified diffs with REAL line numbers from the actual
  files (which is why you should `cat` them first).
- Include 3 lines of context above and below each hunk.
- Make minimal changes; don't break existing tests.
- DO NOT submit empty patches. Commit to a hypothesis even if uncertain.
- When confident, call final. Don't loop forever.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the repository container (cwd = repo root). "
                "stdout/stderr/exit_code are appended to `context`; chat sees a truncated preview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {
                        "type": "integer",
                        "description": "Seconds; clamped to [1,300]. Default 30.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repl",
            "description": (
                "Execute Python in a persistent in-process REPL. Globals: `context` "
                "(transcript list), `llm_query(prompt)` (gpt-4o-mini sub-LLM), json, re. "
                "Do NOT run shell commands here — use bash."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final",
            "description": (
                "Submit the unified-diff patch. Will be validated with git apply --check; "
                "if invalid, you'll see the error and may call final again with a fix."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "Unified diff that `git apply` accepts.",
                    },
                },
                "required": ["patch"],
            },
        },
    },
]


def _truncate(s: str, n: int = LLM_OBS_TRUNCATE) -> str:
    if not s:
        return s
    if len(s) <= n:
        return s
    half = n // 2 - 50
    return f"{s[:half]}\n... [TRUNCATED {len(s) - 2 * half} chars; full in context[-1]] ...\n{s[-half:]}"


def _maybe_unescape_patch(patch: str) -> str:
    """Some models double-escape newlines in tool-call args, sending '\\n' (literal
    backslash-n) instead of real newlines. Detect and fix.
    """
    if not patch:
        return patch
    # Real newlines present? Already correct.
    if "\n" in patch:
        return patch
    # No real newlines but obvious diff structure with escapes? Unescape.
    if "\\n" in patch and ("diff --git" in patch or "@@" in patch or "--- " in patch):
        try:
            return patch.encode("utf-8").decode("unicode_escape")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return patch
    return patch


def _normalize_patch(patch: str) -> str:
    if not patch:
        return ""
    patch = _maybe_unescape_patch(patch)
    if not patch.endswith("\n"):
        patch += "\n"
    return patch


def _run(cmd: list[str], input_data: str | None = None, timeout: int = EXEC_TIMEOUT) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, input=input_data, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="replace")
        return -1, out or "", f"[timeout after {timeout}s]"
    except FileNotFoundError as e:
        return -1, "", str(e)


class Agent:
    def __init__(self) -> None:
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        try:
            task = json.loads(input_text)
            if not isinstance(task, dict):
                task = {"problem_statement": input_text}
        except json.JSONDecodeError:
            task = {"problem_statement": input_text}

        instance_id = task.get("instance_id", "?")
        image_uri = task.get("docker_image") or task.get("image_uri") or task.get("image") or ""
        base_commit = task.get("base_commit", "")
        repo = task.get("repo", "")
        logger.info(f"[{instance_id}] start image={image_uri!r} repo={repo!r}")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting {instance_id}"),
        )

        container_name = f"swebench-{uuid.uuid4().hex[:12]}"
        patch = ""

        try:
            if image_uri and await self._start_container(image_uri, container_name, updater, instance_id):
                repo_root = await self._find_repo_root(container_name)
                logger.info(f"[{instance_id}] repo_root={repo_root!r}")
                patch = await self._react_loop(task, container_name, repo_root, updater)
            else:
                logger.error(f"[{instance_id}] container start failed; emitting empty patch")
        except Exception as e:
            logger.exception(f"[{instance_id}] unexpected error: {e}")
        finally:
            await asyncio.to_thread(_run, ["docker", "rm", "-f", container_name], None, 30)

        patch = _normalize_patch(patch or "")
        logger.info(f"[{instance_id}] DONE patch_len={len(patch)}")

        # Match the convention other purple agents use: artifact name "patch",
        # raw diff as the text payload. Green's _extract_patch accepts either
        # this or our previous JSON-wrapped form, but this is the canonical shape.
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=patch))],
            name="patch",
        )

    # ------------------------------------------------------------------
    # Sidecar container

    async def _start_container(self, image_uri: str, name: str, updater: TaskUpdater, instance_id: str) -> bool:
        rc, vout, verr = await asyncio.to_thread(
            _run, ["docker", "version", "--format", "{{.Client.Version}}/{{.Server.Version}}"], None, 15
        )
        logger.info(f"[{instance_id}] docker version rc={rc} out={vout.strip()!r} err={verr[:200]!r}")
        if rc != 0:
            logger.error(f"[{instance_id}] docker not reachable: {verr[:300]}")
            return False

        await updater.update_status(
            TaskState.working, new_agent_text_message(f"[{instance_id}] pulling {image_uri[:60]}")
        )
        rc, _, err = await asyncio.to_thread(_run, ["docker", "pull", image_uri], None, DOCKER_PULL_TIMEOUT)
        if rc != 0:
            logger.error(f"[{instance_id}] docker pull failed: {err[-500:]}")
            return False

        # SWE-bench Pro images set ENTRYPOINT=/bin/bash, which means our
        # `sleep N` arg gets passed as a script name to bash and the
        # container exits immediately. Override the entrypoint to /bin/sh
        # so `sleep N` runs as the actual command. Also force linux/amd64
        # for arm64 hosts. Default network (many tests need it).
        rc, _, err = await asyncio.to_thread(
            _run,
            [
                "docker", "run", "-d", "--rm",
                "--name", name,
                "--platform", "linux/amd64",
                "--entrypoint", "/bin/sh",
                image_uri,
                "-c", f"sleep {CONTAINER_LIFETIME_SEC}",
            ],
            None,
            60,
        )
        if rc != 0:
            logger.error(f"[{instance_id}] docker run failed: {err[:500]}")
            return False
        logger.info(f"[{instance_id}] container started: {name}")

        # Sanity probe: run a trivial command inside and log result. If this
        # fails with "cannot execute binary file" we have a platform mismatch.
        # If it fails with permission/network errors we see them once instead
        # of buried inside every step's stderr.
        rc2, sout, serr = await asyncio.to_thread(
            self._exec, name, ["sh", "-c", "uname -a && id && ls /app | head -5"], 30,
        )
        logger.info(
            f"[{instance_id}] probe rc={rc2} stdout={sout[:300]!r} stderr={serr[:300]!r}"
        )
        return True

    def _exec(self, name: str, cmd: list[str], timeout: int = EXEC_TIMEOUT) -> tuple[int, str, str]:
        return _run(["docker", "exec", name, *cmd], None, timeout)

    def _exec_input(self, name: str, cmd: list[str], stdin: str, timeout: int = EXEC_TIMEOUT) -> tuple[int, str, str]:
        return _run(["docker", "exec", "-i", name, *cmd], stdin, timeout)

    async def _find_repo_root(self, name: str) -> str:
        rc, _, _ = await asyncio.to_thread(self._exec, name, ["test", "-d", "/app/.git"], 10)
        if rc == 0:
            return "/app"
        rc, out, _ = await asyncio.to_thread(
            self._exec, name,
            ["sh", "-c", "find / -maxdepth 4 -name '.git' -type d 2>/dev/null | head -1"],
            30,
        )
        if rc == 0 and out.strip():
            return out.strip().replace("/.git", "")
        return "/app"

    # ------------------------------------------------------------------
    # ReAct loop

    async def _react_loop(self, task: dict, container_name: str, repo_root: str, updater: TaskUpdater) -> str:
        instance_id = task.get("instance_id", "?")
        problem = (task.get("problem_statement") or "").strip()
        hints = (task.get("hints") or task.get("hints_text") or "").strip()
        repo = task.get("repo", "").strip()
        base_commit = task.get("base_commit", "")

        # Per-task LLM clients (sync for llm_query inside REPL, async for root)
        sync_client = self  # we use litellm.completion via asyncio.to_thread

        # Per-task state
        transcript: list[dict[str, Any]] = []
        bash_count = repl_count = llm_query_count = final_attempts = 0

        # llm_query function injected into REPL globals
        def _llm_query(prompt: str) -> str:
            nonlocal llm_query_count
            llm_query_count += 1
            if llm_query_count > MAX_LLM_QUERY:
                return "[llm_query budget exhausted]"
            text = prompt[:SUB_PROMPT_MAX_CHARS]
            try:
                resp = litellm.completion(
                    model=SUB_MODEL,
                    messages=[{"role": "user", "content": text}],
                    temperature=0,
                    max_tokens=SUB_MAX_TOKENS,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                return f"[llm_query error: {str(e)[:200]}]"

        repl_globals: dict[str, Any] = {
            "context": transcript,
            "llm_query": _llm_query,
            "json": json,
            "re": re,
        }

        def _exec_repl(code: str) -> tuple[str, str, str | None]:
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            exc: str | None = None
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    exec(code, repl_globals)
            except Exception:
                exc = traceback.format_exc()
            return stdout_buf.getvalue(), stderr_buf.getvalue(), exc

        # Build initial user message
        initial_user = (
            f"Repository: {repo}\n"
            f"Base commit: {base_commit}\n\n"
            f"## Issue\n\n{problem}\n"
        )
        if hints:
            initial_user += f"\n## Hints (from issue thread)\n\n{hints}\n"
        initial_user += (
            "\nThe repo is at " + repo_root + " inside a sandbox. "
            "Begin by exploring with bash. Call exactly one tool per response."
        )

        history: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_user},
        ]

        last_valid_patch = ""

        for step in range(MAX_INNER_STEPS):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"[{instance_id}] step {step}"),
            )

            try:
                completion = await asyncio.to_thread(
                    litellm.completion,
                    model=ROOT_MODEL,
                    messages=history,
                    temperature=0,
                    max_tokens=ROOT_MAX_TOKENS,
                    tools=TOOLS,
                    tool_choice="required",  # force a tool call every step
                )
            except Exception as e:
                err = str(e)
                err_l = err.lower()
                # Retryable transient errors. NOTE: "exceeded your current quota" is NOT
                # retryable (account is out of credits) — surface it as a hard error.
                is_quota_exhausted = "exceeded your current quota" in err_l
                is_retryable = (
                    not is_quota_exhausted
                    and any(s in err_l for s in (
                        "503", "overloaded", "429", "rate limit", "rate_limit",
                        "ratelimit", "tpm", "rpm", "tokens per minute", "requests per minute",
                    ))
                )
                if is_retryable:
                    base = min(RETRY_BACKOFF_CAP, 5 * (2 ** min(step, 3)))
                    wait = base + random.uniform(0, base * 0.3)
                    logger.warning(f"[{instance_id}] step {step} transient LLM error, waiting {wait:.1f}s: {err[:200]}")
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"[{instance_id}] step {step} LLM error (not retryable): {err[:300]}")
                break

            msg = completion.choices[0].message
            assistant_msg = {"role": "assistant", "content": msg.content or ""}
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            history.append(assistant_msg)

            if not tool_calls:
                # With tool_choice="required" this should be rare, but defend.
                logger.warning(f"[{instance_id}] step {step}: no tool_calls in response, content={(msg.content or '')[:200]!r}")
                history.append({"role": "user", "content": "Call exactly one of bash, repl, or final."})
                continue

            logger.info(
                f"[{instance_id}] step {step}: tool={tool_calls[0].function.name} "
                f"args={(tool_calls[0].function.arguments or '')[:200]!r}"
            )

            # We act on the first tool call only; respond to all to satisfy API.
            primary = tool_calls[0]
            extras = tool_calls[1:]

            name = primary.function.name
            raw_args = primary.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                logger.warning(f"[{instance_id}] step {step}: JSON parse failed for args: {e}; raw={raw_args[:300]!r}")
                args = {}

            tool_result_text = ""

            if name == "final":
                final_attempts += 1
                patch = _normalize_patch(str(args.get("patch", "")))
                if not patch:
                    tool_result_text = "Empty patch. You must submit a real unified diff."
                else:
                    ok, err = await self._validate_patch(container_name, repo_root, patch)
                    if ok:
                        last_valid_patch = patch
                        # Append tool results then bail with success
                        history.append({
                            "role": "tool", "tool_call_id": primary.id,
                            "content": "Patch validated by git apply --check. Submission accepted.",
                        })
                        for tc in extras:
                            history.append({"role": "tool", "tool_call_id": tc.id, "content": "[ignored]"})
                        logger.info(f"[{instance_id}] final accepted at step {step}, len={len(patch)}")
                        return last_valid_patch
                    tool_result_text = (
                        f"git apply --check FAILED:\n{_truncate(err, 2000)}\n\n"
                        "Fix the patch and call final again."
                    )
                    if final_attempts >= MAX_FINAL_ATTEMPTS:
                        # Save best effort and stop
                        last_valid_patch = last_valid_patch or patch
                        history.append({
                            "role": "tool", "tool_call_id": primary.id,
                            "content": tool_result_text + " [final_attempts exhausted]",
                        })
                        for tc in extras:
                            history.append({"role": "tool", "tool_call_id": tc.id, "content": "[ignored]"})
                        return last_valid_patch

            elif name == "bash":
                if bash_count >= MAX_BASH:
                    tool_result_text = "[bash budget exhausted; call final with your best patch]"
                else:
                    cmd = str(args.get("command", "")).strip()
                    if not cmd:
                        logger.warning(f"[{instance_id}] step {step}: bash got empty command; args={args!r}")
                        tool_result_text = "[empty command]"
                    else:
                        tmo = args.get("timeout", 30)
                        try:
                            tmo = max(1, min(int(tmo), 300))
                        except (TypeError, ValueError):
                            tmo = 30
                        bash_count += 1
                        logger.info(f"[{instance_id}] step {step}: executing bash {cmd[:200]!r}")
                        rc, out, err = await asyncio.to_thread(
                            self._exec_input, container_name,
                            ["sh", "-c", f"cd {repo_root} && {cmd}"], "", tmo,
                        )
                        # Log the actual error content so we can see WHY bash failed
                        if rc != 0 or err:
                            logger.warning(
                                f"[{instance_id}] step {step}: bash rc={rc} stderr={err[:500]!r}"
                            )
                        logger.info(f"[{instance_id}] step {step}: bash done rc={rc} stdout_len={len(out)} stderr_len={len(err)}")
                        # Store full in transcript
                        transcript.append({
                            "kind": "bash",
                            "command": cmd,
                            "exit_code": rc,
                            "stdout": out,
                            "stderr": err,
                        })
                        repl_globals["context"] = transcript
                        # Show truncated to LLM
                        tool_result_text = (
                            f"exit_code={rc}\n"
                            f"stdout (truncated; full in context[-1]['stdout']):\n{_truncate(out)}\n"
                            f"stderr (truncated):\n{_truncate(err)}"
                        )

            elif name == "repl":
                if repl_count >= MAX_REPL:
                    tool_result_text = "[repl budget exhausted]"
                else:
                    code = str(args.get("code", ""))
                    repl_count += 1
                    out, err, exc = await asyncio.to_thread(_exec_repl, code)
                    transcript.append({
                        "kind": "repl",
                        "code": code,
                        "stdout": out,
                        "stderr": err,
                        "exception": exc,
                    })
                    parts = []
                    if out:
                        parts.append(f"stdout:\n{_truncate(out)}")
                    if err:
                        parts.append(f"stderr:\n{_truncate(err)}")
                    if exc:
                        parts.append(f"exception:\n{exc}")
                    tool_result_text = "\n\n".join(parts) or "(no output)"

            else:
                tool_result_text = f"[unknown tool {name}]"

            history.append({
                "role": "tool",
                "tool_call_id": primary.id,
                "content": tool_result_text,
            })
            for tc in extras:
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[ignored: call exactly one tool per response]",
                })

        logger.info(f"[{instance_id}] step cap reached, returning last patch len={len(last_valid_patch)}")
        return last_valid_patch

    async def _validate_patch(self, container_name: str, repo_root: str, patch: str) -> tuple[bool, str]:
        rc, _, err = await asyncio.to_thread(
            self._exec_input, container_name,
            ["sh", "-c", f"cd {repo_root} && git apply --check -"],
            patch, 60,
        )
        return rc == 0, err
