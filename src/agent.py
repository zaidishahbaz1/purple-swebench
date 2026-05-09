"""
SWE-bench Pro Purple Agent — explore, patch, run tests, iterate.

Per-task flow:
  1. Pull image_uri, start a sidecar container (repo at /app, base_commit checked out)
  2. LLM call #1: from issue + repo file tree, pick 2-4 source files + 1-2 test files
  3. Read those files inside the container
  4. LLM call #2: generate unified-diff patch (model now sees expected behavior in tests)
  5. Validate via git apply --check; one repair attempt if invalid
  6. Apply patch and run pytest on the candidate test files
  7. If tests pass: done. Else: feed test output back to LLM, ask for revised patch
  8. Up to 2 revision passes (3 LLM patch generations total)
  9. Reset repo between iterations so each patch starts from base_commit
 10. Return the validated patch from the most successful attempt
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import subprocess
import sys
import uuid

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

import litellm
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

MODEL = "openai/gpt-4o-mini"
MAX_TOKENS = 4096

# Resource caps
PROBLEM_MAX_CHARS = 4000
HINTS_MAX_CHARS = 800
FILE_TREE_MAX_LINES = 600
FILE_CONTENT_MAX_CHARS = 20000
NUM_SOURCE_FILES = 4
NUM_TEST_FILES = 2
TEST_OUTPUT_MAX_CHARS = 4000
RETRY_MAX_ATTEMPTS = 12
RETRY_BACKOFF_CAP = 30
MAX_REVISION_PASSES = 2  # initial + 2 revisions = 3 patch attempts
EXEC_TIMEOUT = 60
TEST_RUN_TIMEOUT = 240
CONTAINER_LIFETIME_SEC = 1800


SYSTEM_FILE_ID = """You are an expert software engineer triaging a bug.

You will see an issue and the repository's file tree. Pick the files most likely
to contain the bug AND the test files that exercise the buggy behavior.

Respond with a single JSON object:
  {"source_files": ["path/to/src1.py", "path/to/src2.py"],
   "test_files":   ["path/to/test_foo.py"]}

Rules:
- 1-4 source files, 0-2 test files.
- Use real paths from the listing only — do not invent.
- Test files: prefer ones whose names match the issue's keywords or the source files.
- Output ONLY the JSON object."""


SYSTEM_PATCH_GEN = """You are an expert software engineer fixing a bug.

You will see:
- An issue description
- ACTUAL CONTENTS of the source files
- ACTUAL CONTENTS of the relevant test files (these encode the expected behavior)

Your job: produce a unified-diff patch that fixes the bug AND makes the relevant
tests pass.

Respond with a single JSON object: {"patch": "<unified diff>"}

Rules:
- The patch must be a valid unified diff that `git apply` accepts.
- Use the exact file paths shown to you. Compute line numbers from the file
  content shown — they MUST match.
- Include 3 lines of context above and below each change.
- Look at the test files: they tell you what behavior is expected. Make the
  source code conform to that expected behavior.
- Output ONLY the JSON object, no prose."""


SYSTEM_PATCH_REVISE = """Your previous patch was applied but the tests still fail.

You will see:
- The original issue
- The source and test file contents
- Your previous patch
- The test failure output

Identify why your fix didn't work and produce a revised patch.

Respond with the SAME JSON shape: {"patch": "<corrected unified diff>"}.
Output ONLY the JSON object."""


SYSTEM_PATCH_REPAIR = """Your previous patch failed `git apply --check`. Common issues:
- Wrong line numbers in the @@ header
- Context lines don't exactly match the file (whitespace, tabs vs spaces)
- File path wrong

Look at the file content again, fix the patch, return JSON: {"patch": "<corrected diff>"}."""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated {len(text) - limit} chars ...]"


def _extract_json_obj(response: str, key: str):
    if not response:
        return None
    response = response.strip()
    try:
        d = json.loads(response)
        if isinstance(d, dict) and key in d:
            return d[key]
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if fenced:
        try:
            d = json.loads(fenced.group(1))
            if isinstance(d, dict) and key in d:
                return d[key]
        except json.JSONDecodeError:
            pass
    obj = re.search(r"\{[\s\S]*\}", response)
    if obj:
        try:
            d = json.loads(obj.group(0))
            if isinstance(d, dict) and key in d:
                return d[key]
        except json.JSONDecodeError:
            pass
    return None


def _extract_files_obj(response: str) -> dict[str, list[str]]:
    """Extract {source_files, test_files} from response."""
    sources = _extract_json_obj(response, "source_files") or []
    tests = _extract_json_obj(response, "test_files") or []
    if not isinstance(sources, list):
        sources = []
    if not isinstance(tests, list):
        tests = []
    sources = [s for s in sources if isinstance(s, str) and s.strip()][:NUM_SOURCE_FILES]
    tests = [t for t in tests if isinstance(t, str) and t.strip()][:NUM_TEST_FILES]
    return {"source": sources, "tests": tests}


def _normalize_patch(patch: str) -> str:
    if not patch:
        return ""
    if not patch.endswith("\n"):
        patch += "\n"
    return patch


def _run(cmd: list[str], input_data: str | None = None, timeout: int = EXEC_TIMEOUT) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, input=input_data, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return -1, e.stdout or "", f"[timeout after {timeout}s]"
    except FileNotFoundError as e:
        return -1, "", str(e)


class Agent:
    def __init__(self):
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
        # The green sends the docker image under different field names depending on
        # version. Accept both. Also log the available keys so misnamings show up.
        image_uri = task.get("docker_image") or task.get("image_uri") or task.get("image") or ""
        logger.info(f"[{instance_id}] start; image={image_uri!r}; task_keys={list(task.keys())}")

        await updater.update_status(
            TaskState.working, new_agent_text_message(f"Starting {instance_id}")
        )

        patch = ""
        container_name = f"swebench-{uuid.uuid4().hex[:12]}"

        try:
            if image_uri and await self._start_container(image_uri, container_name, updater):
                patch = await self._iterate_to_patch(task, container_name, updater)
            else:
                patch = await self._blind_patch(task)
        except Exception as e:
            logger.exception(f"[{instance_id}] unexpected: {e}")
        finally:
            await asyncio.to_thread(_run, ["docker", "rm", "-f", container_name], None, 30)

        patch = _normalize_patch(patch or "")
        logger.info(f"[{instance_id}] final patch len={len(patch)}")

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=json.dumps({"patch": patch})))],
            name="Result",
        )

    async def _start_container(self, image_uri: str, name: str, updater: TaskUpdater) -> bool:
        # First: prove docker CLI is reachable inside our container.
        rc, vout, verr = await asyncio.to_thread(_run, ["docker", "version", "--format", "{{.Client.Version}} / {{.Server.Version}}"], None, 15)
        logger.info(f"docker version rc={rc} out={vout.strip()!r} err={verr[:200]!r}")
        if rc != 0:
            logger.error(f"docker CLI not reachable; falling back to blind patch")
            return False

        await updater.update_status(
            TaskState.working, new_agent_text_message(f"Pulling {image_uri[:50]}")
        )
        logger.info(f"docker pull image={image_uri!r}")
        rc, pout, err = await asyncio.to_thread(_run, ["docker", "pull", image_uri], None, 600)
        logger.info(f"docker pull rc={rc} stdout_tail={pout[-200:]!r} stderr_tail={err[-200:]!r}")
        if rc != 0:
            logger.error(f"docker pull failed: {err[:500]}")
            return False
        rc, _, err = await asyncio.to_thread(
            _run,
            ["docker", "run", "-d", "--rm", "--name", name, "--network", "none",
             image_uri, "sleep", str(CONTAINER_LIFETIME_SEC)],
            None, 60,
        )
        if rc != 0:
            logger.error(f"docker run failed: {err[:300]}")
            return False
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

    async def _list_files(self, name: str, repo_root: str) -> list[str]:
        rc, out, _ = await asyncio.to_thread(
            self._exec, name,
            ["sh", "-c", f"cd {repo_root} && git ls-files 2>/dev/null | head -800"],
            60,
        )
        if rc != 0 or not out.strip():
            rc, out, _ = await asyncio.to_thread(
                self._exec, name,
                ["sh", "-c", f"cd {repo_root} && find . -type f -not -path '*/node_modules/*' -not -path '*/.git/*' 2>/dev/null | head -800"],
                60,
            )
        return [l.strip() for l in out.splitlines() if l.strip()]

    async def _read_file(self, name: str, repo_root: str, path: str) -> str:
        safe = path.lstrip("./").replace("..", "")
        rc, content, _ = await asyncio.to_thread(
            self._exec, name, ["sh", "-c", f"cat {repo_root}/{safe}"], 30,
        )
        if rc != 0:
            return ""
        return _truncate(content, FILE_CONTENT_MAX_CHARS)

    async def _validate_patch(self, name: str, repo_root: str, patch: str) -> tuple[bool, str]:
        rc, _, err = await asyncio.to_thread(
            self._exec_input, name,
            ["sh", "-c", f"cd {repo_root} && git apply --check -"],
            patch, 60,
        )
        return rc == 0, err

    async def _apply_patch(self, name: str, repo_root: str, patch: str) -> tuple[bool, str]:
        rc, _, err = await asyncio.to_thread(
            self._exec_input, name,
            ["sh", "-c", f"cd {repo_root} && git apply -"],
            patch, 60,
        )
        return rc == 0, err

    async def _reset_repo(self, name: str, repo_root: str, base_commit: str):
        target = base_commit if base_commit else "HEAD"
        await asyncio.to_thread(
            self._exec, name,
            ["sh", "-c", f"cd {repo_root} && git reset --hard {target} 2>&1"],
            60,
        )

    async def _run_tests(self, name: str, repo_root: str, test_files: list[str]) -> tuple[bool, str]:
        """Try pytest on the test files. Return (all_passed, stdout+stderr)."""
        if not test_files:
            return True, ""  # no tests to run, treat as pass-through
        files_arg = " ".join(f"'{repo_root}/{tf.lstrip('./')}'" for tf in test_files)
        # -x stops on first fail; --tb=short shorter trace; -q quieter
        cmd = f"cd {repo_root} && python -m pytest {files_arg} -x --tb=short -q 2>&1 | tail -200"
        rc, out, err = await asyncio.to_thread(
            self._exec, name, ["sh", "-c", cmd], TEST_RUN_TIMEOUT,
        )
        combined = (out + err).strip()
        passed = rc == 0 and ("failed" not in combined.lower() or "0 failed" in combined.lower())
        return passed, _truncate(combined, TEST_OUTPUT_MAX_CHARS)

    async def _iterate_to_patch(self, task: dict, name: str, updater: TaskUpdater) -> str:
        instance_id = task.get("instance_id", "?")
        base_commit = task.get("base_commit", "")

        # Find repo + list files
        repo_root = await self._find_repo_root(name)
        await updater.update_status(
            TaskState.working, new_agent_text_message(f"[{instance_id}] listing repo")
        )
        all_files = await self._list_files(name, repo_root)
        if not all_files:
            return await self._blind_patch(task)
        file_tree = "\n".join(all_files[:FILE_TREE_MAX_LINES])

        # Identify source + test files
        candidates = await self._identify_files(task, file_tree)
        if not candidates["source"]:
            return await self._blind_patch(task)
        logger.info(f"[{instance_id}] source={candidates['source']} tests={candidates['tests']}")

        # Read all candidate files
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[{instance_id}] reading {len(candidates['source'])} src + {len(candidates['tests'])} tests"),
        )
        source_contents: dict[str, str] = {}
        for p in candidates["source"]:
            c = await self._read_file(name, repo_root, p)
            if c:
                source_contents[p] = c
        test_contents: dict[str, str] = {}
        for p in candidates["tests"]:
            c = await self._read_file(name, repo_root, p)
            if c:
                test_contents[p] = c
        if not source_contents:
            return await self._blind_patch(task)

        # Iteration loop
        best_patch = ""
        last_test_output = ""
        for revision in range(MAX_REVISION_PASSES + 1):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"[{instance_id}] generating patch (rev {revision})"),
            )

            if revision == 0:
                patch = await self._generate_patch(task, source_contents, test_contents)
            else:
                patch = await self._revise_patch(
                    task, source_contents, test_contents, best_patch, last_test_output,
                )
            if not patch:
                break

            # Validate patch syntactically
            ok, err = await self._validate_patch(name, repo_root, patch)
            if not ok:
                logger.info(f"[{instance_id}] rev{revision} apply --check failed: {err[:120]}")
                # one syntactic repair attempt
                patch = await self._repair_patch(task, source_contents, patch, err)
                if not patch:
                    continue
                ok, err = await self._validate_patch(name, repo_root, patch)
                if not ok:
                    # Save as best so far (might still apply with --3way somewhere)
                    if not best_patch:
                        best_patch = patch
                    continue

            best_patch = patch  # latest validating patch wins

            # If we have test files, actually run them
            if test_contents:
                await self._reset_repo(name, repo_root, base_commit)
                applied_ok, apply_err = await self._apply_patch(name, repo_root, patch)
                if not applied_ok:
                    logger.info(f"[{instance_id}] rev{revision} apply failed at runtime: {apply_err[:120]}")
                    last_test_output = f"git apply error:\n{apply_err}"
                    continue
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"[{instance_id}] running tests (rev {revision})"),
                )
                passed, output = await self._run_tests(name, repo_root, list(test_contents.keys()))
                logger.info(f"[{instance_id}] rev{revision} tests passed={passed}")
                if passed:
                    return patch
                last_test_output = output
            else:
                # No tests to verify with; submit current patch
                return patch

        return best_patch

    async def _identify_files(self, task: dict, file_tree: str) -> dict[str, list[str]]:
        problem = _truncate((task.get("problem_statement") or "").strip(), PROBLEM_MAX_CHARS)
        hints = _truncate((task.get("hints") or task.get("hints_text") or "").strip(), HINTS_MAX_CHARS)
        repo = task.get("repo", "").strip()
        user = (
            f"Repository: {repo}\n\n"
            f"## Issue\n\n{problem}\n\n"
            + (f"## Hints\n\n{hints}\n\n" if hints else "")
            + f"## Repo file tree\n\n{file_tree}\n\n"
            "Pick source files (where the bug lives) and test files (that exercise it)."
        )
        resp = await self._call_model(SYSTEM_FILE_ID, user)
        return _extract_files_obj(resp)

    async def _generate_patch(self, task: dict, sources: dict[str, str], tests: dict[str, str]) -> str:
        problem = _truncate((task.get("problem_statement") or "").strip(), PROBLEM_MAX_CHARS)
        hints = _truncate((task.get("hints") or task.get("hints_text") or "").strip(), HINTS_MAX_CHARS)
        repo = task.get("repo", "").strip()
        src_block = "\n\n".join(f"### {p}\n```\n{c}\n```" for p, c in sources.items())
        tst_block = "\n\n".join(f"### {p}\n```\n{c}\n```" for p, c in tests.items())
        user = (
            f"Repository: {repo}\n\n"
            f"## Issue\n\n{problem}\n\n"
            + (f"## Hints\n\n{hints}\n\n" if hints else "")
            + f"## Source files\n\n{src_block}\n\n"
            + (f"## Test files (expected behavior)\n\n{tst_block}\n\n" if tst_block else "")
            + 'Produce the patch as JSON: {"patch": "<diff>"}'
        )
        resp = await self._call_model(SYSTEM_PATCH_GEN, user)
        patch = _extract_json_obj(resp, "patch")
        return patch if isinstance(patch, str) else ""

    async def _revise_patch(self, task: dict, sources: dict[str, str], tests: dict[str, str],
                            prev_patch: str, test_output: str) -> str:
        problem = _truncate((task.get("problem_statement") or "").strip(), PROBLEM_MAX_CHARS)
        src_block = "\n\n".join(f"### {p}\n```\n{c}\n```" for p, c in sources.items())
        tst_block = "\n\n".join(f"### {p}\n```\n{c}\n```" for p, c in tests.items())
        user = (
            f"## Issue\n\n{problem}\n\n"
            f"## Source files\n\n{src_block}\n\n"
            + (f"## Test files\n\n{tst_block}\n\n" if tst_block else "")
            + f"## Your previous patch\n\n```\n{prev_patch}\n```\n\n"
            + f"## Test failure output\n\n```\n{_truncate(test_output, TEST_OUTPUT_MAX_CHARS)}\n```\n\n"
            'Produce a revised patch as JSON: {"patch": "<corrected diff>"}'
        )
        resp = await self._call_model(SYSTEM_PATCH_REVISE, user)
        patch = _extract_json_obj(resp, "patch")
        return patch if isinstance(patch, str) else ""

    async def _repair_patch(self, task: dict, sources: dict[str, str], bad: str, err: str) -> str:
        problem = _truncate((task.get("problem_statement") or "").strip(), PROBLEM_MAX_CHARS)
        src_block = "\n\n".join(f"### {p}\n```\n{c}\n```" for p, c in sources.items())
        user = (
            f"## Issue\n\n{problem}\n\n"
            f"## Source files\n\n{src_block}\n\n"
            f"## Failed patch\n\n```\n{bad}\n```\n\n"
            f"## git apply error\n\n```\n{_truncate(err, 1500)}\n```\n\n"
            'Fix and return JSON: {"patch": "<corrected diff>"}'
        )
        resp = await self._call_model(SYSTEM_PATCH_REPAIR, user)
        patch = _extract_json_obj(resp, "patch")
        return patch if isinstance(patch, str) else ""

    async def _blind_patch(self, task: dict) -> str:
        problem = _truncate((task.get("problem_statement") or "").strip(), PROBLEM_MAX_CHARS)
        hints = _truncate((task.get("hints") or task.get("hints_text") or "").strip(), HINTS_MAX_CHARS)
        repo = task.get("repo", "").strip()
        user = (
            f"Repository: {repo}\n\n"
            f"## Issue\n\n{problem}\n\n"
            + (f"## Hints\n\n{hints}\n\n" if hints else "")
            + 'Container access failed; best-effort patch from issue alone. Return JSON: {"patch": "<diff>"}'
        )
        resp = await self._call_model(SYSTEM_PATCH_GEN, user)
        patch = _extract_json_obj(resp, "patch")
        return patch if isinstance(patch, str) else ""

    async def _call_model(self, system_prompt: str, user_prompt: str) -> str:
        await asyncio.sleep(random.uniform(0, 5))  # spread shard fan-out

        # OpenAI auto-caches prefixes >1024 tokens server-side; no cache_control marker needed.
        system_msg = {"role": "system", "content": system_prompt}
        user_msg = {"role": "user", "content": user_prompt}

        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                completion = await asyncio.to_thread(
                    litellm.completion,
                    model=MODEL,
                    messages=[system_msg, user_msg],
                    temperature=0,
                    max_tokens=MAX_TOKENS,
                )
                return completion.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                if any(s in err for s in ("503", "overloaded", "429", "rate", "limit")):
                    base = min(RETRY_BACKOFF_CAP, 5 * (2 ** min(attempt, 3)))
                    wait = base + random.uniform(0, base * 0.3)
                    logger.warning(
                        f"retry {attempt + 1}/{RETRY_MAX_ATTEMPTS} in {wait:.1f}s: {err[:200]}"
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"LLM call failed (non-retryable): {err}")
                return ""
        logger.error("LLM call exhausted retries")
        return ""
