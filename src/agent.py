"""
SWE-bench Pro Purple Agent — code-exploring single-shot patch generation.

Flow per task:
  1. Receive task: {instance_id, problem_statement, image_uri, base_commit, repo, hints}
  2. Pull image_uri and start a container (the SWE-bench instance image already
     contains the repo checked out at base_commit at /app).
  3. LLM call #1: given issue + file tree, list 1-5 candidate files most likely
     to contain the bug.
  4. cat those files inside the container.
  5. LLM call #2: given issue + actual file contents, produce a unified diff
     with REAL line numbers from the actual code.
  6. Validate via `git apply --check` inside the container.
  7. If invalid, one repair LLM call with the validation error.
  8. Return {"patch": <validated diff>}.
  9. Stop container.

This sidesteps the line-number-hallucination failure mode that pinned previous
runs at 0% — the model sees real code and writes real line numbers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import subprocess
import uuid

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

import litellm
from dotenv import load_dotenv

from messenger import Messenger

load_dotenv()
logger = logging.getLogger(__name__)

MODEL = "anthropic/claude-haiku-4-5-20251001"
MAX_TOKENS = 8192

# Resource caps
PROBLEM_MAX_CHARS = 4000
HINTS_MAX_CHARS = 800
FILE_TREE_MAX_LINES = 600
FILE_CONTENT_MAX_CHARS = 30000  # per file
NUM_CANDIDATE_FILES = 4
RETRY_MAX_ATTEMPTS = 12
RETRY_BACKOFF_CAP = 30

# Container settings
CONTAINER_LIFETIME_SEC = 1800  # 30 min ceiling per task
EXEC_TIMEOUT = 60


SYSTEM_FILE_ID = """You are an expert software engineer triaging a bug.

You will be shown an issue and a list of files in the repository. Your job is
to identify which files most likely contain the bug or need modification.

Respond with a single JSON object:
  {"files": ["path/to/file1.py", "path/to/file2.py"]}

Rules:
- Pick 1 to 4 paths. Fewer is better if you're confident.
- Use real paths from the file listing only. Do not invent paths.
- Prefer source files over test files.
- Output ONLY the JSON object, no prose."""


SYSTEM_PATCH_GEN = """You are an expert software engineer fixing a bug.

You will be shown an issue and the ACTUAL CONTENTS of the files most likely
to contain the bug. Your job is to produce a unified-diff patch that fixes
the issue, using REAL line numbers from the code shown.

Respond with a single JSON object:
  {"patch": "<unified diff>"}

The "patch" string must be a valid unified diff that `git apply` accepts:
- Use the exact file paths from the headers shown to you.
- Compute line numbers from the file content you've been shown — they must match.
- Include 3 lines of context above and below each change.
- Make minimal changes; don't break existing tests.
- Output ONLY the JSON object, no prose, no markdown fences.

Example:
{"patch": "diff --git a/path/to/file.py b/path/to/file.py\\n--- a/path/to/file.py\\n+++ b/path/to/file.py\\n@@ -10,7 +10,7 @@\\n context line\\n context line\\n context line\\n-old line\\n+new line\\n context line\\n context line\\n context line\\n"}"""


SYSTEM_PATCH_REPAIR = """You wrote a patch that failed `git apply --check`. The error is below.

Common issues:
- Wrong line numbers in the @@ header
- Context lines don't match the file exactly (whitespace, tab vs spaces)
- File path wrong

Look at the file content again, fix the patch, and respond with the SAME JSON
shape: {"patch": "<corrected unified diff>"}. Output ONLY the JSON object."""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated {len(text) - limit} chars ...]"


def _extract_json_obj(response: str, key: str):
    """Pull a JSON object containing the given key out of a model response."""
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


def _normalize_patch(patch: str) -> str:
    if not patch:
        return ""
    if not patch.endswith("\n"):
        patch += "\n"
    return patch


def _run(cmd: list[str], input_data: str | None = None, timeout: int = EXEC_TIMEOUT) -> tuple[int, str, str]:
    """Sync subprocess.run wrapper. Returns (returncode, stdout, stderr)."""
    try:
        p = subprocess.run(
            cmd,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
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
        image_uri = task.get("image_uri", "")
        repo = task.get("repo", "")
        logger.info(f"[{instance_id}] start; image={image_uri}")

        await updater.update_status(
            TaskState.working, new_agent_text_message(f"Starting {instance_id}...")
        )

        patch = ""
        container_name = f"swebench-{uuid.uuid4().hex[:12]}"

        try:
            if image_uri:
                started = await self._start_container(image_uri, container_name, updater)
                if started:
                    patch = await self._explore_and_patch(task, container_name, updater)
                else:
                    logger.warning(f"[{instance_id}] failed to start container, falling back")
                    patch = await self._blind_patch(task)
            else:
                patch = await self._blind_patch(task)
        except Exception as e:
            logger.exception(f"[{instance_id}] unexpected error: {e}")
        finally:
            await asyncio.to_thread(_run, ["docker", "rm", "-f", container_name], None, 30)

        patch = _normalize_patch(patch or "")
        logger.info(f"[{instance_id}] final patch length: {len(patch)}")

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=json.dumps({"patch": patch})))],
            name="Result",
        )

    async def _start_container(self, image_uri: str, container_name: str, updater: TaskUpdater) -> bool:
        await updater.update_status(
            TaskState.working, new_agent_text_message(f"Pulling image {image_uri[:60]}...")
        )
        rc, _, err = await asyncio.to_thread(_run, ["docker", "pull", image_uri], None, 600)
        if rc != 0:
            logger.error(f"docker pull failed: {err[:300]}")
            return False

        rc, _, err = await asyncio.to_thread(
            _run,
            [
                "docker", "run", "-d", "--rm",
                "--name", container_name,
                "--network", "none",
                image_uri,
                "sleep", str(CONTAINER_LIFETIME_SEC),
            ],
            None,
            60,
        )
        if rc != 0:
            logger.error(f"docker run failed: {err[:300]}")
            return False
        return True

    def _exec_in_container(self, container_name: str, cmd: list[str], timeout: int = EXEC_TIMEOUT) -> tuple[int, str, str]:
        return _run(["docker", "exec", container_name, *cmd], None, timeout)

    async def _explore_and_patch(self, task: dict, container_name: str, updater: TaskUpdater) -> str:
        instance_id = task.get("instance_id", "?")

        # Step 1: get repo file tree
        await updater.update_status(
            TaskState.working, new_agent_text_message(f"[{instance_id}] listing repo...")
        )
        # Many SWE-bench Pro images have the repo at /app; fall back to wherever .git lives
        rc, _, _ = await asyncio.to_thread(self._exec_in_container, container_name, ["test", "-d", "/app/.git"], 10)
        repo_root = "/app"
        if rc != 0:
            rc, out, _ = await asyncio.to_thread(
                self._exec_in_container,
                container_name,
                ["sh", "-c", "find / -maxdepth 4 -name '.git' -type d 2>/dev/null | head -1"],
                30,
            )
            if rc == 0 and out.strip():
                repo_root = out.strip().replace("/.git", "")
        logger.info(f"[{instance_id}] repo_root={repo_root}")

        rc, tree_out, _ = await asyncio.to_thread(
            self._exec_in_container,
            container_name,
            ["sh", "-c", f"cd {repo_root} && git ls-files 2>/dev/null | head -800"],
            60,
        )
        if rc != 0 or not tree_out.strip():
            # Fallback to find
            rc, tree_out, _ = await asyncio.to_thread(
                self._exec_in_container,
                container_name,
                ["sh", "-c", f"cd {repo_root} && find . -type f -not -path '*/node_modules/*' -not -path '*/.git/*' 2>/dev/null | head -800"],
                60,
            )
        file_lines = [l.strip() for l in tree_out.splitlines() if l.strip()]
        if not file_lines:
            logger.warning(f"[{instance_id}] empty file listing, fallback")
            return await self._blind_patch(task)
        file_tree = "\n".join(file_lines[:FILE_TREE_MAX_LINES])

        # Step 2: ask model to identify candidate files
        candidates = await self._identify_files(task, file_tree)
        if not candidates:
            return await self._blind_patch(task)

        # Step 3: read those files
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[{instance_id}] reading {len(candidates)} files..."),
        )
        file_contents: dict[str, str] = {}
        for path in candidates[:NUM_CANDIDATE_FILES]:
            safe_path = path.lstrip("./").replace("..", "")
            rc, content, _ = await asyncio.to_thread(
                self._exec_in_container,
                container_name,
                ["sh", "-c", f"cat {repo_root}/{safe_path}"],
                30,
            )
            if rc == 0 and content:
                file_contents[safe_path] = _truncate(content, FILE_CONTENT_MAX_CHARS)
        if not file_contents:
            logger.warning(f"[{instance_id}] no candidate file contents read")
            return await self._blind_patch(task)

        # Step 4: ask for the patch
        await updater.update_status(
            TaskState.working, new_agent_text_message(f"[{instance_id}] generating patch...")
        )
        patch = await self._generate_patch(task, file_contents)
        if not patch:
            return ""

        # Step 5: validate via git apply --check; one repair attempt
        ok, err = await self._validate_patch(container_name, repo_root, patch)
        if not ok:
            logger.info(f"[{instance_id}] first patch invalid, attempting repair: {err[:200]}")
            patch = await self._repair_patch(task, file_contents, patch, err)
            if patch:
                ok, err = await self._validate_patch(container_name, repo_root, patch)
                if not ok:
                    logger.info(f"[{instance_id}] repair also invalid: {err[:200]}")
                    # Submit anyway — green will silently fail to apply but at least we tried.
        return patch

    async def _validate_patch(self, container_name: str, repo_root: str, patch: str) -> tuple[bool, str]:
        cmd = [
            "docker", "exec", "-i", container_name,
            "sh", "-c", f"cd {repo_root} && git apply --check -",
        ]
        rc, _, err = await asyncio.to_thread(_run, cmd, patch, 60)
        return rc == 0, err

    async def _identify_files(self, task: dict, file_tree: str) -> list[str]:
        problem = _truncate((task.get("problem_statement") or "").strip(), PROBLEM_MAX_CHARS)
        hints = _truncate((task.get("hints") or task.get("hints_text") or "").strip(), HINTS_MAX_CHARS)
        repo = task.get("repo", "").strip()
        user = (
            f"Repository: {repo}\n\n"
            f"## Issue\n\n{problem}\n\n"
            + (f"## Hints\n\n{hints}\n\n" if hints else "")
            + f"## Repo file tree\n\n{file_tree}\n\n"
            "Which 1-4 files most likely contain the bug?"
        )
        resp = await self._call_model(SYSTEM_FILE_ID, user)
        files = _extract_json_obj(resp, "files")
        if isinstance(files, list):
            return [f for f in files if isinstance(f, str) and f.strip()][:NUM_CANDIDATE_FILES]
        return []

    async def _generate_patch(self, task: dict, file_contents: dict[str, str]) -> str:
        problem = _truncate((task.get("problem_statement") or "").strip(), PROBLEM_MAX_CHARS)
        hints = _truncate((task.get("hints") or task.get("hints_text") or "").strip(), HINTS_MAX_CHARS)
        repo = task.get("repo", "").strip()

        files_section = []
        for path, content in file_contents.items():
            files_section.append(f"### {path}\n```\n{content}\n```")
        files_block = "\n\n".join(files_section)

        user = (
            f"Repository: {repo}\n\n"
            f"## Issue\n\n{problem}\n\n"
            + (f"## Hints\n\n{hints}\n\n" if hints else "")
            + f"## File contents\n\n{files_block}\n\n"
            "Produce the unified-diff patch as JSON: {\"patch\": \"<diff>\"}"
        )
        resp = await self._call_model(SYSTEM_PATCH_GEN, user)
        patch = _extract_json_obj(resp, "patch")
        return patch if isinstance(patch, str) else ""

    async def _repair_patch(self, task: dict, file_contents: dict[str, str], bad_patch: str, error: str) -> str:
        problem = _truncate((task.get("problem_statement") or "").strip(), PROBLEM_MAX_CHARS)
        files_section = []
        for path, content in file_contents.items():
            files_section.append(f"### {path}\n```\n{content}\n```")
        files_block = "\n\n".join(files_section)

        user = (
            f"## Issue\n\n{problem}\n\n"
            f"## File contents\n\n{files_block}\n\n"
            f"## Failed patch\n\n```\n{bad_patch}\n```\n\n"
            f"## git apply error\n\n{_truncate(error, 1500)}\n\n"
            "Fix the patch and return JSON: {\"patch\": \"<corrected diff>\"}"
        )
        resp = await self._call_model(SYSTEM_PATCH_REPAIR, user)
        patch = _extract_json_obj(resp, "patch")
        return patch if isinstance(patch, str) else ""

    async def _blind_patch(self, task: dict) -> str:
        """Fallback when we can't pull/start the container — best-effort patch from issue alone."""
        problem = _truncate((task.get("problem_statement") or "").strip(), PROBLEM_MAX_CHARS)
        hints = _truncate((task.get("hints") or task.get("hints_text") or "").strip(), HINTS_MAX_CHARS)
        repo = task.get("repo", "").strip()
        user = (
            f"Repository: {repo}\n\n"
            f"## Issue\n\n{problem}\n\n"
            + (f"## Hints\n\n{hints}\n\n" if hints else "")
            + "Container access failed; produce best-effort patch from issue alone. "
            "Return JSON: {\"patch\": \"<diff>\"}"
        )
        resp = await self._call_model(SYSTEM_PATCH_GEN, user)
        patch = _extract_json_obj(resp, "patch")
        return patch if isinstance(patch, str) else ""

    async def _call_model(self, system_prompt: str, user_prompt: str) -> str:
        await asyncio.sleep(random.uniform(0, 3))  # spread shard fan-out

        system_msg = {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
            ],
        }
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
