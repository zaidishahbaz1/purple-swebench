"""
SWE-bench Purple Agent - Solves GitHub issues using LLM reasoning.

This agent:
1. Receives raw task data from Green Agent (issue description, repo info)
2. Builds LLM prompts with its own strategy
3. Explores codebase via bash commands
4. Generates patches to fix issues
5. Handles patch retry on failure
"""

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
import litellm

from dotenv import load_dotenv
import asyncio
import json
import re

load_dotenv()


def fix_json_newlines(text: str) -> str:
    """Fix unescaped newlines inside JSON string values."""
    result = []
    in_string = False
    escape = False
    i = 0

    while i < len(text):
        char = text[i]

        if escape:
            result.append(char)
            escape = False
            i += 1
            continue

        if char == '\\':
            result.append(char)
            escape = True
            i += 1
            continue

        if char == '"':
            result.append(char)
            in_string = not in_string
            i += 1
            continue

        if in_string and char == '\n':
            # Replace literal newline with escaped newline
            result.append('\\n')
            i += 1
            continue

        if in_string and char == '\r':
            # Skip carriage returns
            i += 1
            continue

        if in_string and char == '\t':
            # Replace literal tab with escaped tab
            result.append('\\t')
            i += 1
            continue

        result.append(char)
        i += 1

    return ''.join(result)


def fix_triple_quotes(text: str) -> str:
    """Fix Python-style triple quotes in JSON (Claude sometimes uses these)."""
    # Replace """ with " and handle the content properly
    # Pattern: "key": """content""" -> "key": "content"
    pattern = r':\s*"""([\s\S]*?)"""'

    def replace_triple(match):
        content = match.group(1)
        # Only escape unescaped quotes, don't double-escape backslashes
        content = content.replace('\r', '').replace('\n', '\\n').replace('\t', '\\t')
        # Escape unescaped double quotes (not already escaped)
        content = re.sub(r'(?<!\\)"', '\\"', content)
        return f': "{content}"'

    return re.sub(pattern, replace_triple, text)


def extract_json(text: str) -> str:
    """Extract first valid JSON object from text, handling markdown and extra content."""
    text = text.strip()

    # Remove ```json ... ``` or ``` ... ``` wrappers
    pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(pattern, text)
    if match:
        text = match.group(1).strip()

    # Fix triple quotes before processing
    text = fix_triple_quotes(text)

    # Find the first complete JSON object by matching braces
    if not text.startswith('{'):
        # Try to find JSON object in the text
        start = text.find('{')
        if start == -1:
            return text
        text = text[start:]

    # Balance braces to find complete JSON object
    depth = 0
    in_string = False
    escape = False
    end = 0

    for i, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == '\\' and in_string:
            escape = True
            continue
        if char == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end > 0:
        text = text[:end]

    # Fix unescaped newlines in string values
    return fix_json_newlines(text)


# Response format keys
RESPONSE_KEY = "action"
CONTENT_KEY = "content"

# System prompt for the LLM
SYSTEM_PROMPT = """You are an expert software engineer tasked with fixing bugs in a repository.

You will receive a GitHub issue description and must fix it by exploring the codebase and/or submitting a patch.

## Response Format

You MUST respond with a single JSON object in one of these formats:

1. To explore the codebase or fetch context (run a bash command):
   - Format: {"action": "bash", "content": "<shell command>"}
   - Example: {"action": "bash", "content": "ls sklearn/metrics"}
   - Outputs from the command will be returned to you.
   - Only read-only commands are allowed; do not modify files yet.

2. To test changes in debug mode (changes are NOT saved):
   - Format: {"action": "debug", "content": "<bash command>"}
   - Example: {"action": "debug", "content": "sed -i 's/old/new/' file.py && python -m pytest tests/"}
   - In debug mode, you can modify files (vim, sed, echo >>, etc.) and run tests.
   - All changes are rolled back after the command completes.
   - Use this to experiment with fixes before submitting a final patch.

3. To submit your fix (unified diff format):
   - Format: {"action": "patch", "content": "<unified diff>"}
   - Example 1 (single file patch): {"action": "patch", "content": "
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,7 +10,7 @@
 context line
-old line to remove
+new line to add
 context line
"}
    - Example 2 (multi files patch): {"action": "patch", "content": "
diff --git a/pkg/foo.py b/pkg/foo.py
--- a/pkg/foo.py
+++ b/pkg/foo.py
@@ -42,7 +42,9 @@ def f(x):
     if x is None:
         return 0
-    return bar(x)
+    y = normalize(x)
+    return bar(y)

diff --git a/pkg/bar.py b/pkg/bar.py
--- a/pkg/bar.py
+++ b/pkg/bar.py
@@ -10,6 +10,8 @@ def normalize(x):
     if x < 0:
         raise ValueError()
+    if x == 0:
+        return 1
     return x
"}
   - You may generate the patch as a minimal diff; it will be executed for you and output returned to you.

## Important Rules

- Respond with ONLY the JSON object. No explanations, no markdown, no extra text.
- Use bash commands to explore: ls, cat, grep, find, git log, git diff, etc.
- The codebase is READ-ONLY. You cannot modify files with bash commands.
- When ready, submit a patch in unified diff format (like `git diff` output).
- If your patch fails, you'll receive the error. Analyze it and try again.

## Execution Output Format

After every bash command that you submit, you'll receive its execution output as follows:
{
  "cwd": "/workspace/repo/current/directory",
  "stdout": "command output...",
  "stderr": "any errors..."
}

Use 'cwd' to track your current location. You can use `cd` to navigate within the repo but never outside it.
"""


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.messages = []
        self.task_data = None

    def _format_patch(self, patch_text: str) -> str:
        # append '\n' at the end of patch
        return patch_text if patch_text.endswith('\n') else patch_text + "\n"
    
    def _parse_task_data(self, input_text: str) -> dict | None:
        """Parse raw task data from Green Agent."""
        try:
            return json.loads(input_text)
        except json.JSONDecodeError:
            return None

    def _parse_bash_result(self, input_text: str) -> dict | None:
        """Parse bash result from Green Agent."""
        try:
            data = json.loads(input_text)
            if "cwd" in data and ("stdout" in data or "stderr" in data):
                return data
            return None
        except json.JSONDecodeError:
            return None

    def _parse_patch_failure(self, input_text: str) -> dict | None:
        """Parse patch failure feedback from Green Agent."""
        try:
            data = json.loads(input_text)
            if data.get("patch_failed"):
                return data
            return None
        except json.JSONDecodeError:
            return None

    def _parse_error_feedback(self, input_text: str) -> dict | None:
        """Parse error feedback from Green Agent."""
        try:
            data = json.loads(input_text)
            if "error" in data:
                return data
            return None
        except json.JSONDecodeError:
            return None

    def _build_initial_prompt(self, task_data: dict) -> str:
        """Build the initial user prompt from task data."""
        problem_statement = task_data.get(
            "problem_statement", "No description provided"
        )
        hints = task_data.get("hints_text", "")
        cwd = task_data.get("cwd", "unknown")
        python_version = task_data.get("python_version", "3.9")
        fail_to_pass = task_data.get("fail_to_pass", [])

        prompt = f"""Current Working Directory (cwd): {cwd}
Python Version: {python_version}

## Issue Description

{problem_statement}

"""
        if hints:
            prompt += f"""## Additional Context (from issue discussion)

{hints}

"""
        if fail_to_pass:
            prompt += f"""## Tests to Fix

The following tests are currently failing and should pass after your fix:
{chr(10).join(f'- {test}' for test in fail_to_pass[:5])}
{'...' if len(fail_to_pass) > 5 else ''}

"""

        return prompt

    def _format_bash_result_for_llm(self, result: dict) -> str:
        """Format bash result for the LLM."""
        cwd = result.get("cwd", "/workspace/repo")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        if stdout:
            # Truncate very long outputs
            if len(stdout) > 8000:
                stdout = stdout[:8000] + "\n... [output truncated]"

        return json.dumps({'cwd': cwd, 'stdout': stdout, 'stderr': stderr}, indent=2)

    def _format_patch_failure_for_llm(self, result: dict) -> str:
        """Format patch failure for the LLM to retry."""
        stderr = result.get("stderr", "Unknown error")
        cwd = result.get("cwd", "/workspace/repo")

        return f"""Patch application FAILED.

Current Working Directory: {cwd}

Error details:
{stderr}

Please analyze the error and submit a corrected patch. Common issues:
- Incorrect file path in the diff header
- Wrong line numbers (the code may have changed)
- Missing context lines
- Whitespace issues

Try using `cat` to view the exact current content of the file, then create a new patch."""

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Handle messages from Green Agent.

        Messages can be:
        1. Initial task data (JSON with problem_statement, repo, etc.)
        2. Bash command result (JSON with cwd, stdout, stderr)
        3. Patch failure feedback (JSON with patch_failed, stderr)
        4. Error feedback (JSON with error message)
        """
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Processing...")
        )

        # Determine message type and format appropriately
        user_content = None

        # Check if this is initial task data
        task_data = self._parse_task_data(input_text)
        if task_data and "problem_statement" in task_data:
            # Initial task - build prompt and start fresh conversation
            self.task_data = task_data
            self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            user_content = self._build_initial_prompt(task_data)

        # Check if this is a bash result
        elif bash_result := self._parse_bash_result(input_text):
            user_content = self._format_bash_result_for_llm(bash_result)

        # Check if this is a patch failure
        elif patch_failure := self._parse_patch_failure(input_text):
            user_content = self._format_patch_failure_for_llm(patch_failure)

        # Check if this is an error feedback
        elif error_feedback := self._parse_error_feedback(input_text):
            user_content = f"Error: {error_feedback.get('message', error_feedback.get('error', 'Unknown error'))}\n\nPlease respond with valid JSON: {{\"action\": \"bash\"| \"debug\" | \"patch\", \"content\": \"...\"}}"

        # Fallback - treat as raw text
        else:
            user_content = input_text

        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_content})

        # Call LLM
        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )

        try:
            print("green response > ", self.messages[-1]["content"])

            # Retry with exponential backoff for overloaded/rate-limited APIs
            max_retries = 5
            response = None
            for attempt in range(max_retries):
                try:
                    completion = litellm.completion(
                        model="gemini/gemini-2.5-flash", #gemini/gemini-2.5-flash-lite",
                        messages=self.messages,
                        response_format={"type": "json_object"},
                    )
                    response = completion.choices[0].message.content
                    print(f"\n[DEBUG] Raw LLM response: {response[:200]}...")
                    break
                except Exception as retry_err:
                    err_str = str(retry_err)
                    if "503" in err_str or "overloaded" in err_str.lower() or "429" in err_str or "rate" in err_str.lower():
                        wait_time = (2 ** attempt) * 5  # 5, 10, 20, 40, 80 seconds
                        print(f"\n[DEBUG] Retry {attempt + 1}/{max_retries}: waiting {wait_time}s due to: {err_str[:100]}")
                        await asyncio.sleep(wait_time)
                    else:
                        raise retry_err

            if response is None:
                raise Exception("Max retries exceeded - API still unavailable")

        except Exception as e:
            # Handle LLM errors
            print(f"\n[DEBUG] LLM error: {e}")
            await updater.add_artifact(
                name="error",
                parts=[Part(root=TextPart(text=f"LLM error: {str(e)}"))],
            )
            return

        # Add assistant response to conversation history
        self.messages.append({"role": "assistant", "content": response})

        # Parse and return response
        try:
            # Extract first valid JSON object from response
            clean_response = extract_json(response)
            response_json = json.loads(clean_response)

            action = response_json.get(RESPONSE_KEY, "unknown")
            content = response_json.get(CONTENT_KEY, "")

            if action == "patch":
                content = self._format_patch(content)

            print(f"\n\npurple response > action={action}, content={content}")
            

            await updater.add_artifact(
                name=action,
                parts=[Part(root=TextPart(text=content))],
            )

        except json.JSONDecodeError as e:
            print(f"\n[DEBUG] === JSON DECODE ERROR ===")
            print(f"[DEBUG] Error: {e}")
            print(f"[DEBUG] Error position: line {e.lineno}, col {e.colno}, char {e.pos}")
            print(f"[DEBUG] Problematic area: {repr(clean_response[max(0,e.pos-20):e.pos+20]) if 'clean_response' in dir() else 'N/A'}")
            print(f"\n[DEBUG] JSON parse error: {e}")
            print(f"[DEBUG] Response was: {response[:300]}")

            # Fallback: try to interpret non-JSON response
            response_stripped = response.strip()

            # Check if it looks like a bash command (common patterns)
            bash_patterns = [
                r'^(ls|cat|grep|find|cd|pwd|echo|head|tail|wc|git|python|pip)\b',
                r'^[./]',  # Starts with ./ or /
            ]
            is_likely_bash = any(re.match(p, response_stripped) for p in bash_patterns)

            # Check if it looks like a patch
            is_likely_patch = response_stripped.startswith('diff --git') or response_stripped.startswith('--- ')

            if is_likely_patch:
                print(f"\n[DEBUG] Fallback: treating as patch")
                await updater.add_artifact(
                    name="patch",
                    parts=[Part(root=TextPart(text=self._format_patch(response_stripped)))],
                )
            elif is_likely_bash:
                print(f"\n[DEBUG] Fallback: treating as bash command")
                await updater.add_artifact(
                    name="bash",
                    parts=[Part(root=TextPart(text=response_stripped))],
                )
            else:
                # Send error feedback to help LLM correct itself
                await updater.add_artifact(
                    name="error",
                    parts=[
                        Part(root=TextPart(text=f"Invalid JSON response. Please respond with valid JSON: {{\"action\": \"bash\"|\"patch\", \"content\": \"...\"}}"))
                    ],
                )
