#!/usr/bin/env python3
import argparse
import atexit
import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Allow overriding the OpenHands repo used for agent code via OPENHANDS_REPO_DIR.
# When set, that directory's openhands/ and third_party/ are imported instead of
# this script's own, enabling evaluation of modified agent code.
REPO_DIR = os.environ.get("OPENHANDS_REPO_DIR", SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)
if REPO_DIR != SCRIPT_DIR:
    sys.path.insert(1, SCRIPT_DIR)
    print(f"Agent code: {REPO_DIR} (override via OPENHANDS_REPO_DIR)")
else:
    print(f"Agent code: {REPO_DIR}")

# Ensure ~/.modal.toml exists for SWE-bench Pro evaluation (Phase 2).
_modal_toml = Path.home() / ".modal.toml"
if not _modal_toml.exists():
    _token_id = os.environ.get("MODAL_TOKEN_ID", "")
    _token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
    if _token_id and _token_secret:
        _modal_toml.write_text(
            f"[default]\ntoken_id = \"{_token_id}\"\ntoken_secret = \"{_token_secret}\"\n"
        )

_runtimes_to_cleanup = []

def _silent_cleanup():
    for runtime in _runtimes_to_cleanup:
        if hasattr(runtime, '_sandbox') and runtime._sandbox is not None:
            pass

atexit.register(_silent_cleanup)

TEMP_DIR = os.path.join(SCRIPT_DIR, ".tmp")
_RUN_CACHE_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache", _RUN_CACHE_ID)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TMPDIR"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
tempfile.tempdir = TEMP_DIR

from costs import InstanceCost, collect as collect_costs, aggregate as aggregate_costs, write as write_costs
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, OpenHandsConfig, SandboxConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.logger import openhands_logger as _oh_logger
from openhands.core.main import run_controller
from openhands.events import EventStream
from openhands.events.action import CmdRunAction, FileEditAction, FileReadAction, MessageAction
from openhands.events.observation import CmdOutputObservation, ErrorObservation, FileReadObservation
from openhands.llm.llm_registry import LLMRegistry
from openhands.resolver.utils import codeact_user_response
from openhands.runtime.base import Runtime
from openhands.storage import InMemoryFileStore
from third_party.runtime.impl.modal.modal_runtime import ModalRuntime

_oh_resolved = os.path.dirname(os.path.abspath(sys.modules['openhands'].__file__))
_oh_expected = os.path.join(REPO_DIR, "openhands")
if os.path.realpath(_oh_resolved) != os.path.realpath(_oh_expected):
    print(f"WARNING: openhands loaded from {_oh_resolved}, expected {_oh_expected}")
    sys.exit(1)
print(f"openhands loaded from: {_oh_resolved}")

logging.getLogger("modal").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

_oh_logger.setLevel(logging.ERROR)

logger = logging.getLogger("swebench_pro_eval")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    "CodeActAgent": codeact_user_response,
}

MODEL = "openai/gpt-5-mini"
MODEL_NAME = "swebench_pro_eval"

SWEBENCH_PRO_DATASET = "ScaleAI/SWE-bench_Pro"
SWEBENCH_PRO_REPO_DIR = os.path.join(SCRIPT_DIR, "third_party", "swebench_pro")


def remove_binary_diffs(patch_text: str) -> str:
    lines = patch_text.splitlines()
    cleaned_lines = []
    block = []
    is_binary_block = False

    for line in lines:
        if line.startswith('diff --git '):
            if block and not is_binary_block:
                cleaned_lines.extend(block)
            block = [line]
            is_binary_block = False
        elif 'Binary files' in line:
            is_binary_block = True
            block.append(line)
        else:
            block.append(line)

    if block and not is_binary_block:
        cleaned_lines.extend(block)
    return '\n'.join(cleaned_lines)


def remove_binary_files_from_git() -> str:
    return """
    for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do
        if [ -f "$file" ]; then
            filetype=$(file "$file")
            if echo "$filetype" | grep -qE "(ELF|Mach-O|PE32|executable.*binary)" || git check-attr binary "$file" | grep -q "binary: set"; then
                git rm -f "$file" 2>/dev/null || rm -f "$file"
                echo "Removed: $file"
            fi
        fi
    done
    """.strip()


def _load_pro_dataset() -> dict[str, pd.Series]:
    dataset = load_dataset(SWEBENCH_PRO_DATASET, split="test")
    by_id: dict[str, pd.Series] = {}
    for item in dataset:
        by_id[item["instance_id"]] = pd.Series(item)
    return by_id


def get_swebench_instances(num_instances: int, seed: int = 42) -> list[pd.Series]:
    """Select instances via deterministic stratified sampling across languages.

    Slots are filled by round-robin across languages (shuffled within each
    language group with a fixed seed) so that every language gets roughly
    equal representation.
    """
    import random

    logger.info(f"Loading {num_instances} instances from SWE-bench Pro dataset...")
    by_id = _load_pro_dataset()

    by_lang: dict[str, list[pd.Series]] = {}
    for inst in by_id.values():
        lang = getattr(inst, "repo_language", "unknown")
        by_lang.setdefault(lang, []).append(inst)

    rng = random.Random(seed)
    for lang_instances in by_lang.values():
        rng.shuffle(lang_instances)

    instances: list[pd.Series] = []
    lang_keys = sorted(by_lang.keys())
    lang_iters = {lang: iter(by_lang[lang]) for lang in lang_keys}
    while len(instances) < num_instances and lang_iters:
        exhausted = []
        for lang in lang_keys:
            if lang not in lang_iters:
                continue
            if len(instances) >= num_instances:
                break
            try:
                instances.append(next(lang_iters[lang]))
            except StopIteration:
                exhausted.append(lang)
        for lang in exhausted:
            del lang_iters[lang]

    repos: dict[str, int] = {}
    languages: dict[str, int] = {}
    for inst in instances:
        repos[inst.repo] = repos.get(inst.repo, 0) + 1
        lang = getattr(inst, "repo_language", "unknown")
        languages[lang] = languages.get(lang, 0) + 1

    logger.info(f"Loaded {len(instances)} instances across {len(repos)} repos:")
    for repo, count in sorted(repos.items(), key=lambda x: -x[1]):
        logger.info(f"  {repo}: {count}")
    logger.info(f"Language distribution:")
    for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
        logger.info(f"  {lang}: {count}")
    return instances


def get_workspace_path(instance: pd.Series) -> str:
    return "/app"


def get_container_image(instance: pd.Series) -> str:
    tag = instance.dockerhub_tag
    return f"docker.io/jefzda/sweap-images:{tag}"


def initialize_runtime(runtime: Runtime, instance: pd.Series, instance_id: str):
    workspace_path = get_workspace_path(instance)
    base_commit = instance["base_commit"]

    action = CmdRunAction(
        command=f"""echo 'export SWE_INSTANCE_ID={instance["instance_id"]}' >> ~/.bashrc && echo "alias git='git --no-pager'" >> ~/.bashrc && git config --global core.pager "" && git config --global diff.binary false"""
    )
    action.set_hard_timeout(600)
    runtime.run_action(action)

    action = CmdRunAction(command="""export USER=$(whoami); echo USER=${USER}""")
    action.set_hard_timeout(600)
    runtime.run_action(action)

    action = CmdRunAction(command=f"cd {workspace_path} && git reset --hard {base_commit} && git checkout {base_commit}")
    action.set_hard_timeout(600)
    runtime.run_action(action)

    action = CmdRunAction(command="git remote remove origin 2>/dev/null || true; git remote remove upstream 2>/dev/null || true")
    action.set_hard_timeout(600)
    runtime.run_action(action)


def _run_action_with_retry(runtime: Runtime, action, retries: int = 3, backoff: float = 2.0):
    for attempt in range(retries):
        try:
            return runtime.run_action(action)
        except Exception as e:
            if attempt == retries - 1:
                raise
            err = str(e).lower()
            if not any(p in err for p in ("ssl", "eof", "timed out", "broken pipe", "connection")):
                raise
            time.sleep(backoff * (attempt + 1))


def _save_partial_trajectory(event_stream: EventStream, output_dir: str, attempt: int):
    """Save whatever events exist when the agent is killed by a timeout."""
    from openhands.events.serialization.event import event_to_dict
    try:
        events = list(event_stream.get_events())
        if events:
            trajectory = [event_to_dict(e) for e in events]
            path = os.path.join(output_dir, f"trajectory_attempt{attempt}.json")
            with open(path, "w") as f:
                json.dump(trajectory, f, indent=4)
            logger.info(f"Saved partial trajectory ({len(events)} events) to {path}")
    except Exception as e:
        logger.warning(f"Failed to save partial trajectory: {e}")


def complete_runtime(runtime: Runtime, instance: pd.Series) -> dict:
    workspace_path = get_workspace_path(instance)

    action = CmdRunAction(command=f"cd {workspace_path}")
    action.set_hard_timeout(600)
    _run_action_with_retry(runtime, action)

    action = CmdRunAction(command='git config --global core.pager ""')
    action.set_hard_timeout(600)
    _run_action_with_retry(runtime, action)

    action = CmdRunAction(command='find . -type d -name .git -not -path "./.git"')
    action.set_hard_timeout(600)
    obs = _run_action_with_retry(runtime, action)

    git_dirs = [p for p in obs.content.strip().split("\n") if p]
    for git_dir in git_dirs:
        action = CmdRunAction(command=f'rm -rf "{git_dir}"')
        action.set_hard_timeout(600)
        _run_action_with_retry(runtime, action)

    action = CmdRunAction(command="git add -A")
    action.set_hard_timeout(600)
    _run_action_with_retry(runtime, action)

    action = CmdRunAction(command=remove_binary_files_from_git())
    action.set_hard_timeout(600)
    _run_action_with_retry(runtime, action)

    action = CmdRunAction(
        command=f'git diff --no-color --cached {instance["base_commit"]} > patch.diff'
    )
    action.set_hard_timeout(600)
    _run_action_with_retry(runtime, action)

    action = FileReadAction(path="patch.diff")
    action.set_hard_timeout(600)
    obs = _run_action_with_retry(runtime, action)

    if isinstance(obs, FileReadObservation):
        git_patch = obs.content
    elif isinstance(obs, ErrorObservation):
        action = CmdRunAction(command="cat patch.diff")
        action.set_hard_timeout(600)
        obs = _run_action_with_retry(runtime, action)
        git_patch = obs.content
    else:
        git_patch = ""

    if not git_patch.strip():
        action = CmdRunAction(
            command=f'git diff --no-color {instance["base_commit"]} HEAD'
        )
        action.set_hard_timeout(600)
        obs = _run_action_with_retry(runtime, action)
        if isinstance(obs, CmdOutputObservation) and obs.content.strip():
            logger.info("Recovered patch from committed changes (HEAD vs base)")
            git_patch = obs.content

    git_patch = remove_binary_diffs(git_patch)
    return {"git_patch": git_patch}


def get_instruction(instance: pd.Series) -> str:
    repo_path = get_workspace_path(instance)
    lang = getattr(instance, "repo_language", "unknown")

    instruction = f"""I have access to a code repository in the directory {repo_path}. The primary language is {lang}. You can explore and modify files using the available tools. Consider the following issue description:

<issue>
{instance["problem_statement"]}
</issue>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue> are met?
I've already taken care of all changes to any of the test files described in the <issue>. This means you DON'T have to modify the testing logic or any of the tests in any way!
The development environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the {repo_path} directory to ensure the <issue> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets, express in words any best practices or conventions in them.
   1.2 Highlight message errors, method names, variables, file names, stack traces, and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Highlight any best practices to take into account when testing and fixing the issue.

Phase 2. EXPLORATION: find the files that are related to the problem and possible solutions
   2.1 Use `grep` or `find` to search for relevant methods, classes, keywords and error messages.
   2.2 Identify all files related to the problem statement.
   2.3 Propose the methods and files to fix the issue and explain why.
   2.4 From the possible file locations, select the most likely location to fix the issue.

Phase 3. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue.
   3.1 Look at existing test files in the repository to understand the test format/structure.
   3.2 Create a minimal reproduction script that reproduces the located issue.
   3.3 Run the reproduction script to confirm you are reproducing the issue.
   3.4 Adjust the reproduction script as necessary.

Phase 4. FIX ANALYSIS: state clearly the problem and how to fix it
   4.1 State clearly what the problem is.
   4.2 State clearly where the problem is located.
   4.3 State clearly how the test reproduces the issue.
   4.4 State clearly the best practices to take into account in the fix.
   4.5 State clearly how to fix the problem.

Phase 5. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution.
   5.1 Make minimal, focused changes to fix the issue.

Phase 6. VERIFICATION: Test your implementation thoroughly.
   6.1 Run your reproduction script to verify the fix works.
   6.2 Add edge cases to your test script to ensure comprehensive coverage.
   6.3 Run existing tests related to the modified code to ensure you haven't broken anything.

Phase 7. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit {instance["base_commit"]}.
   7.1 Ensure you've fully addressed all requirements.
   7.2 Run any tests in the repository related to:
       7.2.1 The issue you are fixing
       7.2.2 The files you modified
       7.2.3 The functions you changed
   7.3 If any tests fail, revise your implementation until all tests pass

Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity.
"""
    return instruction


def filter_patch(patch: str) -> str:
    if not patch:
        return ""
    lines = patch.split("\n")
    filtered_lines = []
    in_garbage_file = False
    for line in lines:
        if line.startswith("diff --git"):
            filename = line.split(" b/")[-1] if " b/" in line else ""
            if filename.startswith("=") or "openhands" in filename.lower():
                in_garbage_file = True
            else:
                in_garbage_file = False
        if not in_garbage_file:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)


def recover_patch(state: State, instance: pd.Series) -> str:
    """Reconstruct a patch from str_replace edits in the agent trajectory.

    Builds a unified diff directly from the old_str/new_str pairs without
    needing the original file or an LLM call.  Uses a search-replace style
    diff that ``git apply`` can handle.
    """
    if state is None:
        return ""

    workspace_prefix = f"{get_workspace_path(instance)}/"

    edits: list[dict[str, str]] = []
    for event in state.history:
        if (
            isinstance(event, FileEditAction)
            and event.command == "str_replace"
            and event.old_str
            and event.new_str
            and event.old_str != event.new_str
        ):
            path = event.path
            if path.startswith(workspace_prefix):
                path = path[len(workspace_prefix):]
            edits.append({
                "path": path,
                "old_str": event.old_str,
                "new_str": event.new_str,
            })

    if not edits:
        return ""

    import difflib

    patches: list[str] = []
    for edit in edits:
        path = edit["path"]
        old_lines = edit["old_str"].splitlines(keepends=True)
        new_lines = edit["new_str"].splitlines(keepends=True)
        if old_lines and not old_lines[-1].endswith("\n"):
            old_lines[-1] += "\n"
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        diff_lines = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{path}", tofile=f"b/{path}",
        ))
        if not diff_lines:
            continue

        header = f"diff --git a/{path} b/{path}\n"
        patches.append(header + "".join(diff_lines))

    return "\n".join(patches)


def append_prediction(output_base_dir: str, instance_id: str, patch: str):
    predictions_path = os.path.join(output_base_dir, "predictions.jsonl")
    entry = {
        "instance_id": instance_id,
        "model_name_or_path": MODEL_NAME,
        "model_patch": patch,
    }
    with open(predictions_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


MAX_RETRIES = 5


class _StderrCapture:
    """Routes Python-level stderr writes through a logger so they land in the
    per-instance log file with timestamps."""

    def __init__(self, target_logger: logging.Logger, original):
        self._logger = target_logger
        self._original = original
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._logger.debug("[stderr] %s", line)

    def flush(self):
        if self._buf.strip():
            self._logger.debug("[stderr] %s", self._buf)
            self._buf = ""

    def __getattr__(self, name):
        return getattr(self._original, name)


def _setup_instance_logging(output_dir: str):
    """Configure verbose file logging for a single instance generation run.

    Adds a DEBUG-level FileHandler to OpenHands, Modal, and supporting loggers
    so that after a run the file contains everything needed to distinguish
    infra failures from agent failures.  Console output is unchanged.

    Returns a cleanup callable.
    """
    log_path = os.path.join(output_dir, "generation.log")

    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    loggers_and_levels = {
        "openhands": logging.DEBUG,
        "modal": logging.DEBUG,
        "httpcore": logging.INFO,
        "urllib3": logging.INFO,
        "swebench_pro_eval": logging.DEBUG,
    }

    saved_levels: dict[str, int] = {}
    muted_console: list[tuple[logging.Handler, int]] = []

    for name, level in loggers_and_levels.items():
        lgr = logging.getLogger(name)
        saved_levels[name] = lgr.level
        lgr.setLevel(level)
        lgr.addHandler(handler)

    for h in logging.getLogger("openhands").handlers:
        if h is not handler and isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            muted_console.append((h, h.level))
            h.setLevel(logging.CRITICAL + 1)

    root = logging.getLogger()
    saved_levels["root"] = root.level
    root.addHandler(handler)

    original_stderr = sys.stderr
    stderr_logger = logging.getLogger("swebench_pro_eval.stderr")
    stderr_logger.setLevel(logging.DEBUG)
    stderr_logger.addHandler(handler)
    stderr_logger.propagate = False
    sys.stderr = _StderrCapture(stderr_logger, original_stderr)

    def cleanup():
        sys.stderr = original_stderr
        for name in loggers_and_levels:
            lgr = logging.getLogger(name)
            lgr.removeHandler(handler)
            lgr.setLevel(saved_levels[name])
        root.removeHandler(handler)
        root.setLevel(saved_levels["root"])
        for h, level in muted_console:
            h.setLevel(level)
        stderr_logger.removeHandler(handler)
        handler.close()

    return cleanup


INFRA_ERROR_PATTERNS = (
    "retryerror",
    "remoteprotocolerror",
    "connecterror",
    "connection reset",
    "connection refused",
    "connection attempts failed",
    "broken pipe",
    "eof occurred",
    "server disconnected",
    "timed out",
    "502",
    "503",
    "504",
    "bad gateway",
    "container preemption",
    "sandbox connect",
    "networkerror",
)


def _classify_error(error: str) -> str:
    lower = error.lower()
    if any(pat in lower for pat in INFRA_ERROR_PATTERNS):
        return "infra"
    return "agent"


def _query_sandbox_status(runtime, instance_id: str) -> str | None:
    """Best-effort check of Modal sandbox status for post-mortem diagnostics."""
    try:
        sandbox = getattr(runtime, "sandbox", None)
        if sandbox is None:
            return None
        rc = sandbox.poll()
        if rc is not None:
            stderr_lines = []
            try:
                for line in sandbox.stderr:
                    stderr_lines.append(line.rstrip())
                    if len(stderr_lines) >= 20:
                        break
            except Exception:
                pass
            stderr_tail = "\n".join(stderr_lines[-10:]) if stderr_lines else "(no stderr)"
            return f"exited with code {rc}, stderr tail: {stderr_tail}"
        return "still running"
    except Exception as e:
        return f"could not query: {e}"


def run_single_instance(instance: pd.Series, output_base_dir: str, llm_config: LLMConfig, instance_idx: int) -> dict:
    instance_id = instance.instance_id
    output_dir = os.path.join(output_base_dir, instance_id)
    os.makedirs(output_dir, exist_ok=True)

    cleanup_logging = _setup_instance_logging(output_dir)

    try:
        for attempt in range(MAX_RETRIES):
            result = _run_single_instance_attempt(instance, output_base_dir, llm_config, attempt, instance_idx)

            if result["success"]:
                return result

            error_type = result.get("error_type", _classify_error(result["error"] or ""))
            if result["error"] is None or error_type != "infra":
                return result

            if attempt < MAX_RETRIES - 1:
                logger.warning(f"[{instance_id}] [INFRA] Attempt {attempt + 1} failed ({result['error']}), retrying in 5s...")
                time.sleep(5)

        return result
    finally:
        cleanup_logging()


def _run_single_instance_attempt(instance: pd.Series, output_base_dir: str, llm_config: LLMConfig, attempt: int, instance_idx: int) -> dict:
    instance_id = instance.instance_id
    output_dir = os.path.join(output_base_dir, instance_id)
    os.makedirs(output_dir, exist_ok=True)
    
    result = {
        "instance_id": instance_id,
        "success": False,
        "patch": "",
        "error": None,
        "num_events": 0,
        "attempt": attempt,
    }
    
    logger.info(f"[{instance_id}] Starting (attempt {attempt + 1}/{MAX_RETRIES})...")
    
    container_image = get_container_image(instance)
    
    sandbox_config = SandboxConfig(
        base_container_image=container_image
    )

    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_llm_editor=False,
        enable_browsing=False,
    )

    config = OpenHandsConfig(
        sandbox=sandbox_config,
        run_as_openhands=False,
        enable_browser=False,
        max_iterations=100,
        default_agent="CodeActAgent",
        save_trajectory_path=os.path.join(output_dir, f"trajectory_attempt{attempt}.json"),
        cache_dir=CACHE_DIR,
        file_store="memory",
        file_store_path=CACHE_DIR,
        debug=True,
    )
    config.set_llm_config(llm_config)
    config.set_agent_config(agent_config)

    file_store = InMemoryFileStore()
    event_stream = EventStream(sid=f"test-{instance_id}-{attempt}", file_store=file_store)
    llm_registry = LLMRegistry(config)

    sid = f"test-{instance_id}-{attempt}"
    runtime = ModalRuntime(
        config=config,
        event_stream=event_stream,
        llm_registry=llm_registry,
        sid=sid,
    )

    async def _run_instance():
        sandbox_start = time.time()
        await asyncio.wait_for(runtime.connect(), timeout=300)
        logger.info(f"[{instance_id}] Connected to Modal sandbox")
        
        initialize_runtime(runtime, instance, instance_id)
        logger.info(f"[{instance_id}] Runtime initialized")

        instruction = get_instruction(instance)

        state = None
        timed_out = False

        try:
            state = await asyncio.wait_for(
                run_controller(
                    config=config,
                    initial_user_action=MessageAction(content=instruction),
                    runtime=runtime,
                    fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN["CodeActAgent"],
                ),
                timeout=7200,
            )
            result["num_events"] = len(state.history)
            logger.info(f"[{instance_id}] Agent completed with {len(state.history)} events")
        except asyncio.TimeoutError:
            timed_out = True
            logger.error(f"[{instance_id}] Agent timed out after 60 minutes")
            result["error"] = "Timed out after 60 minutes"
            _save_partial_trajectory(event_stream, output_dir, attempt)

        git_patch = ""
        sandbox_dead = False
        for cr_attempt in range(3):
            try:
                patch_result = complete_runtime(runtime, instance)
                git_patch = patch_result["git_patch"]
                break
            except Exception as e:
                err = str(e).lower()
                is_transient = any(p in err for p in ("ssl", "eof", "timed out", "broken pipe", "connection", "retryerror", "remoteprotocol"))
                if cr_attempt == 2 or not is_transient:
                    sandbox_dead = True
                    logger.warning(f"[{instance_id}] Sandbox unreachable after {cr_attempt + 1} attempts: {e}")
                    break
                logger.warning(f"[{instance_id}] complete_runtime attempt {cr_attempt + 1} failed ({e}), retrying...")
                time.sleep(3 * (cr_attempt + 1))
        sandbox_duration = time.time() - sandbox_start

        instance_cost = collect_costs(state, sandbox_duration, instance_id)
        result["cost"] = instance_cost
        logger.info(f"[{instance_id}] Cost: LLM=${instance_cost.llm_cost:.4f} Sandbox=${instance_cost.sandbox_cost:.4f} Time={sandbox_duration:.2f}s")

        filtered_patch = filter_patch(git_patch)

        if not filtered_patch.strip() and state is not None:
            reason = "sandbox died before patch extraction" if sandbox_dead else "git diff empty"
            logger.info(f"[{instance_id}] {reason}, attempting trajectory-based patch recovery...")
            try:
                recovered = recover_patch(state, instance)
                if recovered.strip():
                    filtered_patch = filter_patch(recovered)
                    logger.info(f"[{instance_id}] Recovered patch from trajectory edits ({len(filtered_patch)} chars)")
                else:
                    logger.info(f"[{instance_id}] No str_replace edits found in trajectory to recover")
            except Exception as e:
                logger.warning(f"[{instance_id}] Trajectory patch recovery failed: {e}")

        patch_file = os.path.join(output_dir, "patch.diff")
        with open(patch_file, "w") as f:
            f.write(filtered_patch)
        
        result["patch"] = filtered_patch
        result["success"] = bool(filtered_patch and filtered_patch.strip())
        
        if result["success"]:
            result["error"] = None
            logger.info(f"[{instance_id}] Generated patch ({len(filtered_patch)} chars){' (after timeout)' if timed_out else ''}")
            append_prediction(output_base_dir, instance_id, filtered_patch)
        else:
            logger.warning(f"[{instance_id}] No meaningful patch generated")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run_instance())
        loop.close()
    except (asyncio.TimeoutError, TimeoutError):
        result["error"] = "Timed out during sandbox connect/init"
        result["error_type"] = "infra"
        logger.error(f"[{instance_id}] [INFRA] {result['error']}")
    except Exception as e:
        error_str = str(e) or type(e).__name__
        error_type = _classify_error(error_str)
        sandbox_status = _query_sandbox_status(runtime, instance_id)
        result["error"] = error_str
        result["error_type"] = error_type
        label = error_type.upper()
        logger.error(f"[{instance_id}] [{label}] Error: {error_str}")
        if sandbox_status:
            logger.error(f"[{instance_id}] [{label}] Sandbox status at failure: {sandbox_status}")
    finally:
        runtime.close()
    
    return result


def find_patches(input_dir: str) -> dict[str, str]:
    patches = {}
    input_path = Path(input_dir)
    
    for subdir in input_path.iterdir():
        if not subdir.is_dir():
            continue
        
        instance_id = subdir.name
        patch_file = subdir / "patch.diff"
        
        if patch_file.exists():
            patch_content = patch_file.read_text()
            if patch_content.strip():
                patches[instance_id] = patch_content
    
    return patches


def create_predictions_file(patches: dict[str, str], output_path: str, model_name: str):
    with open(output_path, "w") as f:
        for instance_id, patch in patches.items():
            entry = {
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": patch,
            }
            f.write(json.dumps(entry) + "\n")


def _prepare_pro_raw_samples(output_dir: str) -> str:
    """Write the SWE-bench Pro dataset to a JSONL file for the Pro eval script."""
    raw_path = os.path.join(output_dir, "swebench_pro_instances.jsonl")
    if os.path.exists(raw_path):
        return raw_path
    dataset = load_dataset(SWEBENCH_PRO_DATASET, split="test")
    with open(raw_path, "w") as f:
        for item in dataset:
            row = dict(item)
            if isinstance(row.get("fail_to_pass"), list):
                row["fail_to_pass"] = json.dumps(row["fail_to_pass"])
            if isinstance(row.get("pass_to_pass"), list):
                row["pass_to_pass"] = json.dumps(row["pass_to_pass"])
            if isinstance(row.get("selected_test_files_to_run"), list):
                row["selected_test_files_to_run"] = json.dumps(row["selected_test_files_to_run"])
            f.write(json.dumps(row) + "\n")
    return raw_path


def _create_pro_patches_json(patches: dict[str, str], output_dir: str) -> str:
    """Write patches in the JSON format expected by swe_bench_pro_eval.py."""
    path = os.path.join(output_dir, "pro_patches.json")
    entries = [
        {"instance_id": iid, "patch": patch, "prefix": MODEL_NAME}
        for iid, patch in patches.items()
    ]
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    return path


def run_swebench_evaluation(
    predictions_path: str,
    run_id: str,
    output_dir: str,
    max_workers: int = 50,
    timeout: int = 3600,
):
    patches = find_patches(output_dir)
    if not patches:
        logger.info("No patches to evaluate")
        return

    raw_sample_path = _prepare_pro_raw_samples(output_dir)
    patches_json = _create_pro_patches_json(patches, output_dir)
    eval_output_dir = os.path.join(output_dir, "pro_eval_output")
    os.makedirs(eval_output_dir, exist_ok=True)

    pro_eval_script = os.path.join(SWEBENCH_PRO_REPO_DIR, "swe_bench_pro_eval.py")
    scripts_dir = os.path.join(SWEBENCH_PRO_REPO_DIR, "run_scripts")

    cmd = [
        sys.executable, pro_eval_script,
        "--raw_sample_path", raw_sample_path,
        "--patch_path", patches_json,
        "--output_dir", eval_output_dir,
        "--scripts_dir", scripts_dir,
        "--dockerhub_username", "jefzda",
        "--num_workers", str(max_workers),
    ]

    logger.info(f"\nRunning SWE-bench Pro evaluation:")
    logger.info(f"  {' '.join(cmd)}")
    logger.info(f"  cwd: {SWEBENCH_PRO_REPO_DIR}")

    proc = subprocess.run(cmd, cwd=SWEBENCH_PRO_REPO_DIR)
    if proc.returncode != 0:
        logger.info(f"Warning: SWE-bench Pro eval exited with code {proc.returncode}")


def save_evaluation_results(run_id: str, model_name: str, instance_ids: list[str], input_dir: str) -> dict:
    eval_output_dir = os.path.join(input_dir, "pro_eval_output")
    report_file = os.path.join(eval_output_dir, "eval_results.json")

    if not os.path.exists(report_file):
        logger.info(f"\nNo eval_results.json found at {report_file}")
        for f in Path(eval_output_dir).glob("*.json") if Path(eval_output_dir).exists() else []:
            logger.info(f"  Found: {f}")
        return {}

    with open(report_file) as f:
        eval_results = json.load(f)

    resolved = [iid for iid, passed in eval_results.items() if passed]
    unresolved = [iid for iid in instance_ids if iid not in resolved]

    input_path = Path(input_dir)
    for instance_id in instance_ids:
        instance_dir = input_path / instance_id
        if instance_dir.exists():
            result = {
                "instance_id": instance_id,
                "resolved": instance_id in resolved,
            }
            result_file = instance_dir / "eval_result.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

    summary = {
        "run_id": run_id,
        "total": len(instance_ids),
        "resolved": len(resolved),
        "resolved_ids": resolved,
        "unresolved_ids": unresolved,
    }
    summary_file = input_path / "eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def run_generation_and_eval(output_base_dir: str, llm_config: LLMConfig, run_id: str, num_instances: int = 50) -> tuple[list[dict], dict]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1: PATCH GENERATION")
    logger.info(f"{'='*60}\n")

    instances = get_swebench_instances(num_instances)

    logger.info(f"\nStarting {len(instances)} instances in parallel with ProcessPoolExecutor...")

    successful = 0
    failed = 0
    no_patch = 0
    summary = []
    instance_costs: list[InstanceCost] = []
    completed_count = 0

    max_concurrent = num_instances
    spawn_delay = 0.25

    with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        for idx, instance in enumerate(instances):
            futures[executor.submit(run_single_instance, instance, output_base_dir, llm_config, idx)] = instance.instance_id
            if idx < max_concurrent - 1:
                time.sleep(spawn_delay)

        for future in as_completed(futures):
            instance_id = futures[future]
            completed_count += 1

            try:
                r = future.result()
                if r.get("cost") is not None:
                    instance_costs.append(r.pop("cost"))
                if r["error"]:
                    failed += 1
                    summary.append(r)
                elif r["success"]:
                    successful += 1
                    summary.append(r)
                else:
                    no_patch += 1
                    summary.append(r)
            except Exception as e:
                failed += 1
                summary.append({"instance_id": instance_id, "success": False, "error": str(e)})

            logger.info(f"\n[Progress] {completed_count}/{len(instances)} completed | {successful} patches | {failed} failed")

    summary_file = os.path.join(output_base_dir, "generation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    run_cost = aggregate_costs(instance_costs)
    cost_file = os.path.join(os.getcwd(), "eval_costs.json")
    write_costs(run_cost, cost_file)

    logger.info(f"\n{'='*60}")
    logger.info("GENERATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total instances: {len(instances)}")
    logger.info(f"Successful (with patch): {successful}")
    logger.info(f"Completed (no patch): {no_patch}")
    logger.info(f"Failed: {failed}")
    logger.info(f"\nCosts: LLM=${run_cost.total_llm_cost:.4f} Sandbox=${run_cost.total_sandbox_cost:.4f} Total=${run_cost.total_cost:.4f}")
    logger.info(f"Costs saved to: {cost_file}")
    logger.info(f"Summary saved to: {summary_file}")

    eval_summary = run_evaluation_phase(output_base_dir, run_id)

    return summary, eval_summary


def run_evaluation_phase(output_base_dir: str, run_id: str) -> dict:
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2: PATCH EVALUATION")
    logger.info(f"{'='*60}\n")
    
    logger.info(f"Scanning directory for patches: {output_base_dir}")
    patches = find_patches(output_base_dir)
    
    if not patches:
        logger.info("\nNo patches found to evaluate!")
        return {}
    
    logger.info(f"Found {len(patches)} patches to evaluate")
    
    predictions_path = os.path.join(output_base_dir, "predictions.jsonl")
    create_predictions_file(patches, predictions_path, MODEL_NAME)
    logger.info(f"Created predictions file: {predictions_path}")
    
    instance_ids = list(patches.keys())
    
    logger.info(f"\nStarting evaluation...")
    logger.info(f"  Run ID: {run_id}")
    logger.info(f"  Max workers: {len(patches)}")
    logger.info(f"  Timeout: 3600s per instance")
    
    run_swebench_evaluation(
        predictions_path=predictions_path,
        run_id=run_id,
        output_dir=output_base_dir,
        max_workers=len(patches),
    )
    
    eval_summary = save_evaluation_results(run_id, MODEL_NAME, instance_ids, output_base_dir)
    
    if eval_summary:
        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"\nTotal evaluated: {eval_summary['total']}")
        logger.info(f"Resolved: {eval_summary['resolved']} ({100*eval_summary['resolved']/eval_summary['total']:.1f}%)")
        logger.info(f"Unresolved: {len(eval_summary['unresolved_ids'])}")
        
        if eval_summary['resolved_ids']:
            logger.info(f"\nResolved instances:")
            for inst in eval_summary['resolved_ids']:
                logger.info(f"   - {inst}")
        
        logger.info(f"\nEval summary saved to: {output_base_dir}/eval_summary.json")
    
    return eval_summary


def save_config(output_dir: str, num_instances: int):
    """Persist run configuration and a snapshot of this script to output_dir."""
    config = {
        "openhands_path": REPO_DIR,
        "script_path": os.path.abspath(__file__),
        "model": MODEL,
        "num_instances": num_instances,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    shutil.copy2(__file__, os.path.join(output_dir, "run_swebench_pro_eval.py"))


def main(num_instances: int = 50):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join(os.getcwd(), "trajectories", timestamp)
    os.makedirs(output_base_dir, exist_ok=True)

    save_config(output_base_dir, num_instances)

    logger.info(f"\n{'='*60}")
    logger.info("SWE-BENCH PRO EVALUATION PIPELINE")
    logger.info(f"{'='*60}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info(f"Model: {MODEL}")
    logger.info(f"Instances: {num_instances}")
    logger.info(f"{'='*60}\n")

    llm_config = LLMConfig(
        model=MODEL,
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_message_chars=30000,
        reasoning_effort="high",
    )

    generation_results, eval_results = run_generation_and_eval(output_base_dir, llm_config, timestamp, num_instances)
    
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    
    patches_generated = sum(1 for r in generation_results if isinstance(r, dict) and r.get("success"))
    resolved = eval_results.get("resolved", 0) if eval_results else 0
    total = eval_results.get("total", 0) if eval_results else 0
    
    logger.info(f"\nPatches generated: {patches_generated}/{num_instances}")
    logger.info(f"Tests resolved: {resolved}/{total}")
    if total > 0:
        logger.info(f"Resolution rate: {100*resolved/total:.1f}%")
    logger.info(f"\nAll results saved to: {output_base_dir}")
    
    return {
        "output_dir": output_base_dir,
        "patches_generated": patches_generated,
        "resolved": resolved,
        "total": total,
        "num_instances": num_instances,
    }


def find_failed_instances(trajectory_dir: str) -> list[str]:
    """Find instance IDs that have no subdir or an empty patch.diff.

    Determines the expected instance set from generation_summary.json (if it
    exists) unioned with any subdirectory whose name is a valid SWE-bench Pro
    instance ID.
    """
    trajectory_path = Path(trajectory_dir)

    by_id = _load_pro_dataset()
    all_ids = set(by_id.keys())

    expected_ids: set[str] = set()

    gen_summary = trajectory_path / "generation_summary.json"
    if gen_summary.exists():
        with open(gen_summary) as f:
            for entry in json.load(f):
                iid = entry.get("instance_id", "")
                if iid in all_ids:
                    expected_ids.add(iid)

    for subdir in trajectory_path.iterdir():
        if subdir.is_dir() and subdir.name in all_ids:
            expected_ids.add(subdir.name)

    if not expected_ids:
        logger.warning("Could not determine instance set from trajectory directory")
        return []

    failed: list[str] = []
    for iid in sorted(expected_ids):
        subdir = trajectory_path / iid
        if not subdir.exists():
            failed.append(iid)
            continue
        patch_file = subdir / "patch.diff"
        if not patch_file.exists() or not patch_file.read_text().strip():
            failed.append(iid)

    return failed


def _load_existing_eval(trajectory_dir: str) -> dict:
    """Load a previous eval_summary.json, returning empty structure if absent."""
    summary_path = Path(trajectory_dir) / "eval_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {"run_id": "", "total": 0, "resolved": 0, "resolved_ids": [], "unresolved_ids": []}


def _merge_eval(trajectory_dir: str, old_eval: dict, new_eval: dict) -> dict:
    """Merge new evaluation results into old, updating totals and per-instance files."""
    old_resolved = set(old_eval.get("resolved_ids", []))
    new_resolved = set(new_eval.get("resolved_ids", []))
    new_evaluated = set(new_eval.get("resolved_ids", [])) | set(new_eval.get("unresolved_ids", []))

    old_unresolved = set(old_eval.get("unresolved_ids", []))
    all_resolved = sorted(old_resolved | new_resolved)
    all_unresolved = sorted((old_unresolved - new_evaluated) | (new_evaluated - new_resolved))
    total = len(all_resolved) + len(all_unresolved)

    merged = {
        "run_id": new_eval.get("run_id") or old_eval.get("run_id", ""),
        "total": total,
        "resolved": len(all_resolved),
        "resolved_ids": all_resolved,
        "unresolved_ids": all_unresolved,
    }

    summary_path = Path(trajectory_dir) / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(merged, f, indent=2)

    return merged


def retry_failed(trajectory_dir: str):
    """Re-generate patches for failed instances, then evaluate only the new ones."""
    trajectory_dir = os.path.abspath(trajectory_dir)

    logger.info(f"\n{'='*60}")
    logger.info("RETRY FAILED INSTANCES")
    logger.info(f"{'='*60}")
    logger.info(f"Trajectory directory: {trajectory_dir}")

    failed_ids = find_failed_instances(trajectory_dir)
    existing_patches = find_patches(trajectory_dir)
    old_eval = _load_existing_eval(trajectory_dir)

    logger.info(f"Existing patches: {len(existing_patches)}")
    logger.info(f"Previously evaluated: {old_eval['total']} ({old_eval['resolved']} resolved)")
    logger.info(f"Failed/missing instances: {len(failed_ids)}")

    if not failed_ids:
        logger.info("Nothing to retry — all instances have patches.")
        return old_eval

    for iid in failed_ids:
        logger.info(f"  - {iid}")

    by_id = _load_pro_dataset()
    instances_to_retry = [by_id[iid] for iid in failed_ids if iid in by_id]

    llm_config = LLMConfig(
        model=MODEL,
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_message_chars=30000,
        reasoning_effort="high",
    )

    logger.info(f"\nRe-generating {len(instances_to_retry)} instances...")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    successful = 0
    failed_count = 0
    no_patch = 0
    completed_count = 0
    instance_costs: list[InstanceCost] = []

    max_concurrent = len(instances_to_retry)
    spawn_delay = 0.25

    with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        for idx, instance in enumerate(instances_to_retry):
            futures[executor.submit(run_single_instance, instance, trajectory_dir, llm_config, idx)] = instance.instance_id
            if idx < max_concurrent - 1:
                time.sleep(spawn_delay)

        for future in as_completed(futures):
            instance_id = futures[future]
            completed_count += 1
            try:
                r = future.result()
                if r.get("cost") is not None:
                    instance_costs.append(r.pop("cost"))
                if r.get("error"):
                    failed_count += 1
                elif r.get("success"):
                    successful += 1
                else:
                    no_patch += 1
            except Exception as e:
                failed_count += 1
                logger.error(f"[{instance_id}] Exception: {e}")

            logger.info(f"\n[Progress] {completed_count}/{len(instances_to_retry)} completed | {successful} patches | {failed_count} failed")

    if instance_costs:
        retry_cost = aggregate_costs(instance_costs)
        logger.info(f"\nRetry costs: LLM=${retry_cost.total_llm_cost:.4f} Sandbox=${retry_cost.total_sandbox_cost:.4f} Total=${retry_cost.total_cost:.4f}")

    new_patches = {iid: patch for iid, patch in find_patches(trajectory_dir).items() if iid in failed_ids}

    logger.info(f"\n{'='*60}")
    logger.info("RETRY GENERATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Previously had patches: {len(existing_patches)}")
    logger.info(f"Retried: {len(instances_to_retry)} ({successful} new patches, {failed_count} failed, {no_patch} no patch)")
    logger.info(f"Total patches now: {len(existing_patches) + len(new_patches)}")

    if not new_patches:
        logger.info("No new patches to evaluate.")
        return old_eval

    run_id = Path(trajectory_dir).name + "_retry"
    logger.info(f"\nEvaluating {len(new_patches)} new patches only...")

    predictions_path = os.path.join(trajectory_dir, "predictions_retry.jsonl")
    create_predictions_file(new_patches, predictions_path, MODEL_NAME)

    run_swebench_evaluation(
        predictions_path=predictions_path,
        run_id=run_id,
        output_dir=trajectory_dir,
        max_workers=len(new_patches),
        timeout=3600,
    )

    new_eval = save_evaluation_results(run_id, MODEL_NAME, list(new_patches.keys()), trajectory_dir)
    merged = _merge_eval(trajectory_dir, old_eval, new_eval)

    logger.info(f"\n{'='*60}")
    logger.info("COMBINED EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total evaluated: {merged['total']}")
    logger.info(f"Resolved: {merged['resolved']} ({100*merged['resolved']/merged['total']:.1f}%)")
    logger.info(f"Unresolved: {len(merged['unresolved_ids'])}")
    if merged['resolved_ids']:
        logger.info(f"\nAll resolved instances:")
        for inst in merged['resolved_ids']:
            logger.info(f"   - {inst}")
    logger.info(f"\nEval summary saved to: {trajectory_dir}/eval_summary.json")

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWE-bench Pro evaluation pipeline")
    subparsers = parser.add_subparsers(dest="command")

    p_run = subparsers.add_parser("run", help="Full pipeline: generate patches + evaluate")
    p_run.add_argument("-n", "--num-instances", type=int, default=50,
                       choices=range(1, 732), metavar="[1-731]",
                       help="Number of SWE-bench Pro instances (default: 50, max: 731)")

    # eval: evaluate existing patches
    p_eval = subparsers.add_parser("eval", help="Evaluate existing patches only")
    p_eval.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing generated patches")

    # retry: re-generate failed instances, then evaluate all
    p_retry = subparsers.add_parser("retry", help="Retry failed/missing instances and evaluate")
    p_retry.add_argument("--input-dir", type=str, required=True,
                         help="Trajectory directory from a previous run")

    # Support legacy usage without subcommand: just -n
    parser.add_argument("-n", "--num-instances", type=int, default=50,
                        choices=range(1, 732), metavar="[1-731]",
                        help="Number of SWE-bench Pro instances (default: 50, max: 731)")

    args = parser.parse_args()
    start_time = time.time()

    if args.command == "eval":
        run_id = Path(args.input_dir).name
        eval_results = run_evaluation_phase(args.input_dir, run_id)
        resolved = eval_results.get("resolved", 0) if eval_results else 0
        total = eval_results.get("total", 0) if eval_results else 0
        logger.info(f"\n{'='*60}")
        logger.info(f"TOTAL CORRECT: {resolved}/{total}")
        if total > 0:
            logger.info(f"PERCENT CORRECT: {100*resolved/total:.1f}%")
        elapsed = time.time() - start_time
        logger.info(f"ELAPSED TIME: {elapsed/60:.1f} minutes")
        logger.info(f"{'='*60}")

    elif args.command == "retry":
        eval_results = retry_failed(args.input_dir)
        resolved = eval_results.get("resolved", 0) if eval_results else 0
        total = eval_results.get("total", 0) if eval_results else 0
        logger.info(f"\n{'='*60}")
        logger.info(f"TOTAL CORRECT: {resolved}/{total}")
        if total > 0:
            logger.info(f"PERCENT CORRECT: {100*resolved/total:.1f}%")
        elapsed = time.time() - start_time
        logger.info(f"ELAPSED TIME: {elapsed/60:.1f} minutes")
        logger.info(f"{'='*60}")

    else:
        results = main(num_instances=args.num_instances)
        resolved = results.get("resolved", 0)
        total = results.get("total", 0)
        patches = results.get("patches_generated", 0)
        num_instances = results.get("num_instances", args.num_instances)

        sys.stdout.flush()
        sys.stderr.flush()

        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"TOTAL CORRECT: {resolved}/{total}")
        if total > 0:
            logger.info(f"PERCENT CORRECT: {100*resolved/total:.1f}%")
        logger.info(f"PATCHES GENERATED: {patches}/{num_instances}")
        logger.info(f"ELAPSED TIME: {elapsed/60:.1f} minutes")
        logger.info(f"{'='*60}")

    sys.stdout.flush()
    os._exit(0)
