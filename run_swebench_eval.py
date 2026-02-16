#!/usr/bin/env python3
import atexit
import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("openhands").setLevel(logging.ERROR)
logging.getLogger("modal").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

import pandas as pd
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

_runtimes_to_cleanup = []

def _silent_cleanup():
    for runtime in _runtimes_to_cleanup:
        if hasattr(runtime, '_sandbox') and runtime._sandbox is not None:
            pass

atexit.register(_silent_cleanup)

modal_toml = os.path.expanduser("~/.modal.toml")
with open(modal_toml) as f:
    content = f.read()
token_id = re.search(r'token_id = "([^"]+)"', content).group(1)
token_secret = re.search(r'token_secret = "([^"]+)"', content).group(1)

os.environ["MODAL_TOKEN_ID"] = token_id
os.environ["MODAL_TOKEN_SECRET"] = token_secret

TEMP_DIR = os.path.join(SCRIPT_DIR, ".tmp")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")
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
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import run_controller
from openhands.events import EventStream
from openhands.events.action import CmdRunAction, FileReadAction, MessageAction
from openhands.events.observation import CmdOutputObservation, ErrorObservation, FileReadObservation
from openhands.llm.llm_registry import LLMRegistry
from openhands.resolver.utils import codeact_user_response
from openhands.runtime.base import Runtime
from openhands.storage import InMemoryFileStore
from third_party.runtime.impl.modal.modal_runtime import ModalRuntime

logger.setLevel(logging.ERROR)


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    "CodeActAgent": codeact_user_response,
}

NUM_INSTANCES = 50
MODEL = "openai/gpt-5-mini"
MODEL_NAME = "swebench_eval"

INSTANCE_SWE_ENTRY_SCRIPT = os.path.join(SCRIPT_DIR, "scripts", "swe_bench", "instance_swe_entry.sh")


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


def get_swebench_instances(num_instances: int) -> list[pd.Series]:
    print(f"Loading first {num_instances} instances from SWE-bench_Verified dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    instances = []
    for i, item in enumerate(dataset):
        if i >= num_instances:
            break
        instances.append(pd.Series(item))
    print(f"Loaded {len(instances)} instances")
    return instances


def get_swebench_workspace_dir_name(instance: pd.Series) -> str:
    return f'{instance.repo}__{instance.version}'.replace('/', '__')


def get_container_image(instance: pd.Series) -> str:
    instance_id = instance.instance_id
    repo, name = instance_id.split("__")
    return f"docker.io/swebench/sweb.eval.x86_64.{repo}_1776_{name}:latest".lower()


def initialize_runtime(runtime: Runtime, instance: pd.Series, instance_id: str):
    workspace_dir_name = get_swebench_workspace_dir_name(instance)

    action = CmdRunAction(
        command=f"""echo 'export SWE_INSTANCE_ID={instance["instance_id"]}' >> ~/.bashrc && echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && echo "alias git='git --no-pager'" >> ~/.bashrc && git config --global core.pager "" && git config --global diff.binary false"""
    )
    action.set_hard_timeout(600)
    runtime.run_action(action)

    action = CmdRunAction(command="""export USER=$(whoami); echo USER=${USER}""")
    action.set_hard_timeout(600)
    runtime.run_action(action)

    action = CmdRunAction(command="mkdir -p /swe_util/eval_data/instances")
    action.set_hard_timeout(600)
    runtime.run_action(action)

    swe_instance_json_name = "swe-bench-instance.json"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, swe_instance_json_name)
        with open(temp_file_path, "w") as f:
            json.dump([instance.to_dict()], f)
        runtime.copy_to(temp_file_path, "/swe_util/eval_data/instances/")
        runtime.copy_to(INSTANCE_SWE_ENTRY_SCRIPT, "/swe_util/")

    action = CmdRunAction(command="source ~/.bashrc")
    action.set_hard_timeout(600)
    runtime.run_action(action)

    action = CmdRunAction(command="source /swe_util/instance_swe_entry.sh")
    action.set_hard_timeout(600)
    runtime.run_action(action)

    action = CmdRunAction(command=f"cd /workspace/{workspace_dir_name}")
    action.set_hard_timeout(600)
    runtime.run_action(action)

    action = CmdRunAction(command="git remote remove origin 2>/dev/null || true; git remote remove upstream 2>/dev/null || true")
    action.set_hard_timeout(600)
    runtime.run_action(action)


def complete_runtime(runtime: Runtime, instance: pd.Series) -> dict:
    workspace_dir_name = get_swebench_workspace_dir_name(instance)

    action = CmdRunAction(command=f"cd /workspace/{workspace_dir_name}")
    action.set_hard_timeout(600)
    runtime.run_action(action)

    action = CmdRunAction(command='git config --global core.pager ""')
    action.set_hard_timeout(600)
    runtime.run_action(action)

    action = CmdRunAction(command='find . -type d -name .git -not -path "./.git"')
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)

    git_dirs = [p for p in obs.content.strip().split("\n") if p]
    for git_dir in git_dirs:
        action = CmdRunAction(command=f'rm -rf "{git_dir}"')
        action.set_hard_timeout(600)
        runtime.run_action(action)

    action = CmdRunAction(command="git add -A")
    action.set_hard_timeout(600)
    runtime.run_action(action)

    action = CmdRunAction(command=remove_binary_files_from_git())
    action.set_hard_timeout(600)
    runtime.run_action(action)

    action = CmdRunAction(
        command=f'git diff --no-color --cached {instance["base_commit"]} > patch.diff'
    )
    action.set_hard_timeout(600)
    runtime.run_action(action)

    action = FileReadAction(path="patch.diff")
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)

    if isinstance(obs, FileReadObservation):
        git_patch = obs.content
    elif isinstance(obs, ErrorObservation):
        action = CmdRunAction(command="cat patch.diff")
        action.set_hard_timeout(600)
        obs = runtime.run_action(action)
        git_patch = obs.content
    else:
        git_patch = ""

    git_patch = remove_binary_diffs(git_patch)
    return {"git_patch": git_patch}


def get_instruction(instance: pd.Series) -> str:
    workspace_dir_name = get_swebench_workspace_dir_name(instance)

    instruction = f"""<uploaded_files>
/workspace/{workspace_dir_name}
</uploaded_files>
I've uploaded a python code repository in /workspace/{workspace_dir_name}. Your task is to make changes to the repo to fix the following issue:

<issue>
{instance["problem_statement"]}
</issue>

Please analyze the issue and make the necessary changes to fix it. You should:
1. First explore the repository structure to understand the codebase
2. Locate the relevant files that need to be modified
3. Make the necessary code changes to fix the issue
4. Verify your changes work correctly

Important: When you think you have fixed the issue, use the `finish` action to indicate completion.
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


def append_prediction(output_base_dir: str, instance_id: str, patch: str):
    predictions_path = os.path.join(output_base_dir, "predictions.jsonl")
    entry = {
        "instance_id": instance_id,
        "model_name_or_path": MODEL_NAME,
        "model_patch": patch,
    }
    with open(predictions_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


MAX_RETRIES = 3


def run_single_instance(instance: pd.Series, output_base_dir: str, llm_config: LLMConfig, instance_idx: int) -> dict:
    instance_id = instance.instance_id
    
    for attempt in range(MAX_RETRIES):
        result = _run_single_instance_attempt(instance, output_base_dir, llm_config, attempt, instance_idx)
        
        if result["success"] or result["error"] is None:
            return result
        
        if attempt < MAX_RETRIES - 1:
            print(f"[{instance_id}] Attempt {attempt + 1} failed, retrying... ({result['error']})")
            time.sleep(5)
    
    return result


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
    
    print(f"[{instance_id}] Starting (attempt {attempt + 1}/{MAX_RETRIES})...")
    
    container_image = get_container_image(instance)
    
    sandbox_config = SandboxConfig(
        base_container_image=container_image,
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
        max_iterations=75,
        default_agent="CodeActAgent",
        save_trajectory_path=os.path.join(output_dir, f"trajectory_attempt{attempt}.json"),
        cache_dir=CACHE_DIR,
        file_store_path=CACHE_DIR,
    )
    config.set_llm_config(llm_config)
    config.set_agent_config(agent_config)

    file_store = InMemoryFileStore()
    event_stream = EventStream(sid=f"test-{instance_id}-{attempt}", file_store=file_store)
    llm_registry = LLMRegistry(config)

    time.sleep(instance_idx * 0.25)
    
    runtime = ModalRuntime(
        config=config,
        event_stream=event_stream,
        llm_registry=llm_registry,
        sid=f"test-{instance_id}-{attempt}",
    )

    async def _run_instance():
        sandbox_start = time.time()
        await runtime.connect()
        print(f"[{instance_id}] Connected to Modal sandbox")
        
        initialize_runtime(runtime, instance, instance_id)
        print(f"[{instance_id}] Runtime initialized")

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
                timeout=1200,
            )
            result["num_events"] = len(state.history)
            print(f"[{instance_id}] Agent completed with {len(state.history)} events")
        except asyncio.TimeoutError:
            timed_out = True
            print(f"[{instance_id}] Agent timed out after 20 minutes")
            result["error"] = "Timed out after 20 minutes"

        patch_result = complete_runtime(runtime, instance)
        sandbox_duration = time.time() - sandbox_start

        instance_cost = collect_costs(state, sandbox_duration, instance_id)
        result["cost"] = instance_cost
        print(f"[{instance_id}] Cost: LLM=${instance_cost.llm_cost:.4f} Sandbox=${instance_cost.sandbox_cost:.4f}")

        git_patch = patch_result["git_patch"]
        filtered_patch = filter_patch(git_patch)
        
        patch_file = os.path.join(output_dir, "patch.diff")
        with open(patch_file, "w") as f:
            f.write(filtered_patch)
        
        result["patch"] = filtered_patch
        result["success"] = bool(filtered_patch and filtered_patch.strip())
        
        if result["success"]:
            print(f"[{instance_id}] Generated patch ({len(filtered_patch)} chars){' (after timeout)' if timed_out else ''}")
            append_prediction(output_base_dir, instance_id, filtered_patch)
        else:
            print(f"[{instance_id}] No meaningful patch generated")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run_instance())
        loop.close()
    except Exception as e:
        result["error"] = str(e)
        print(f"[{instance_id}] Error: {e}")
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


def run_swebench_evaluation(
    predictions_path: str,
    run_id: str,
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    max_workers: int = 50,
    timeout: int = 1200,
):
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset,
        "--split", split,
        "--predictions_path", predictions_path,
        "--run_id", run_id,
        "--timeout", str(timeout),
        "--cache_level", "instance",
        "--max_workers", str(max_workers),
        "--modal", "true",
    ]
    
    print(f"\nRunning evaluation command:")
    print(f"  {' '.join(cmd)}")
    print()
    
    subprocess.run(cmd, check=True)


def save_evaluation_results(run_id: str, model_name: str, instance_ids: list[str], input_dir: str) -> dict:
    report_file = f"{model_name}.{run_id}.json"
    
    if not os.path.exists(report_file):
        print(f"\nNo report file found at {report_file}")
        for f in Path(".").glob("*.json"):
            print(f"  Found: {f}")
        return {}
    
    with open(report_file) as f:
        report = json.load(f)
    
    resolved = report.get("resolved_ids", report.get("resolved", []))
    total = len(instance_ids)
    num_resolved = len(resolved)
    unresolved = [i for i in instance_ids if i not in resolved]
    
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
        "total": total,
        "resolved": num_resolved,
        "resolved_ids": resolved,
        "unresolved_ids": unresolved,
    }
    summary_file = input_path / "eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def run_generation_and_eval(output_base_dir: str, llm_config: LLMConfig, run_id: str) -> tuple[list[dict], dict]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    print(f"\n{'='*60}")
    print("PHASE 1: PATCH GENERATION")
    print(f"{'='*60}\n")

    instances = get_swebench_instances(NUM_INSTANCES)

    print(f"\nStarting {len(instances)} instances in parallel with ProcessPoolExecutor...")

    successful = 0
    failed = 0
    no_patch = 0
    summary = []
    instance_costs: list[InstanceCost] = []
    completed_count = 0

    with ProcessPoolExecutor(max_workers=NUM_INSTANCES) as executor:
        futures = {
            executor.submit(run_single_instance, instance, output_base_dir, llm_config, idx): instance.instance_id
            for idx, instance in enumerate(instances)
        }
        
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

            print(f"\n[Progress] {completed_count}/{len(instances)} completed | {successful} patches | {failed} failed")

    summary_file = os.path.join(output_base_dir, "generation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    run_cost = aggregate_costs(instance_costs)
    costs_file = os.path.join(output_base_dir, "costs.json")
    write_costs(run_cost, costs_file)

    print(f"\n{'='*60}")
    print("GENERATION RESULTS")
    print(f"{'='*60}")
    print(f"Total instances: {len(instances)}")
    print(f"Successful (with patch): {successful}")
    print(f"Completed (no patch): {no_patch}")
    print(f"Failed: {failed}")
    print(f"\nCosts: LLM=${run_cost.total_llm_cost:.4f} Sandbox=${run_cost.total_sandbox_cost:.4f} Total=${run_cost.total_cost:.4f}")
    print(f"Costs saved to: {costs_file}")
    print(f"Summary saved to: {summary_file}")

    eval_summary = run_evaluation_phase(output_base_dir, run_id)

    return summary, eval_summary


def run_evaluation_phase(output_base_dir: str, run_id: str) -> dict:
    print(f"\n{'='*60}")
    print("PHASE 2: PATCH EVALUATION")
    print(f"{'='*60}\n")
    
    print(f"Scanning directory for patches: {output_base_dir}")
    patches = find_patches(output_base_dir)
    
    if not patches:
        print("\nNo patches found to evaluate!")
        return {}
    
    print(f"Found {len(patches)} patches to evaluate")
    
    predictions_path = os.path.join(output_base_dir, "predictions.jsonl")
    create_predictions_file(patches, predictions_path, MODEL_NAME)
    print(f"Created predictions file: {predictions_path}")
    
    instance_ids = list(patches.keys())
    
    print(f"\nStarting evaluation...")
    print(f"  Run ID: {run_id}")
    print(f"  Max workers: {len(patches)}")
    print(f"  Timeout: 1200s per instance")
    
    run_swebench_evaluation(
        predictions_path=predictions_path,
        run_id=run_id,
        max_workers=len(patches),
        timeout=1200,
    )
    
    eval_summary = save_evaluation_results(run_id, MODEL_NAME, instance_ids, output_base_dir)
    
    if eval_summary:
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"\nTotal evaluated: {eval_summary['total']}")
        print(f"Resolved: {eval_summary['resolved']} ({100*eval_summary['resolved']/eval_summary['total']:.1f}%)")
        print(f"Unresolved: {len(eval_summary['unresolved_ids'])}")
        
        if eval_summary['resolved_ids']:
            print(f"\nResolved instances:")
            for inst in eval_summary['resolved_ids']:
                print(f"   - {inst}")
        
        print(f"\nEval summary saved to: {output_base_dir}/eval_summary.json")
    
    return eval_summary


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join(SCRIPT_DIR, "trajectories", timestamp)
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SWE-BENCH EVALUATION PIPELINE")
    print(f"{'='*60}")
    print(f"Timestamp: {timestamp}")
    print(f"Output directory: {output_base_dir}")
    print(f"Model: {MODEL}")
    print(f"Instances: {NUM_INSTANCES}")
    print(f"{'='*60}\n")

    llm_config = LLMConfig(
        model=MODEL,
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_message_chars=30000,
    )

    generation_results, eval_results = run_generation_and_eval(output_base_dir, llm_config, timestamp)
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    patches_generated = sum(1 for r in generation_results if isinstance(r, dict) and r.get("success"))
    resolved = eval_results.get("resolved", 0) if eval_results else 0
    total = eval_results.get("total", 0) if eval_results else 0
    
    print(f"\nPatches generated: {patches_generated}/{NUM_INSTANCES}")
    print(f"Tests resolved: {resolved}/{total}")
    if total > 0:
        print(f"Resolution rate: {100*resolved/total:.1f}%")
    print(f"\nAll results saved to: {output_base_dir}")
    
    return {
        "output_dir": output_base_dir,
        "patches_generated": patches_generated,
        "resolved": resolved,
        "total": total,
    }


if __name__ == "__main__":
    results = main()
    resolved = results.get("resolved", 0)
    total = results.get("total", 0)
    patches = results.get("patches_generated", 0)
    
    sys.stdout.flush()
    sys.stderr.flush()
    
    print(f"\n")
    print(f"{'='*60}")
    print(f"TOTAL CORRECT: {resolved}/{total}")
    if total > 0:
        print(f"PERCENT CORRECT: {100*resolved/total:.1f}%")
    print(f"PATCHES GENERATED: {patches}/{NUM_INSTANCES}")
    print(f"{'='*60}")
    
    sys.stdout.flush()
    os._exit(0 if resolved > 0 else 1)
