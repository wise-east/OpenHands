from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.controller.state.state import State

CPU_PRICING_PER_CORE_SEC = 0.0000131
MEMORY_PRICING_PER_GIB_SEC = 0.00000222

SANDBOX_DEFAULT_CPU = 1.0
SANDBOX_DEFAULT_MEMORY_GIB = 0.25


@dataclass
class TokenBreakdown:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost: float
    num_calls: int


@dataclass
class SandboxCost:
    duration_seconds: float
    estimated_cost: float


@dataclass
class InstanceCost:
    instance_id: str
    llm_cost: float
    sandbox_cost: float
    total_cost: float
    tokens: list[TokenBreakdown]
    sandbox: SandboxCost


@dataclass
class RunCost:
    total_llm_cost: float
    total_sandbox_cost: float
    total_cost: float
    instances: list[InstanceCost] = field(default_factory=list)


def _estimate_sandbox_cost(
    duration_seconds: float,
    cpu: float = SANDBOX_DEFAULT_CPU,
    memory_gib: float = SANDBOX_DEFAULT_MEMORY_GIB,
) -> float:
    cpu_cost = cpu * CPU_PRICING_PER_CORE_SEC * duration_seconds
    memory_cost = memory_gib * MEMORY_PRICING_PER_GIB_SEC * duration_seconds
    return cpu_cost + memory_cost


def collect(state: State, sandbox_duration: float, instance_id: str) -> InstanceCost:
    """Extract LLM and sandbox cost data from a completed controller run."""
    tokens_by_model: dict[str, TokenBreakdown] = {}

    if state is not None and state.conversation_stats is not None:
        for metrics in state.conversation_stats.service_to_metrics.values():
            for usage in metrics.token_usages:
                model = usage.model or metrics.model_name
                if model not in tokens_by_model:
                    tokens_by_model[model] = TokenBreakdown(
                        model=model,
                        prompt_tokens=0,
                        completion_tokens=0,
                        cache_read_tokens=0,
                        cache_write_tokens=0,
                        cost=0.0,
                        num_calls=0,
                    )
                entry = tokens_by_model[model]
                entry.prompt_tokens += usage.prompt_tokens
                entry.completion_tokens += usage.completion_tokens
                entry.cache_read_tokens += usage.cache_read_tokens
                entry.cache_write_tokens += usage.cache_write_tokens
                entry.num_calls += 1

            for cost_entry in metrics.costs:
                model = cost_entry.model or metrics.model_name
                if model not in tokens_by_model:
                    tokens_by_model[model] = TokenBreakdown(
                        model=model,
                        prompt_tokens=0,
                        completion_tokens=0,
                        cache_read_tokens=0,
                        cache_write_tokens=0,
                        cost=0.0,
                        num_calls=0,
                    )
                tokens_by_model[model].cost += cost_entry.cost

    tokens = list(tokens_by_model.values())
    llm_cost = sum(t.cost for t in tokens)
    sandbox_estimated = _estimate_sandbox_cost(sandbox_duration)
    sandbox = SandboxCost(
        duration_seconds=sandbox_duration,
        estimated_cost=sandbox_estimated,
    )

    return InstanceCost(
        instance_id=instance_id,
        llm_cost=llm_cost,
        sandbox_cost=sandbox_estimated,
        total_cost=llm_cost + sandbox_estimated,
        tokens=tokens,
        sandbox=sandbox,
    )


def aggregate(instances: list[InstanceCost]) -> RunCost:
    total_llm = sum(i.llm_cost for i in instances)
    total_sandbox = sum(i.sandbox_cost for i in instances)
    return RunCost(
        total_llm_cost=total_llm,
        total_sandbox_cost=total_sandbox,
        total_cost=total_llm + total_sandbox,
        instances=instances,
    )


def write(run_cost: RunCost, path: str) -> None:
    with open(path, "w") as f:
        json.dump(asdict(run_cost), f, indent=2)
