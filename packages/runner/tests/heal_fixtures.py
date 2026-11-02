"""Fixtures the heal's tests share (#244): a run of `Healing` on a scripted model
through `replays.replay_run`, in real Chromium against #240's drift pages, and
what the model's moves look like. Imported by its path, as pytest names the
runner's test modules (TESTING.md §1, Shared egress fixtures)."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from aqa_core.compiled import CompiledScript
from aqa_core.project import contracts_fingerprint, load_config, load_spec
from aqa_runner.admission import admitted
from aqa_runner.egress_proxy import EgressProxy
from aqa_runner.executor import RunResult
from aqa_runner.heal import Healing
from aqa_runner.locator_generation import register_identity_engine
from aqa_runner.model_router import ModelRouter
from aqa_runner.model_turns import Spending
from aqa_runner.redaction import Redactor
from aqa_runner.repairs import Basis, Drift, Offer
from aqa_runner.replays import Opened, Replaying, replay_run
from aqa_runner.run_record import RunRecord
from langchain_core.messages import AIMessage, BaseMessage
from playwright.async_api import async_playwright

from packages.runner.tests import executor_fixtures as fx
from packages.runner.tests.drift_fixtures import (
    DETAILS,
    SHIPPING,
    order_spec,
    page,
    serving,
)
from packages.runner.tests.egress_fixtures import gate
from packages.runner.tests.explore_graph_fixtures import (
    Move,
    Scripted,
    answer,
    ref,
    tool,
)

MOVED = page("moved")
# Long enough that a heal's reads don't run out under a loaded machine.
BUDGETS = "budgets: { resolve_seconds: 2 }\n"


def rebinding(
    role: str, name: str, *, confidence: float = 0.9, rationale: str = "it fits"
) -> Move:
    """A move that picks the element the request's snapshot gives `role` and
    `name`."""
    return tool(
        "rebind",
        lambda messages: {
            "ref": ref(messages, role, name),
            "confidence": confidence,
            "rationale": rationale,
        },
    )


SHOW = rebinding("button", "Show details", confidence=0.9, rationale="it moved")
NO_TOOL = answer(AIMessage(content="nothing on this page fits"))
FOUND_NONE = "the healer found no element for the meaning"


def shipping_at(root: Path) -> Path:
    return order_spec(root, SHIPPING)


def with_budgets(budgets: str) -> str:
    return f"budgets: {{ resolve_seconds: 2, {budgets} }}\n"


@dataclass(frozen=True)
class Healed:
    """A run that healed on a scripted model: its result, its record's
    directory, the model, each request the app saw, the committed script and
    the run's spend."""

    result: RunResult
    record: Path
    client: Scripted
    seen: list[tuple[str, str]]
    committed: CompiledScript
    spending: Spending
    drifts: list[Drift]

    def requests(self) -> list[str]:
        """The text of each request's last message, as sent."""
        return [request[-1].text for request in self.client.requests]

    def asked(self) -> Sequence[Sequence[BaseMessage]]:
        return self.client.requests


def healed(
    tmp_path: Path,
    *moves: Move,
    script: CompiledScript = DETAILS,
    spec_at: Any = shipping_at,
    html: str = MOVED,
    config: str = BUDGETS,
    before_the_heal: Callable[[Drift], Awaitable[object]] | None = None,
) -> Healed:
    """`script` replayed with every locator on `html`, with `Healing` on a model
    that plays `moves`, for the spec `spec_at` writes under a spec root, whose
    config is `config`. `before_the_heal` runs on each drift first."""
    path = spec_at(tmp_path / "qa")
    (path.parent / "config.yaml").write_text(config)
    loaded = load_config(path.parent / "config.yaml")
    spec = load_spec(path, loaded)
    fingerprint = contracts_fingerprint(loaded, script.spec_id)
    by = script.compiled_by.model_copy(update={"subject_contracts": fingerprint})
    committed = fx.for_spec(
        script.model_copy(update={"compiled_by": by}), spec
    ).model_copy(deep=True)
    client = Scripted(*moves)
    drifts: list[Drift] = []
    with serving(html) as app:
        run_gate = gate(allowed=(app.origin,))
        _, _, bound, _ = admitted(committed, spec, loaded, app.origin)
        record = RunRecord.create(path.parent).redacting(Redactor(bound.values()))
        spending = Spending(record, Decimal(str(loaded.budgets.heal_model_usd)))
        healing = Healing(
            ModelRouter.from_config(loaded, client),
            spending,
            loaded.budgets.heal_turns,
            loaded.budgets.resolve_seconds,
        )

        async def repair(drift: Drift) -> Offer:
            drifts.append(drift)
            if before_the_heal is not None:
                await before_the_heal(drift)
            return await healing.repair(drift)

        @asynccontextmanager
        async def opening() -> AsyncIterator[Opened]:
            async with async_playwright() as playwright, EgressProxy(run_gate) as proxy:
                await register_identity_engine(playwright)
                yield Opened(playwright.chromium, proxy, record)

        async def scenario() -> RunResult:
            replaying = Replaying(spec, loaded, app.origin, run_gate)
            # No heal test may hang the suite.
            async with asyncio.timeout(60):
                return await replay_run(
                    committed, replaying, opening, every_locator=True, repair=repair
                )

        result = asyncio.run(scenario())
    return Healed(result, record.path, client, app.seen, committed, spending, drifts)


def outcomes(run: Healed) -> list[tuple[int, str]]:
    return [(step.seq, step.outcome) for step in run.result.steps]


def basis(run: Healed, index: int = 0) -> Basis:
    given = run.result.repairs[index].basis
    assert given is not None
    return given


def turn_files(run: Healed, key: str = "1") -> list[str]:
    folder = run.record / f"heals/{key}"
    return sorted(path.name for path in folder.glob("[0-9]*.json"))
