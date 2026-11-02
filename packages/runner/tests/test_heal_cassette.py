"""Each drift page's `aqa run` replayed from its live recording (ADR-0024's #244
amendment): in process with the real adapter, in real Chromium against #240's
pages, and the proposal the recorded model's pick leaves. No test here reaches a
provider."""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest
import typer.testing
from aqa_cli.main import app as aqa
from aqa_core.project import load_project
from aqa_runner.publication import published_text

from packages.runner.tests.conftest import Recording
from packages.runner.tests.drift_fixtures import serving
from packages.runner.tests.executor_fixtures import for_spec
from packages.runner.tests.heal_fixtures import HEAL_CASES, HealCase, heal_spec

pytestmark = pytest.mark.usefixtures("reset_tracing")

Cassette = Callable[..., AbstractContextManager[Recording]]


def committed(case: HealCase, root: Path) -> Path:
    """`case`'s spec under `root`, its script committed as published. The spec's
    path."""
    root.mkdir(parents=True)
    path = heal_spec(case, root).path
    spec = load_project(root).specs["order"]
    (root / ".compiled").mkdir()
    (root / ".compiled/order.json").write_text(
        published_text(for_spec(case.script, spec))
    )
    return path


@pytest.mark.parametrize("case", HEAL_CASES, ids=lambda case: case.cassette)
def test_each_heal_cassette_replays_offline_to_drift_consistent(
    cassette: Cassette, tmp_path: Path, case: HealCase
) -> None:
    spec = committed(case, tmp_path / "qa")
    before = (spec.parent / ".compiled/order.json").read_bytes()

    with serving(case.html) as app, cassette(case.cassette, replay_only=True) as played:
        result = typer.testing.CliRunner().invoke(
            aqa, ["run", str(spec), "--url", app.origin]
        )

    assert result.exit_code == 7, result.output
    [record] = (spec.parent / ".aqa/runs").iterdir()
    document: dict[str, Any] = json.loads((record / "replay.json").read_text())
    assert (document["mode"], document["outcome"]) == ("run", "drift_consistent")
    assert document["heal_proposal"]["patch"] == [
        {
            "op": "replace",
            "path": "/targets/toggle/locators",
            "value": case.locators,
        }
    ]
    assert (spec.parent / ".compiled/order.json").read_bytes() == before
    lines = (record / "costs.jsonl").read_text().splitlines()
    costs = [json.loads(line) for line in lines]
    assert [(cost["role"], cost["mode"]) for cost in costs] == [("healer", "heal")]
    assert len(played.sent) == len(costs)


@pytest.mark.parametrize("case", HEAL_CASES, ids=lambda case: case.cassette)
def test_a_heal_request_offers_one_tool_and_holds_no_claim(
    cassette: Cassette, tmp_path: Path, case: HealCase
) -> None:
    spec = committed(case, tmp_path / "qa")

    with serving(case.html) as app, cassette(case.cassette, replay_only=True) as played:
        typer.testing.CliRunner().invoke(aqa, ["run", str(spec), "--url", app.origin])

    [sent] = played.sent
    assert [tool["name"] for tool in sent["tools"]] == ["rebind"]
    # One call a reply, forcing none: no element fitting must stay an answer.
    assert sent["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    text = json.dumps(sent["messages"])
    assert "ships on Monday" not in text
    assert "<target>" in text
    assert "<snapshot>" in text
