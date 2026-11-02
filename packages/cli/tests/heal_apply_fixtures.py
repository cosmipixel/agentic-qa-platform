"""What `test_heal_apply.py` and `test_heal_apply_refusals.py` share (API.md §7): a
committed script, a proposal written by hand into a run record, as a person's edit
would be, and `aqa heal apply` run in this process. No browser and no model."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import typer.testing
from aqa_cli.main import app as aqa
from aqa_core.compiled import CompiledScript
from aqa_core.heal_patch import blob_sha, proposal_id
from aqa_core.project import load_project
from aqa_runner.publication import published_text

from packages.runner.tests.drift_fixtures import (
    DETAILS,
    SHIPPING,
    SUMMARY,
    TOGGLE,
    order_spec,
)
from packages.runner.tests.executor_fixtures import compiled, for_spec

# The toggle's locators as `published_text` writes them, a key order of its own.
LOCATORS = """\
      "locators": [
        {
          "role": "button",
          "name": "Show details"
        },
        {
          "scope": {
            "css": "#toolbar"
          },
          "css": "button"
        }
      ]
"""
MOVED_LOCATORS = """\
      "locators": [
        {
          "role": "button",
          "name": "Moved"
        },
        {
          "css": "#moved"
        }
      ]
"""
# The same, as `replay.json` holds them, whose keys are sorted.
MOVED = [{"name": "Moved", "role": "button"}, {"css": "#moved"}]
REBIND: list[dict[str, Any]] = [
    {"op": "replace", "path": "/targets/toggle/locators", "value": MOVED}
]
# A script with a replay-safe step and a side-effect step.
A_STEP = {"seq": 1, "action": "click", "target": "toggle", "side_effect": False}
KEPT_SIDE_EFFECT = {
    "seq": 2,
    "action": "click",
    "target": "toggle",
    "side_effect": True,
    "side_effect_basis": "network: a write request",
}
SAVING = compiled(
    [A_STEP, KEPT_SIDE_EFFECT],
    targets={"toggle": TOGGLE, "summary": SUMMARY},
    assertions=DETAILS.model_dump(mode="json", exclude_none=True)["assertions"],
)


def committed(root: Path, script: CompiledScript = DETAILS) -> Path:
    """The order spec under the spec root `root`, `script` committed as explore
    publishes it. The compiled file's path."""
    order_spec(root, SHIPPING)
    (root / ".compiled").mkdir()
    spec = load_project(root).specs["order"]
    target = root / ".compiled" / "order.json"
    script = for_spec(script, spec).model_copy(update={"spec_id": "order"})
    target.write_text(published_text(script))
    return target


def recorded(
    root: Path,
    patch: list[dict[str, Any]],
    *,
    run: str = "a-run",
    edit: Callable[[dict[str, Any]], object] | None = None,
) -> str:
    """A run record under `root` holding a proposal of `patch` on the compiled file
    as it is now, in the form `aqa run` writes, then changed by `edit`, as a person
    could. The id its content had before that."""
    blob = blob_sha((root / ".compiled" / "order.json").read_bytes())
    derived = proposal_id("order", blob, patch)
    proposal = {
        "id": derived,
        "compiled": ".compiled/order.json",
        "base": {"blob_sha": blob, "pr_head_sha": None},
        "patch": patch,
    }
    if edit is not None:
        edit(proposal)
    file = root / ".aqa" / "runs" / run / "replay.json"
    file.parent.mkdir(parents=True)
    document = {"spec_id": "order", "heal_proposal": proposal}
    file.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return derived


def files(root: Path) -> dict[Path, bytes]:
    """Every file under `root`, by path, to show a refusal wrote none."""
    return {each: each.read_bytes() for each in root.rglob("*") if each.is_file()}


def apply(
    cwd: Path, monkeypatch: pytest.MonkeyPatch, *args: str
) -> typer.testing.Result:
    """`aqa heal apply <args>` in this process, from the directory `cwd`."""
    monkeypatch.chdir(cwd)
    return typer.testing.CliRunner().invoke(aqa, ["heal", "apply", *args])
