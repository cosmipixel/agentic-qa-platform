"""`aqa heal apply <proposal_id>` (API.md §7; ADR-0024's #247 amendment): a proposal
found by its id in the spec root's run records, checked again from nothing the file
says, and written to the compiled script. A proposal file is untrusted input, so these
write it by hand, as a person's edit would. No browser and no model: `test_run.py`
applies what real runs proposed."""

import fcntl
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import typer.testing
from aqa_cli import common
from aqa_cli import heal as heal_module
from aqa_cli.main import app as aqa
from aqa_core.compiled import CompiledScript
from aqa_core.heal_patch import blob_sha, proposal_id
from aqa_core.project import load_project
from aqa_runner import publication
from aqa_runner.publication import published_text
from playwright._impl._browser_type import BrowserType

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
# What a refusal says after the first line, by why.
EDITED = "the proposal doesn't match its id, so it was edited after its run"
NOT_ONE = "the proposal isn't one aqa run writes"


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
OUTSIDE = "a heal changes nothing but target locators and inserted steps"
A_SIDE_EFFECT = {
    "seq": 9,
    "action": "click",
    "target": "toggle",
    "side_effect": True,
    "side_effect_basis": "network: a write request",
}
A_SAFE_STEP = {"seq": 9, "action": "click", "target": "toggle", "side_effect": False}


def replacing(path: str, value: object) -> list[dict[str, Any]]:
    return [{"op": "replace", "path": path, "value": value}]


def inserting(step: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"op": "add", "path": "/steps/0", "value": step}]


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


def test_apply_changes_only_the_paths_it_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    before = target.read_text()
    assert LOCATORS in before
    proposal = recorded(root, REBIND)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 0, result.output
    assert target.read_text() == before.replace(LOCATORS, MOVED_LOCATORS)
    assert result.stdout == (
        f"applied {proposal} order: {target}\n  /targets/toggle/locators\n"
    )
    assert result.stderr == ""


@pytest.mark.parametrize(
    "patch",
    [
        # One the validator allows: only the id can tell it from what was reviewed.
        pytest.param(
            [
                {
                    "op": "replace",
                    "path": "/targets/toggle/locators",
                    "value": [{"css": "#decoy"}],
                }
            ],
            id="a-decoy-locator",
        ),
        pytest.param(
            [{"op": "replace", "path": "/assertions/0/text", "value": "anything"}],
            id="an-assertion",
        ),
    ],
)
def test_a_patch_edited_after_its_run_is_refused_by_its_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch: list[dict[str, Any]]
) -> None:
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(root, REBIND, edit=lambda held: held.update(patch=patch))
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert result.stderr == (
        f"spec_error {proposal}: nothing was applied\n  {EDITED}\n"
    )
    assert result.stdout == ""
    assert files(tmp_path) == before


def test_a_base_edited_to_match_a_changed_script_is_refused_by_its_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    typed = proposal_id("order", blob_sha(target.read_bytes()), REBIND)
    target.write_text(target.read_text() + "\n")
    recorded(root, REBIND, edit=lambda held: held.update(id=typed))
    before = files(tmp_path)

    result = apply(root, monkeypatch, typed)

    assert result.exit_code == 5, result.output
    assert EDITED in result.stderr
    assert files(tmp_path) == before


def test_a_rewritten_file_isnt_found_under_its_old_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    typed = proposal_id("order", blob_sha(target.read_bytes()), REBIND)
    recorded(
        root,
        [
            {
                "op": "replace",
                "path": "/targets/toggle/locators",
                "value": [{"css": "#d"}],
            }
        ],
    )
    before = files(tmp_path)

    result = apply(root, monkeypatch, typed)

    assert result.exit_code == 5, result.output
    assert f"  no run record under {root} holds it\n" in result.stderr
    assert files(tmp_path) == before


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        pytest.param(
            lambda held: held.update(compiled="../../evil.json"),
            EDITED,
            id="compiled-out-of-the-root",
        ),
        pytest.param(
            lambda held: held.update(compiled=".compiled/other.json"),
            EDITED,
            id="compiled-another-file",
        ),
        pytest.param(
            lambda held: held["base"].update(pr_head_sha="abc123"),
            NOT_ONE,
            id="a-pr-head",
        ),
        pytest.param(lambda held: held.update(note="hi"), NOT_ONE, id="an-extra-key"),
        pytest.param(lambda held: held.pop("base"), NOT_ONE, id="no-base"),
        pytest.param(lambda held: held.update(patch="x"), NOT_ONE, id="no-patch-list"),
    ],
)
def test_edited_fields_outside_the_id_are_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edit: Callable[[dict[str, Any]], object],
    reason: str,
) -> None:
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(root, REBIND, edit=edit)
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert result.stderr == (
        f"spec_error {proposal}: nothing was applied\n  {reason}\n"
    )
    assert files(tmp_path) == before


@pytest.mark.parametrize(
    "change",
    [
        {"op": "remove", "path": "/targets/toggle/locators"},
        {"op": "move", "path": "/targets/toggle/locators", "from": "/steps"},
        {"op": "copy", "path": "/targets/toggle/locators", "from": "/steps"},
        {"op": "test", "path": "/targets/toggle/locators", "value": []},
        {"op": "replace", "path": "/targets/toggle/locators"},
        {"op": "replace", "path": "/targets/toggle/locators", "value": [], "from": "/"},
        {"op": "replace", "path": 7, "value": []},
    ],
    ids=["remove", "move", "copy", "test", "no-value", "an-extra-key", "a-number-path"],
)
def test_a_proposal_takes_only_a_heals_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    """Each with the id its content has, so only the proposal's shape can refuse."""
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(root, [change])
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert f"  {NOT_ONE}\n" in result.stderr
    assert files(tmp_path) == before


def test_two_runs_that_made_one_proposal_apply_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    before = target.read_text()
    proposal = recorded(root, REBIND, run="first")
    assert recorded(root, REBIND, run="second") == proposal

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 0, result.output
    assert target.read_text() == before.replace(LOCATORS, MOVED_LOCATORS)


def test_two_records_that_differ_under_one_id_are_refused_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(root, REBIND, run="first")
    recorded(root, REBIND, run="second", edit=lambda held: held.update(patch=[]))
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    runs = root / ".aqa" / "runs"
    assert result.stderr == (
        f"spec_error {proposal}: nothing was applied\n"
        "  run records hold different proposals under this id, so at least one "
        "was edited\n"
        f"  {runs / 'first'}\n"
        f"  {runs / 'second'}\n"
    )
    assert files(tmp_path) == before


def test_an_id_no_run_record_holds_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    committed(root)
    recorded(root, REBIND)
    before = files(tmp_path)

    result = apply(root, monkeypatch, "ah_0123456789ab")

    assert result.exit_code == 5, result.output
    assert result.stderr == (
        "spec_error ah_0123456789ab: nothing was applied\n"
        f"  no run record under {root} holds it\n"
    )
    assert files(tmp_path) == before


def test_a_project_with_no_run_records_holds_no_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    committed(root)

    result = apply(root, monkeypatch, "ah_0123456789ab")

    assert result.exit_code == 5, result.output
    assert f"  no run record under {root} holds it\n" in result.stderr


@pytest.mark.parametrize(
    "typed",
    [
        "ah_",
        "ah_0123456789a",
        "ah_0123456789abc",
        "ah_0123456789AB",
        "xx_0123456789ab",
        "ah_0123456789ab\n",
        "../../etc/passwd",
    ],
)
def test_a_malformed_id_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, typed: str
) -> None:
    root = tmp_path / "qa"
    committed(root)
    before = files(tmp_path)

    result = apply(root, monkeypatch, typed)

    assert result.exit_code == 2, result.output
    assert files(tmp_path) == before


@pytest.mark.parametrize(
    "broken",
    [
        b"not json\n",
        b"\xff\xfe not utf-8",
        b"[" * 1_000_000,
        b"\xef\xbb\xbf{}",
    ],
    ids=["not-json", "not-utf-8", "nested-too-deep", "a-byte-order-mark"],
)
def test_a_record_that_cant_be_read_is_counted_and_stops_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: bytes
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    before = target.read_text()
    proposal = recorded(root, REBIND, run="a-good-run")
    other = root / ".aqa" / "runs" / "a-broken-run" / "replay.json"
    other.parent.mkdir()
    other.write_bytes(broken)

    missing = apply(root, monkeypatch, "ah_0123456789ab")
    applied = apply(root, monkeypatch, proposal)

    assert missing.exit_code == 5, missing.output
    assert (
        f"  no run record under {root} holds it, and 1 couldn't be read\n"
        in missing.stderr
    )
    assert applied.exit_code == 0, applied.output
    assert target.read_text() == before.replace(LOCATORS, MOVED_LOCATORS)


@pytest.mark.parametrize(
    "unrelated",
    ["null", "[]", '{"heal_proposal": 7}', '{"heal_proposal": {"id": 7}}', "{}"],
)
def test_a_record_holding_no_such_proposal_is_passed_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unrelated: str
) -> None:
    root = tmp_path / "qa"
    committed(root)
    other = root / ".aqa" / "runs" / "another-run" / "replay.json"
    other.parent.mkdir(parents=True)
    other.write_text(unrelated)

    result = apply(root, monkeypatch, "ah_0123456789ab")

    assert result.exit_code == 5, result.output
    assert f"  no run record under {root} holds it\n" in result.stderr


def test_a_run_with_no_document_or_a_file_for_a_run_is_passed_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    committed(root)
    runs = root / ".aqa" / "runs"
    (runs / "an-explore" / "replay.json").mkdir(parents=True)
    (runs / "a-file").write_text("not a run\n")

    result = apply(root, monkeypatch, "ah_0123456789ab")

    assert result.exit_code == 5, result.output
    assert f"  no run record under {root} holds it\n" in result.stderr


@pytest.mark.parametrize("linked", [".aqa", ".aqa/runs"])
def test_a_link_at_the_records_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked: str
) -> None:
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(root, REBIND)
    place = root / linked
    place.rename(tmp_path / "elsewhere")
    place.symlink_to(tmp_path / "elsewhere")
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert result.stderr == (
        f"spec_error {proposal}: nothing was applied\n"
        f"  {place}: a link, and a run record stays in {root}\n"
    )
    assert files(tmp_path) == before


@pytest.mark.parametrize("linked", ["a-run", "a-run/replay.json"])
def test_a_linked_record_holds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked: str
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    before = target.read_bytes()
    proposal = recorded(root, REBIND)
    place = root / ".aqa" / "runs" / linked
    place.rename(tmp_path / "elsewhere")
    place.symlink_to(tmp_path / "elsewhere")

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert f"  no run record under {root} holds it\n" in result.stderr
    assert target.read_bytes() == before


def test_run_records_that_cant_be_listed_exit_15(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    before = target.read_bytes()
    proposal = recorded(root, REBIND)
    runs = root / ".aqa" / "runs"
    runs.chmod(0o000)

    try:
        result = apply(root, monkeypatch, proposal)
    finally:
        runs.chmod(0o755)

    assert result.exit_code == 15, result.output
    assert result.stderr == (
        f"record_error {proposal}: the project or its run records could not be "
        "read, so nothing was applied\n"
    )
    assert target.read_bytes() == before


@pytest.mark.parametrize("spec_id", ["../../evil", "another", "order/../order"])
def test_a_proposal_naming_no_spec_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec_id: str
) -> None:
    root = tmp_path / "qa"
    committed(root)
    recorded(root, REBIND)
    blob = blob_sha((root / ".compiled" / "order.json").read_bytes())
    proposal = proposal_id(spec_id, blob, REBIND)
    file = root / ".aqa" / "runs" / "a-run" / "replay.json"
    document = {
        "spec_id": spec_id,
        "heal_proposal": {
            "id": proposal,
            "compiled": f".compiled/{spec_id}.json",
            "base": {"blob_sha": blob, "pr_head_sha": None},
            "patch": REBIND,
        },
    }
    file.write_text(json.dumps(document))
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert result.stderr == (
        f"spec_error {proposal}: nothing was applied\n"
        "  it names no spec of this project\n"
    )
    assert files(tmp_path) == before


def test_a_spec_id_that_isnt_text_holds_no_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(root, REBIND)
    file = root / ".aqa" / "runs" / "a-run" / "replay.json"
    document = json.loads(file.read_text())
    file.write_text(json.dumps({**document, "spec_id": ["order"]}))

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert f"  no run record under {root} holds it\n" in result.stderr


@pytest.mark.parametrize(
    "cwd",
    [".", "qa", "qa/.compiled", "src/deeper"],
    ids=["the-repo-root", "the-spec-root", "below-the-spec-root", "beside-it"],
)
def test_apply_finds_the_spec_root_from_the_repo_root_and_below_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cwd: str
) -> None:
    """A repo's own `config.yaml`, which isn't one, doesn't win over `qa/`."""
    root = tmp_path / "qa"
    target = committed(root)
    before = target.read_text()
    proposal = recorded(root, REBIND)
    (tmp_path / "config.yaml").write_text("not_a_key_aqa_reads: true\n")
    (tmp_path / "src" / "deeper").mkdir(parents=True)

    result = apply(tmp_path / cwd, monkeypatch, proposal)

    assert result.exit_code == 0, result.output
    assert target.read_text() == before.replace(LOCATORS, MOVED_LOCATORS)


def test_a_changed_script_is_stale_exits_3_and_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    proposal = recorded(root, REBIND)
    target.write_text(target.read_text() + "\n")
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 3, result.output
    assert result.stderr == (
        f"stale {proposal}: the compiled script changed since this proposal was "
        "made, so nothing was applied: run aqa run again\n"
    )
    assert result.stdout == ""
    assert files(tmp_path) == before


def test_a_proposal_applied_once_is_stale_the_second_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    before = target.read_text()
    proposal = recorded(root, REBIND)

    first = apply(root, monkeypatch, proposal)
    second = apply(root, monkeypatch, proposal)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 3, second.output
    assert target.read_text() == before.replace(LOCATORS, MOVED_LOCATORS)


def test_a_script_changed_after_its_read_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    proposal = recorded(root, REBIND)
    changed = target.read_text() + "\n"

    async def changing_it_first(*args: Any, **kwargs: Any) -> None:
        target.write_text(changed)
        await publication.publish_text(*args, **kwargs)

    monkeypatch.setattr(heal_module, "publish_text", changing_it_first)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 3, result.output
    assert result.stderr == (
        f"gave_up {proposal}: nothing was applied\n"
        "  the compiled script changed during the run\n"
    )
    assert target.read_text() == changed


def test_a_held_lock_exits_3_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    before = target.read_bytes()
    proposal = recorded(root, REBIND)
    lock = root / ".aqa" / "locks" / "order.lock"
    lock.parent.mkdir()
    monkeypatch.setattr(publication, "LOCK_SECONDS", 0.05)

    with lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 3, result.output
    assert result.stderr == (
        f"gave_up {proposal}: nothing was applied\n"
        "  another aqa process held this spec's publication lock\n"
    )
    assert target.read_bytes() == before


def test_a_write_that_fails_leaves_the_file_and_exits_15(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    before = target.read_bytes()
    proposal = recorded(root, REBIND)
    target.parent.chmod(0o555)

    try:
        result = apply(root, monkeypatch, proposal)
    finally:
        target.parent.chmod(0o755)

    assert result.exit_code == 15, result.output
    assert result.stderr == (
        f"record_error {proposal}: the compiled script couldn't be written\n"
    )
    assert target.read_bytes() == before
    assert [each.name for each in target.parent.iterdir()] == ["order.json"]


def test_a_fault_after_installing_says_the_script_was_written_and_exits_15(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(root, REBIND)

    async def failing_after_it(*_: object, **__: object) -> None:
        raise publication.PublishWriteError(installed=True)

    monkeypatch.setattr(heal_module, "publish_text", failing_after_it)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 15, result.output
    assert result.stderr == (
        f"record_error {proposal}: the compiled script was published, but the file "
        "system failed after it, so it may not survive a crash\n"
    )


def test_a_link_at_the_compiled_script_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    proposal = recorded(root, REBIND)
    elsewhere = tmp_path / "elsewhere.json"
    target.rename(elsewhere)
    target.symlink_to(elsewhere)
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert f"  {target}: a link, and explore writes only in {root}\n" in result.stderr
    assert files(tmp_path) == before


def test_a_compiled_script_gone_since_the_run_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    proposal = recorded(root, REBIND)
    target.unlink()

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert result.stderr == "spec_error order: no compiled script: explore it first\n"
    assert not target.exists()


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        pytest.param(
            replacing("/assertions/0/text", "anything"),
            f"/assertions: {OUTSIDE}",
            id="an-assertions-text",
        ),
        pytest.param(
            replacing("/assertions/1/min_size_px", [1, 2]),
            f"/assertions: {OUTSIDE}",
            id="an-assertions-bound",
        ),
        pytest.param(
            replacing("/targets/toggle/semantic", "anything"),
            "/targets/toggle/semantic: a heal never changes a target's meaning",
            id="a-meaning",
        ),
        pytest.param(
            [
                {
                    "op": "add",
                    "path": "/targets/toggle/contract",
                    "value": {"region": "main", "part": "button"},
                }
            ],
            "/targets/toggle/contract: a heal never changes a subject contract, "
            "which only the reviewed config gives",
            id="a-contract",
        ),
        pytest.param(
            [
                *replacing("/steps/0/side_effect", True),
                {"op": "add", "path": "/steps/0/side_effect_basis", "value": "a basis"},
            ],
            "/steps/0/side_effect: a heal rewrites no existing step\n"
            "  /steps/0/side_effect_basis: a heal rewrites no existing step",
            id="a-replay-safe-step-made-a-side-effect",
        ),
        pytest.param(
            replacing("/steps/1/side_effect_basis", "a basis"),
            "/steps/1/side_effect_basis: a heal rewrites no existing step",
            id="a-side-effects-basis",
        ),
        pytest.param(
            replacing("/steps/1/target", "summary"),
            "/steps/1/target: a heal rewrites no existing step",
            id="a-side-effect-steps-target",
        ),
        pytest.param(
            replacing("/steps", [KEPT_SIDE_EFFECT]),
            "/steps/0: a heal removes no step",
            id="a-step-removed",
        ),
        pytest.param(
            replacing("/steps", [KEPT_SIDE_EFFECT, A_STEP]),
            "/steps: a heal reorders no step",
            id="the-steps-reordered",
        ),
        pytest.param(
            inserting(A_SIDE_EFFECT),
            "/steps/0/side_effect: a heal inserts only replay-safe steps",
            id="an-inserted-side-effect",
        ),
        pytest.param(
            inserting(
                {"seq": 9, "action": "navigate", "url": "/x", "side_effect": False}
            ),
            "/steps/0/action: a heal inserts no navigation, which could route "
            "around a path that is broken for its users",
            id="an-inserted-navigate",
        ),
        pytest.param(
            inserting({"seq": 9, "action": "reload", "side_effect": False}),
            "/steps/0/action: a heal inserts no reload, which loads the page again "
            "and could route around a path that is broken for its users",
            id="an-inserted-reload",
        ),
        pytest.param(
            replacing("/spec_hash", "sha256:" + "1" * 64),
            f"/spec_hash: {OUTSIDE}",
            id="the-spec-hash",
        ),
        pytest.param(
            replacing("/compiled_by/price_map", "another"),
            f"/compiled_by: {OUTSIDE}",
            id="provenance",
        ),
        pytest.param(
            replacing("/browser/timezone", "Asia/Tokyo"),
            f"/browser: {OUTSIDE}",
            id="the-browser",
        ),
        pytest.param(
            replacing("/coverage/expectations/0/claim", "anything"),
            f"/coverage: {OUTSIDE}",
            id="the-coverage",
        ),
        pytest.param(
            replacing("/confirmed", False), f"/confirmed: {OUTSIDE}", id="confirmed"
        ),
        pytest.param(
            [
                {
                    "op": "add",
                    "path": "/targets/new",
                    "value": {"semantic": "new", "locators": [{"css": "#n"}]},
                }
            ],
            "/targets/new: a heal adds no target",
            id="a-new-target",
        ),
        pytest.param(
            replacing("/targets/toggle/locators", [{"css": "li:nth-child(2) a"}]),
            "/targets/toggle/locators/0/css: a heal finds no element by its position "
            "among its siblings",
            id="a-positional-locator",
        ),
        pytest.param(
            replacing("/targets/summary/locators", [{"label": "Summary"}]),
            "/targets/summary/locators/0: a heal finds no assertion's target by a "
            "label or placeholder, or by a name when its text is checked",
            id="an-assertions-target-by-a-label",
        ),
    ],
)
def test_a_patch_the_validator_refuses_changes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch: list[dict[str, Any]],
    reason: str,
) -> None:
    """Each with the id its content has, so only the validator can refuse."""
    root = tmp_path / "qa"
    committed(root, SAVING)
    proposal = recorded(root, patch)
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert result.stderr == (
        f"spec_error {proposal}: nothing was applied\n  {reason}\n"
    )
    assert result.stdout == ""
    assert files(tmp_path) == before


@pytest.mark.parametrize(
    ("patch", "said"),
    [
        pytest.param(
            replacing("/steps/1/side_effect", False),
            "steps[1]",
            id="a-side-effect-flag-lowered",
        ),
        pytest.param(
            replacing("/targets/toggle/locators", []),
            "targets.toggle.locators",
            id="no-locators",
        ),
        pytest.param(
            replacing("/targets/toggle/locators", [{"css": "\ud800"}]),
            "unable to parse raw data as a unicode string",
            id="a-lone-surrogate",
        ),
        pytest.param(
            replacing("/targets/toggle/locators", [{"css": float("nan")}]),
            "targets.toggle.locators",
            id="not-a-number",
        ),
        pytest.param(
            [{"op": "add", "path": "/targets/toggle/unknown", "value": 1}],
            "targets.toggle.unknown",
            id="an-unknown-field",
        ),
        pytest.param(inserting(A_STEP), "steps[1].seq", id="a-seq-already-used"),
        pytest.param(
            inserting(
                {
                    "seq": 9,
                    "action": "fill_secret",
                    "target": "toggle",
                    "secret": "TEST_PASSWORD",
                    "side_effect": False,
                }
            ),
            "secret TEST_PASSWORD is not declared",
            id="an-inserted-fill-secret",
        ),
    ],
)
def test_a_patch_the_loader_refuses_changes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch: list[dict[str, Any]],
    said: str,
) -> None:
    root = tmp_path / "qa"
    target = committed(root, SAVING)
    proposal = recorded(root, patch)
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert result.stderr.startswith(
        f"spec_error {proposal}: nothing was applied\n  {target}: "
    )
    assert said in result.stderr
    assert files(tmp_path) == before


def test_a_path_the_patch_cant_follow_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(root, replacing("/missing/deeper", 1))
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert result.stderr == (
        f"spec_error {proposal}: nothing was applied\n"
        "  /missing/deeper: a patch can't follow this path\n"
    )
    assert files(tmp_path) == before


def test_a_spec_edited_since_the_run_is_refused_as_replay_refuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(root, REBIND)
    spec = root / "order.spec.md"
    spec.write_text(spec.read_text().replace(SHIPPING, "The order says what it holds"))
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert result.stderr == (
        f"spec_error {proposal}: nothing was applied\n"
        "  compiled from another version of the spec: explore it again\n"
    )
    assert files(tmp_path) == before


def test_subject_contracts_changed_since_the_run_are_refused_as_replay_refuses_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(root, REBIND)
    (root / "config.yaml").write_text(
        "budgets: { resolve_seconds: 1 }\n"
        "subjects:\n"
        "  - { spec: order, expect: 0, region: main, part: p }\n"
    )
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert "compiled under other subject contracts: explore it again" in result.stderr
    assert files(tmp_path) == before


def test_an_inserted_step_replay_wouldnt_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(
        root,
        inserting({"seq": 9, "action": "press", "key": "a+b", "side_effect": False}),
    )
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert result.stderr == (
        f"spec_error {proposal}: nothing was applied\n"
        "  steps[0] (seq 9): press takes one key, with only modifiers held before "
        "it: 'a+b'\n"
    )
    assert files(tmp_path) == before


def test_a_refusal_escapes_control_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(
        root,
        [
            {
                "op": "add",
                "path": "/targets/\x1b[31mred",
                "value": {"semantic": "new", "locators": [{"css": "#n"}]},
            }
        ],
    )

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert "\x1b" not in result.stderr
    assert "  /targets/\\x1b[31mred: a heal adds no target\n" in result.stderr


def test_apply_keeps_a_byte_order_mark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    published = target.read_text()
    target.write_bytes(b"\xef\xbb\xbf" + published.encode())
    proposal = recorded(root, REBIND)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 0, result.output
    assert target.read_bytes() == (
        b"\xef\xbb\xbf" + published.replace(LOCATORS, MOVED_LOCATORS).encode()
    )


def test_an_inserted_step_applies_where_the_patch_puts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root, SAVING)
    before = json.loads(target.read_text())
    proposal = recorded(root, [{"op": "add", "path": "/steps/1", "value": A_SAFE_STEP}])

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 0, result.output
    after = json.loads(target.read_text())
    assert [step["seq"] for step in after["steps"]] == [1, 9, 2]
    assert after["steps"][0] == before["steps"][0]
    assert after["steps"][2] == before["steps"][1]
    del after["steps"][1]
    assert after == before
    assert result.stdout == f"applied {proposal} order: {target}\n  /steps/1\n"


def in_its_own_order(root: Path) -> dict[str, Any]:
    """The committed script as a person's edit could leave it, every path still
    what a compiled script may hold, but none as `published_text` writes it: its
    top-level keys reversed, a default left out, null written, a time respelled and
    a name escaped. What it holds."""
    target = root / ".compiled" / "order.json"
    document = json.loads(target.read_text())
    reordered = dict(reversed(list(document.items())))
    del reordered["steps"][0]["satisfies"]
    reordered["targets"]["summary"]["contract"] = None
    reordered["compiled_at"] = "2026-10-02T00:00:00+00:00"
    reordered["targets"]["toggle"]["semantic"] = "the button that shows the détails"
    return reordered


def test_a_script_in_another_order_keeps_every_other_path_and_the_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    document = in_its_own_order(root)
    target.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    before = json.loads(target.read_text())
    proposal = recorded(root, REBIND)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 0, result.output
    after = json.loads(target.read_text())
    before["targets"]["toggle"]["locators"] = MOVED
    assert after == before
    assert list(after) == list(before)
    assert result.stdout == (
        f"applied {proposal} order: {target}\n  /targets/toggle/locators\n"
    )


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda d: json.dumps(d), id="on-one-line"),
        pytest.param(
            lambda d: json.dumps(d, indent=2, ensure_ascii=False).replace("\n", "\r\n"),
            id="crlf",
        ),
        pytest.param(lambda d: json.dumps(d, indent=2), id="no-final-newline"),
    ],
)
def test_a_script_reformatted_by_hand_keeps_every_other_path_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write: Callable[[Any], str]
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    target.write_bytes(write(in_its_own_order(root)).encode())
    before = json.loads(target.read_text())
    proposal = recorded(root, REBIND)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 0, result.output
    after = json.loads(target.read_text())
    before["targets"]["toggle"]["locators"] = MOVED
    assert after == before
    assert list(after) == list(before)
    assert result.stdout == (
        f"applied {proposal} order: {target}\n"
        "  /targets/toggle/locators\n"
        "  the compiled script wasn't in the form aqa writes, so its whitespace "
        "changed too\n"
    )


def test_a_reformatted_script_with_a_byte_order_mark_keeps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    target.write_bytes(b"\xef\xbb\xbf" + json.dumps(in_its_own_order(root)).encode())
    proposal = recorded(root, REBIND)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 0, result.output
    assert target.read_bytes().startswith(b"\xef\xbb\xbf{\n")


def test_apply_needs_no_providers_key_and_starts_no_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def refused(*_: object, **__: object) -> None:
        raise AssertionError("the driver was started")

    monkeypatch.setattr(common, "async_playwright", refused)
    monkeypatch.setattr(BrowserType, "launch", refused)
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(root, REBIND)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 0, result.output


def test_a_spec_root_is_needed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = apply(tmp_path, monkeypatch, "ah_0123456789ab")

    assert result.exit_code == 5, result.output
    assert f"  {tmp_path}: no config.yaml in it, in its qa/ or above it\n" in (
        result.stderr
    )


def test_a_project_that_doesnt_load_exits_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    committed(root)
    proposal = recorded(root, REBIND)
    (root / "config.yaml").write_text("not_a_key_aqa_reads: true\n")
    before = files(tmp_path)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert result.stderr.startswith(f"spec_error {proposal}: nothing was applied\n")
    assert files(tmp_path) == before
