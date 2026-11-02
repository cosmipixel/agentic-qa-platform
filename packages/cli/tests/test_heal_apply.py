"""`aqa heal apply <proposal_id>` (API.md §7; ADR-0024's #247 amendment): a proposal
found by its id in the spec root's run records, refused as stale when the compiled
script moved on, and written to it without touching anything else.
`test_heal_apply_refusals.py` holds what it refuses, and `test_run.py` applies what
real runs proposed. No browser and no model."""

import fcntl
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from aqa_cli import common
from aqa_cli import heal as heal_module
from aqa_core.compiled import CompiledScript
from aqa_core.config import ProjectConfig
from aqa_core.heal_patch import blob_sha, proposal_id
from aqa_core.project import read_compiled
from aqa_runner import publication
from playwright._impl._browser_type import BrowserType

from packages.cli.tests.heal_apply_fixtures import (
    LOCATORS,
    MOVED,
    MOVED_LOCATORS,
    REBIND,
    SAVING,
    apply,
    committed,
    files,
    recorded,
)

A_SAFE_STEP = {"seq": 9, "action": "click", "target": "toggle", "side_effect": False}


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


def test_a_spec_root_is_needed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = apply(tmp_path, monkeypatch, "ah_0123456789ab")

    assert result.exit_code == 5, result.output
    assert f"  {tmp_path}: no config.yaml in it, in its qa/ or above it\n" in (
        result.stderr
    )


@pytest.mark.parametrize(
    "config",
    ["budgets: { resolve_seconds: 1 }\n", "not_a_key_aqa_reads: true\n"],
    ids=["a-project-that-loads", "one-that-doesnt"],
)
def test_a_spec_root_that_is_a_link_is_refused_before_it_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str
) -> None:
    """A repository can commit `qa` as a link into another checkout, where the write
    would land. The second case shows nothing behind the link was read."""
    other = tmp_path / "other" / "qa"
    committed(other)
    proposal = recorded(other, REBIND)
    (other / "config.yaml").write_text(config)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "qa").symlink_to("../other/qa")
    before = files(tmp_path)

    result = apply(checkout, monkeypatch, proposal)

    assert result.exit_code == 5, result.output
    assert result.stderr == (
        f"spec_error {proposal}: nothing was applied\n"
        f"  {checkout / 'qa'}: a link, which could lead into another checkout: run "
        "this inside the project you mean\n"
    )
    assert result.stdout == ""
    assert files(tmp_path) == before


def test_a_spec_root_that_becomes_a_link_writes_nothing_through_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qa"
    target = committed(root)
    written = target.read_bytes()
    proposal = recorded(root, REBIND)
    moved = tmp_path / "moved"

    async def linking_it_first(*args: Any, **kwargs: Any) -> None:
        root.rename(moved)
        root.symlink_to(moved)
        await publication.publish_text(*args, **kwargs)

    monkeypatch.setattr(heal_module, "publish_text", linking_it_first)

    result = apply(root, monkeypatch, proposal)

    assert result.exit_code == 3, result.output
    assert result.stderr == (
        f"gave_up {proposal}: nothing was applied\n"
        "  a directory or file explore publishes through became a link\n"
    )
    assert (moved / ".compiled" / "order.json").read_bytes() == written
    assert not (moved / ".aqa" / "locks").exists()


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
    """The change lands just after the read, so it is refused only if what the lock's
    comparison holds was found before that read."""
    root = tmp_path / "qa"
    target = committed(root)
    proposal = recorded(root, REBIND)
    changed = target.read_text() + "\n"

    def changing_it_after(
        path: Path, config: ProjectConfig
    ) -> tuple[CompiledScript, bytes]:
        read = read_compiled(path, config)
        target.write_text(changed)
        return read

    monkeypatch.setattr(common, "read_compiled", changing_it_after)

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
    a non-ASCII meaning. What it holds."""
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
        pytest.param(json.dumps, id="on-one-line"),
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
        "  the compiled script wasn't in the form aqa writes, so its formatting "
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
