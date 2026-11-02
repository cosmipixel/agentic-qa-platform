"""What a run that heals says of itself (#244; ADR-0024's #244 amendment): real
Chromium against #240's drift pages, a scripted model, through
`replays.replay_run` and `replay_result.present`. `test_replay_result.py` holds
the rules by hand; these hold what a real heal leaves."""

from pathlib import Path
from typing import Any

from aqa_core.compiled import CompiledScript

from packages.runner.tests.drift_fixtures import (
    BUTTON,
    DETAILS,
    IN_VIEW,
    SUMMARY,
    TOGGLE,
    order_spec,
)
from packages.runner.tests.executor_fixtures import compiled
from packages.runner.tests.explore_graph_fixtures import NO_RESPONSE
from packages.runner.tests.heal_fixtures import (
    BLOB,
    FOUND_NONE,
    MOVED,
    NO_TOOL,
    SHOW,
    Healed,
    healed,
    rebinding,
)

THE_TOGGLE_MOVED = [
    {"role": "button", "name": "Show details"},
    {"css": "button", "scope": {"css": "#order"}},
]
SHOWN = MOVED.replace(" hidden>", ">")
THE_SUMMARY = rebinding("paragraph", "Order 42 ships on Monday")
# The summary's two checks, with a locator that finds nothing, so both drift on
# one target.
TEXT_CHECK = {
    "id": "a1",
    "expect_index": 0,
    "check": "text_in_target",
    "target": "summary",
    "text": "ships on Monday",
}
SEEN_CHECK = {
    "id": "a2",
    "expect_index": 0,
    "check": "visible_unoccluded",
    "target": "summary",
    "min_size_px": [1, 1],
    "in_viewport": True,
}
SUMMARY_GONE = compiled(
    [],
    targets={"summary": {**SUMMARY, "locators": [{"css": "#gone"}]}},
    assertions=[TEXT_CHECK, SEEN_CHECK],
)


def in_view_at(root: Path) -> Path:
    return order_spec(root, IN_VIEW)


def paths_cited(run: Healed) -> list[str]:
    """Each path a verdict's `heal` cites, each existing in the run's record."""
    cited: list[str] = []
    for verdict in run.document["verdicts"]:
        heal: dict[str, Any] | None = verdict["heal"]
        if heal is not None:
            cited.extend(path for path in (heal["repair"], heal["record"]) if path)
    return [path for path in cited if not (run.record / path).exists()]


def test_a_healed_step_is_drift_consistent_and_proposes_its_patch(
    tmp_path: Path,
) -> None:
    run = healed(tmp_path, SHOW)

    written = run.document
    assert (written["mode"], written["outcome"], written["exit_code"]) == (
        "run",
        "drift_consistent",
        7,
    )
    assert written["reasons"] == []
    assert run.verdict("step", 1)["verdict"] == "drift_consistent"
    assert run.verdict("step", 1)["heal"] == {
        "repair": "repairs/1.json",
        "record": "heals/1",
        "bound": 'button "Show details"',
        "model_reported": {"confidence": 0.9, "rationale": "it moved"},
    }
    # Its summary's checks read another target, so they just pass.
    assert run.verdict("assertion", "a1")["verdict"] == "pass"
    assert run.verdict("assertion", "a1")["heal"] is None
    proposal = written["heal_proposal"]
    assert proposal["base"] == {"blob_sha": BLOB, "pr_head_sha": None}
    assert proposal["patch"] == [
        {
            "op": "replace",
            "path": "/targets/toggle/locators",
            "value": THE_TOGGLE_MOVED,
        }
    ]
    assert paths_cited(run) == []


def test_a_healed_assertion_target_is_drift_consistent_and_cites_its_record(
    tmp_path: Path,
) -> None:
    run = healed(tmp_path, SHOW, script=BUTTON, spec_at=in_view_at)

    assert run.document["outcome"] == "drift_consistent"
    heal = run.verdict("assertion", "a1")["heal"]
    assert (heal["repair"], heal["record"]) == (
        "repairs/assertion-0.json",
        "heals/assertion-0",
    )
    assert paths_cited(run) == []


def test_a_repaired_assertion_that_fails_its_check_is_expectation_violated(
    tmp_path: Path,
) -> None:
    """The summary's id changed and its date regressed: heal rebinds it, and the
    check, read through the new binding, says it isn't Monday."""
    html = SHOWN.replace("ships on Monday", "ships on Friday")

    run = healed(
        tmp_path,
        rebinding("paragraph", "Order 42 ships on Friday"),
        script=SUMMARY_GONE,
        html=html,
    )

    written = run.document
    assert (written["outcome"], written["exit_code"]) == ("expectation_violated", 1)
    assert written["heal_proposal"] is None
    entry = run.verdict("assertion", "a1")
    assert entry["verdict"] == "expectation_violated"
    assert entry["finding"]["category"] == "functional"
    assert entry["heal"]["repair"] == "repairs/assertion-0.json"


def test_a_second_assertion_on_a_rebound_target_cites_the_repair(
    tmp_path: Path,
) -> None:
    run = healed(tmp_path, THE_SUMMARY, script=SUMMARY_GONE, html=SHOWN)

    assert run.document["outcome"] == "drift_consistent"
    for check in ("a1", "a2"):
        entry = run.verdict("assertion", check)
        assert entry["verdict"] == "drift_consistent"
        assert entry["heal"]["repair"] == "repairs/assertion-0.json"
    assert [each.at.key for each in run.result.repairs] == ["assertion-0"]


def test_a_rebound_step_in_a_run_that_fails_is_inconclusive_with_no_proposal(
    tmp_path: Path,
) -> None:
    html = MOVED.replace("ships on Monday", "ships on Friday")

    run = healed(tmp_path, SHOW, html=html)

    written = run.document
    assert (written["outcome"], written["heal_proposal"]) == (
        "expectation_violated",
        None,
    )
    assert run.verdict("step", 1)["verdict"] == "inconclusive"
    assert run.verdict("step", 1)["reason"] == (
        "heal rebound its target, but the run didn't pass"
    )
    assert run.verdict("assertion", "a1")["verdict"] == "expectation_violated"


def test_a_heal_that_found_nothing_is_inconclusive_and_cites_what_it_left(
    tmp_path: Path,
) -> None:
    run = healed(tmp_path, NO_TOOL)

    written = run.document
    assert (written["outcome"], written["exit_code"]) == ("inconclusive", 3)
    entry = run.verdict("step", 1)
    assert entry["reason"] == (
        f"drift: heal gave up: {FOUND_NONE} (not unanimous, no match)"
    )
    assert entry["heal"] == {
        "repair": "repairs/1.json",
        "record": "heals/1",
        "bound": None,
        "model_reported": None,
    }
    assert paths_cited(run) == []


def test_no_response_stops_the_run_as_11_and_keeps_the_verdicts(
    tmp_path: Path,
) -> None:
    run = healed(tmp_path, NO_RESPONSE)

    written = run.document
    assert (written["outcome"], written["exit_code"]) == ("no_response", 11)
    assert written["reasons"][0] == "the model gave no response (APIConnectionError)"
    assert run.verdict("step", 0)["verdict"] == "pass"
    assert run.verdict("step", 1)["verdict"] == "inconclusive"


def test_a_healer_steered_to_a_decoy_that_fails_the_check_gets_no_proposal(
    tmp_path: Path,
) -> None:
    run = healed(tmp_path, rebinding("button", "Contact us"))

    written = run.document
    assert (written["outcome"], written["heal_proposal"]) == (
        "expectation_violated",
        None,
    )
    assert run.verdict("assertion", "a1")["verdict"] == "expectation_violated"


def test_a_decoy_that_satisfies_a_weak_check_heals_today(tmp_path: Path) -> None:
    """The residual ADR-0024 names, and where #248 starts: `visible_unoccluded`
    with `min_size_px` [1, 1] holds of any button, so a steered model gets a
    proposal, whose evidence is the locator diff a person reviews."""
    run = healed(
        tmp_path,
        rebinding("button", "Contact us"),
        script=BUTTON,
        spec_at=in_view_at,
    )

    written = run.document
    assert written["outcome"] == "drift_consistent"
    assert written["heal_proposal"]["patch"] == [
        {
            "op": "replace",
            "path": "/targets/toggle/locators",
            "value": [{"role": "button", "name": "Contact us"}],
        }
    ]
    assert run.verdict("assertion", "a1")["heal"]["bound"] == 'button "Contact us"'


def test_the_committed_script_is_unchanged_by_a_heal(tmp_path: Path) -> None:
    before: CompiledScript = DETAILS.model_copy(deep=True)

    run = healed(tmp_path, SHOW)

    assert run.committed.targets["toggle"].locators == before.targets["toggle"].locators
    assert [
        locator.model_dump(mode="json", exclude_none=True)
        for locator in run.committed.targets["toggle"].locators
    ] == TOGGLE["locators"]
