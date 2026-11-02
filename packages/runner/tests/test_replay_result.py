"""`aqa_runner.replay_result`: the outcome a strict replay exits with when
several causes hold, per run and across specs (ADR-0024's #238 amendment),
and the verdict each step, assertion and invariant gets in its document,
`replay.json` (DATA_MODEL §7). Records and redactors are real, results are made by hand, and every
expected value is a literal. A result's own `outcome` stays `passed`: what
the run left decides, never the executor's word for it."""

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest
from aqa_core.compiled import Click, CompiledScript, Target, VisibleUnoccluded
from aqa_core.config import ProjectConfig
from aqa_core.spec import InvariantName, Spec
from aqa_runner.egress import InfrastructureEvent
from aqa_runner.egress_proxy import EgressBlocks, RefusedHost
from aqa_runner.evaluation import AssertionResult
from aqa_runner.executor import Evidence, RunResult, StepResult
from aqa_runner.invariants import InvariantResult
from aqa_runner.locators import Miss
from aqa_runner.redaction import NO_SECRETS, Redactor
from aqa_runner.repairs import AssertionAt, Basis, ModelReported, Repaired, StepAt
from aqa_runner.replay_result import (
    EXIT_CODES,
    Outcome,
    ReplayEnd,
    Stop,
    final_outcome,
    present,
    rolled_up,
    unrecorded,
)
from aqa_runner.run_record import RunRecord

from packages.runner.tests.executor_fixtures import a_spec, compiled
from packages.runner.tests.result_redaction_fixtures import fake_secret

APP = "http://app.example.test:8080"
BOUND = Redactor([fake_secret("TEST_PASSWORD", "fake-bound-value", APP)])
CLICK = {"action": "click", "target": "save"}
NAVIGATE = {"action": "navigate", "url": "/"}
CLEAN = EgressBlocks((), False)
CDN = EgressBlocks((RefusedHost("cdn.evil.test", 443),), False)
EVENT = InfrastructureEvent("api.evil.test", 8443, "the name didn't resolve")
SANDBOX = "Chromium's sandbox didn't start"
RESET = "the reset hook answered 500"
UNWRITTEN = "the run record couldn't be written"
SILENT = "the model gave no response (APIConnectionError)"
HELD = tuple(
    InvariantResult(name, "held", (), 0)
    for name in ("console_errors", "js_exceptions", "http_5xx", "broken_images")
)


@dataclass(frozen=True)
class Made:
    """A strict run's spec, script and record, scanning with a redactor."""

    spec: Spec
    script: CompiledScript
    record: RunRecord


def made(
    tmp_path: Path,
    redactor: Redactor = NO_SECRETS,
    *,
    second: str = "The save button shows",
) -> Made:
    """A run of a script that clicks save, then checks three things of the
    spec's first expectation and one of its second, `second`."""
    script = compiled(
        [{"seq": 1, "action": "click", "target": "save", "side_effect": False}],
        targets={"save": {"semantic": "the save button", "locators": [{"css": "b"}]}},
        assertions=[
            {"id": "a1", "expect_index": 0, "check": "url_matches", "pattern": "/"},
            {"id": "a2", "expect_index": 0, "check": "text_visible", "text": "saved"},
            {"id": "a3", "expect_index": 0, "check": "text_visible", "text": "done"},
            {
                "id": "a4",
                "expect_index": 1,
                "check": "visible_unoccluded",
                "target": "save",
                "min_size_px": [10, 10],
                "in_viewport": True,
            },
        ],
    )
    # `extra` follows the spec's first expectation, so it lists a second.
    listed = f"  - {second}\n"
    spec = a_spec(tmp_path, ProjectConfig(), start_url="/page/form", extra=listed)
    return Made(spec, script, RunRecord.create(tmp_path).redacting(redactor))


def ending(made: Made, **changes: Any) -> ReplayEnd:
    """How `made`'s run ended: replay returned, steps 0 and 1 completed, a1
    passed and every invariant held, with no block, event or stop, plus the
    steps, assertions and stops `changes` adds, the blocks and events it
    gives, and the result fields it sets."""
    stops = changes.pop("stops", ())
    blocks = changes.pop("blocks", CLEAN)
    events = changes.pop("infrastructure_events", ())
    base = changes.pop("base", None)
    result = RunResult(
        made.record.run_id,
        "passed",
        (StepResult(0, "completed"), StepResult(1, "completed", locator_index=0)),
        (AssertionResult("a1", "pass"),),
        (),
        (),
        HELD,
        CLEAN,
        None,
        0,
        Evidence(1, None),
    )
    result = replace(
        result,
        steps=result.steps + changes.pop("steps", ()),
        assertions=result.assertions + changes.pop("assertions", ()),
        **changes,
    )
    return ReplayEnd(
        made.record, made.spec, made.script, result, blocks, events, stops, base
    )


BLOB = "ce013625030ba8dba906f756967f9e9ca394464a"
REPORTED = ModelReported(0.9, NO_SECRETS.redact("it moved"))
BOUND_AS = NO_SECRETS.redact('button "Save"')
MISSED: tuple[Miss, ...] = ("no match",)
NEW_SAVE = {"role": "button", "name": "Save"}


def rebinding(script: CompiledScript, **locators: Any) -> CompiledScript:
    """`script` with each named target found by the one locator given."""
    return script.model_copy(
        update={
            "targets": {
                **script.targets,
                **{
                    name: Target.model_validate(
                        {"semantic": script.targets[name].semantic, "locators": [each]}
                    )
                    for name, each in locators.items()
                },
            }
        }
    )


def the_click(script: CompiledScript) -> StepAt:
    step = script.steps[0]
    assert isinstance(step, Click)
    return StepAt(step)


def healing_the_step(made: Made) -> dict[str, Any]:
    """What a run that healed its one step leaves: every check of a passing run
    holds, and `save` is bound anew."""
    taken = Repaired(
        the_click(made.script),
        MISSED,
        (),
        rebinding(made.script, save=NEW_SAVE),
        Basis(BOUND_AS, REPORTED, None, "heals/1"),
    )
    return {"repairs": (taken,), "base": BLOB}


def side_effect_left_unresolved(made: Made) -> dict[str, Any]:
    made.record.step_intent(2, CLICK, side_effect=True, target_used="save")
    return {"steps": (StepResult(2, "failed", error="the page crashed"),)}


# What makes each outcome hold, and nothing more decisive.
CAUSES: dict[str, Callable[[Made], dict[str, Any]]] = {
    "egress_blocked": lambda _: {"blocks": CDN},
    "non_resumable": side_effect_left_unresolved,
    "no_sandbox": lambda _: {
        "stops": (Stop("no_sandbox", NO_SECRETS.redact(SANDBOX)),)
    },
    "reset_unsafe": lambda _: {
        "stops": (Stop("reset_unsafe", NO_SECRETS.redact(RESET)),)
    },
    "unreachable": lambda _: {"infrastructure_events": (EVENT,)},
    "record_error": lambda _: {
        "stops": (Stop("record_error", NO_SECRETS.redact(UNWRITTEN)),)
    },
    "no_response": lambda _: {
        "stops": (Stop("no_response", NO_SECRETS.redact(SILENT)),)
    },
    "expectation_violated": lambda _: {
        "assertions": (AssertionResult("a2", "failed", locator_index=0),)
    },
    "inconclusive": lambda _: {
        "assertions": (AssertionResult("a3", "check_timed_out"),)
    },
    "drift_consistent": healing_the_step,
    "passed": lambda _: {},
}

# Each outcome with its exit code, most decisive first.
EXITS = [
    ("egress_blocked", 6),
    ("non_resumable", 4),
    ("no_sandbox", 10),
    ("reset_unsafe", 14),
    ("unreachable", 13),
    ("record_error", 15),
    ("no_response", 11),
    ("expectation_violated", 1),
    ("inconclusive", 3),
    ("drift_consistent", 7),
    ("passed", 0),
]


def merged(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Two causes' changes at once: what both add, both added."""
    both = first | second
    for added in ("steps", "assertions", "stops", "repairs"):
        both[added] = first.get(added, ()) + second.get(added, ())
    return both


@pytest.mark.parametrize(("cause", "code"), EXITS)
def test_each_outcome_alone_exits_with_its_code(
    tmp_path: Path, cause: str, code: int
) -> None:
    run = made(tmp_path)

    outcome: Outcome = final_outcome(ending(run, **CAUSES[cause](run)))

    assert (outcome, EXIT_CODES[outcome]) == (cause, code)


@pytest.mark.parametrize(
    ("first", "second"), list(combinations([cause for cause, _ in EXITS], 2))
)
def test_each_pair_of_outcomes_resolves_by_precedence(
    tmp_path: Path, first: str, second: str
) -> None:
    run = made(tmp_path)

    end = ending(run, **merged(CAUSES[first](run), CAUSES[second](run)))

    assert final_outcome(end) == first


def test_a_replay_that_returned_no_result_never_passes(tmp_path: Path) -> None:
    run = made(tmp_path)

    end = ReplayEnd(run.record, run.spec, run.script, None, CLEAN, ())

    assert final_outcome(end) == "inconclusive"


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        (["passed"], "passed"),
        (["passed", "inconclusive", "expectation_violated"], "expectation_violated"),
        (["inconclusive", "passed"], "inconclusive"),
        (["expectation_violated", "unreachable", "record_error"], "unreachable"),
        (["passed", "no_sandbox", "non_resumable", "egress_blocked"], "egress_blocked"),
        (["passed", "drift_consistent"], "drift_consistent"),
        (["drift_consistent", "inconclusive"], "inconclusive"),
        (["no_response", "expectation_violated"], "no_response"),
        (["record_error", "no_response"], "record_error"),
    ],
)
def test_runs_roll_up_by_the_same_precedence(
    outcomes: list[Outcome], expected: Outcome
) -> None:
    assert rolled_up(outcomes) == expected


def document(end: ReplayEnd) -> Any:
    """`end`'s document as a reader of its JSON gets it."""
    return json.loads(json.dumps(present(end)))


def judged(end: ReplayEnd, kind: str, name: object) -> dict[str, Any]:
    """The verdict `end`'s document gives the step, assertion or invariant
    `kind` called `name`."""
    verdicts: list[dict[str, Any]] = document(end)["verdicts"]
    return next(verdict for verdict in verdicts if verdict.get(kind) == name)


VIOLATED = (InvariantResult("js_exceptions", "violated", ("boom",), 1),)


@pytest.mark.parametrize(
    ("added", "kind", "name", "expected"),
    [
        ({}, "step", 1, ("pass", None)),
        (
            {
                "steps": (
                    StepResult(2, "drifted", misses=("no match", "not unanimous")),
                )
            },
            "step",
            2,
            (
                "inconclusive",
                "drift: strict replay doesn't heal (no match, not unanimous)",
            ),
        ),
        (
            {"steps": (StepResult(2, "failed", error="the page crashed"),)},
            "step",
            2,
            ("inconclusive", "the page crashed"),
        ),
        ({}, "assertion", "a1", ("pass", None)),
        (
            {"assertions": (AssertionResult("a2", "failed", locator_index=0),)},
            "assertion",
            "a2",
            ("expectation_violated", None),
        ),
        (
            {
                "assertions": (
                    AssertionResult("a2", "binding_unresolved", ("ambiguous",)),
                )
            },
            "assertion",
            "a2",
            ("inconclusive", "drift: strict replay doesn't heal (ambiguous)"),
        ),
        (
            {"assertions": (AssertionResult("a2", "binding_unresolved"),)},
            "assertion",
            "a2",
            ("inconclusive", "drift: strict replay doesn't heal"),
        ),
        (
            {"assertions": (AssertionResult("a2", "check_timed_out"),)},
            "assertion",
            "a2",
            ("inconclusive", "check_timed_out"),
        ),
        (
            {
                "assertions": (
                    AssertionResult("a2", "not_evaluated", error="the page stopped"),
                )
            },
            "assertion",
            "a2",
            ("inconclusive", "the page stopped"),
        ),
        ({}, "invariant", "console_errors", ("pass", None)),
        (
            {"invariants": VIOLATED},
            "invariant",
            "js_exceptions",
            ("expectation_violated", None),
        ),
    ],
)
def test_each_executor_result_gets_its_verdict(
    tmp_path: Path, added: dict[str, Any], kind: str, name: object, expected: object
) -> None:
    verdict = judged(ending(made(tmp_path), **added), kind, name)

    assert (verdict["verdict"], verdict["reason"]) == expected


def unevaluated(seq: int) -> tuple[AssertionResult, ...]:
    return tuple(
        AssertionResult(check, "not_evaluated", stopped_at=seq)
        for check in ("a1", "a2", "a3")
    )


@pytest.mark.parametrize(
    ("steps", "assertions", "events", "expected"),
    [
        # A step drifted, in a script with no check.
        (
            (StepResult(2, "drifted", misses=("no match",)),),
            (),
            (),
            [("pass", "step"), ("pass", "step"), ("inconclusive", "step")],
        ),
        # An infrastructure event after the last step, so no check was
        # evaluated.
        ((), unevaluated(1), (EVENT,), [("pass", "step"), ("pass", "step")]),
        # The same in a run with no check, which only the event shows stopped.
        ((), (), (EVENT,), [("pass", "step"), ("pass", "step")]),
    ],
    ids=["stopped-step", "interrupted", "interrupted-no-check"],
)
def test_what_never_ran_or_was_turned_off_gets_no_verdict(
    tmp_path: Path,
    steps: tuple[StepResult, ...],
    assertions: tuple[AssertionResult, ...],
    events: tuple[InfrastructureEvent, ...],
    expected: list[tuple[str, str]],
) -> None:
    end = ending(made(tmp_path), steps=steps, infrastructure_events=events)
    assert end.result is not None
    # What a stopped run held is no more established than an unevaluated
    # check; what it violated is.
    observed = (
        HELD[0],
        *VIOLATED,
        HELD[2],
        InvariantResult("broken_images", "disabled", (), 0),
    )
    end = replace(
        end, result=replace(end.result, assertions=assertions, invariants=observed)
    )

    verdicts = document(end)["verdicts"]

    assert [(verdict["verdict"], *list(verdict)[:1]) for verdict in verdicts] == [
        *expected,
        ("expectation_violated", "invariant"),
    ]


def test_an_egress_blocked_run_has_no_verdict_and_no_finding(tmp_path: Path) -> None:
    blocked = ending(
        made(tmp_path),
        blocks=CDN,
        assertions=(AssertionResult("a2", "failed", locator_index=0),),
        invariants=VIOLATED,
    )

    presented = document(blocked)

    keys = ("outcome", "exit_code", "error_code", "reasons", "verdicts")
    assert tuple(presented[key] for key in keys) == (
        "egress_blocked",
        6,
        "egress_blocked",
        [],
        [],
    )


def violated(name: InvariantName) -> dict[str, Any]:
    return {"invariants": (InvariantResult(name, "violated", ("seen",), 1),)}


@pytest.mark.parametrize(
    ("added", "kind", "name", "about"),
    [
        (
            {"assertions": (AssertionResult("a2", "failed", locator_index=0),)},
            "assertion",
            "a2",
            ("functional", "The form is saved"),
        ),
        (
            {"assertions": (AssertionResult("a4", "failed", locator_index=0),)},
            "assertion",
            "a4",
            ("visual", "The save button shows"),
        ),
        (
            violated("console_errors"),
            "invariant",
            "console_errors",
            ("js_error", "invariant console_errors violated"),
        ),
        (
            violated("js_exceptions"),
            "invariant",
            "js_exceptions",
            ("js_error", "invariant js_exceptions violated"),
        ),
        (
            violated("http_5xx"),
            "invariant",
            "http_5xx",
            ("backend_5xx", "invariant http_5xx violated"),
        ),
        (
            violated("broken_images"),
            "invariant",
            "broken_images",
            ("visual", "invariant broken_images violated"),
        ),
    ],
)
def test_each_violation_carries_a_finding_of_its_category(
    tmp_path: Path,
    added: dict[str, Any],
    kind: str,
    name: str,
    about: tuple[str, str],
) -> None:
    verdict = judged(ending(made(tmp_path), **added), kind, name)

    category, title = about
    assert verdict["finding"] == {
        "category": category,
        "title": title,
        "repro_steps": [0, 1],
    }


def evidence(seq: int) -> dict[str, object]:
    return {"path": f"evidence/{seq}", "screenshot": None}


def test_the_document_holds_each_verdict_with_its_evidence(tmp_path: Path) -> None:
    run = made(tmp_path)
    end = ending(
        run,
        assertions=(
            AssertionResult("a2", "failed", locator_index=1),
            AssertionResult("a3", "check_timed_out"),
            AssertionResult("a4", "binding_unresolved", ("no match",)),
        ),
        invariants=(
            HELD[0],
            InvariantResult("js_exceptions", "violated", ("boom",), 1),
            HELD[2],
            InvariantResult("broken_images", "disabled", (), 0),
        ),
        # The page where the invariant was first violated.
        evidence=Evidence(0, None),
    )

    presented = document(end)

    assert presented.pop("run_id") == run.record.run_id
    assert presented == {
        "schema_version": 1,
        "spec_id": "replay",
        "mode": "strict",
        "outcome": "expectation_violated",
        "exit_code": 1,
        "withheld": False,
        "error_code": None,
        "reasons": [
            "a2: expectation_violated",
            "a3: inconclusive, check_timed_out",
            "a4: inconclusive, drift: strict replay doesn't heal (no match)",
            "invariant js_exceptions: expectation_violated",
        ],
        "non_resumable": [],
        "infrastructure_events": {"count": 0, "events": []},
        "verdicts": [
            {
                "step": 0,
                "verdict": "pass",
                "reason": None,
                "locator_index": None,
                "misses": [],
                "evidence": evidence(0),
                "finding": None,
            },
            {
                "step": 1,
                "verdict": "pass",
                "reason": None,
                "locator_index": 0,
                "misses": [],
                "evidence": evidence(1),
                "finding": None,
            },
            {
                "assertion": "a1",
                "verdict": "pass",
                "reason": None,
                "locator_index": None,
                "misses": [],
                "evidence": evidence(1),
                "finding": None,
            },
            {
                "assertion": "a2",
                "verdict": "expectation_violated",
                "reason": None,
                "locator_index": 1,
                "misses": [],
                "evidence": evidence(1),
                "finding": {
                    "category": "functional",
                    "title": "The form is saved",
                    "repro_steps": [0, 1],
                },
            },
            {
                "assertion": "a3",
                "verdict": "inconclusive",
                "reason": "check_timed_out",
                "locator_index": None,
                "misses": [],
                "evidence": evidence(1),
                "finding": None,
            },
            {
                "assertion": "a4",
                "verdict": "inconclusive",
                "reason": "drift: strict replay doesn't heal (no match)",
                "locator_index": None,
                "misses": ["no match"],
                "evidence": evidence(1),
                "finding": None,
            },
            {
                "invariant": "console_errors",
                "verdict": "pass",
                "reason": None,
                "evidence": evidence(1),
                "finding": None,
            },
            {
                "invariant": "js_exceptions",
                "verdict": "expectation_violated",
                "reason": None,
                "evidence": evidence(0),
                "finding": {
                    "category": "js_error",
                    "title": "invariant js_exceptions violated",
                    "repro_steps": [0, 1],
                },
            },
            {
                "invariant": "http_5xx",
                "verdict": "pass",
                "reason": None,
                "evidence": evidence(1),
                "finding": None,
            },
        ],
    }


@pytest.mark.parametrize(
    ("redactor", "events"),
    [
        (
            NO_SECRETS,
            [
                {
                    "host": "api.evil.test",
                    "port": 8443,
                    "cause": "the name didn't resolve",
                }
            ],
        ),
        (BOUND, None),
    ],
)
def test_infrastructure_events_are_named_only_in_a_run_that_binds_no_secret(
    tmp_path: Path, redactor: Redactor, events: object
) -> None:
    end = ending(made(tmp_path, redactor), infrastructure_events=(EVENT,))

    presented = document(end)

    assert (presented["outcome"], presented["exit_code"]) == ("unreachable", 13)
    # Every verdict passed, so the event alone says why, in fixed words.
    assert presented["reasons"] == [
        "an allowed host or the egress proxy couldn't be reached"
    ]
    assert presented["withheld"] == (redactor is BOUND)
    assert presented["infrastructure_events"] == {"count": 1, "events": events}


def test_every_reason_and_title_is_scanned_again_with_the_runs_scan(
    tmp_path: Path,
) -> None:
    # Scanned by another scan than the run's, so each still spells the value.
    leak = NO_SECRETS.redact("fake-bound-value was in it")
    end = ending(
        made(tmp_path, BOUND, second=leak),
        steps=(StepResult(2, "failed", error=leak),),
        assertions=(AssertionResult("a4", "failed", locator_index=0),),
        stops=(Stop("record_error", leak),),
    )

    presented = document(end)

    assert presented["reasons"] == [
        "[SECRET:TEST_PASSWORD] was in it",
        "step 2: inconclusive, [SECRET:TEST_PASSWORD] was in it",
        "a4: expectation_violated",
    ]
    assert judged(end, "step", 2)["reason"] == "[SECRET:TEST_PASSWORD] was in it"
    assert judged(end, "assertion", "a4")["finding"]["title"] == (
        "[SECRET:TEST_PASSWORD] was in it"
    )


def test_an_unresolved_read_only_intent_leaves_the_run_resumable(
    tmp_path: Path,
) -> None:
    run = made(tmp_path)
    run.record.step_intent(2, CLICK, side_effect=False, target_used="save")

    end = ending(run, steps=(StepResult(2, "failed", error="the page crashed"),))

    presented = document(end)
    assert (presented["outcome"], presented["non_resumable"]) == ("inconclusive", [])


def test_an_egress_block_decides_though_replay_raised_after_it(tmp_path: Path) -> None:
    run = made(tmp_path)
    side_effect_left_unresolved(run)
    stops = (Stop("record_error", NO_SECRETS.redact(UNWRITTEN)),)

    raised = ReplayEnd(run.record, run.spec, run.script, None, CDN, (), stops)

    keys = ("outcome", "exit_code", "error_code", "reasons", "non_resumable")
    assert tuple(document(raised)[key] for key in keys) == (
        "egress_blocked",
        6,
        "egress_blocked",
        [UNWRITTEN],
        [2],
    )


def test_a_value_only_a_reasons_join_spells_is_scanned_out(tmp_path: Path) -> None:
    joined = Redactor([fake_secret("TEST_PASSWORD", "inconclusive, fake", APP)])
    end = ending(
        made(tmp_path, joined),
        steps=(StepResult(2, "failed", error="fake page crashed"),),
    )

    presented = document(end)

    assert presented["reasons"] == ["step 2: [SECRET:TEST_PASSWORD] page crashed"]


def test_a_replay_that_raised_still_records_its_outcome_reasons_and_non_resumable_steps(
    tmp_path: Path,
) -> None:
    run = made(tmp_path)
    run.record.step_intent(0, NAVIGATE, side_effect=False, target_used=None)
    run.record.step_completed(0, locator_used=None, settled="idle")
    run.record.step_intent(1, CLICK, side_effect=True, target_used="save")
    stops = (Stop("record_error", NO_SECRETS.redact(UNWRITTEN)),)

    raised = ReplayEnd(run.record, run.spec, run.script, None, CLEAN, (), stops)

    presented = document(raised)

    assert presented.pop("run_id") == run.record.run_id
    assert presented == {
        "schema_version": 1,
        "spec_id": "replay",
        "mode": "strict",
        "outcome": "non_resumable",
        "exit_code": 4,
        "withheld": False,
        "error_code": None,
        "reasons": ["the run record couldn't be written"],
        "non_resumable": [1],
        "infrastructure_events": {"count": 0, "events": []},
        "verdicts": [],
    }


def passing_with(run: Made, **changes: Any) -> ReplayEnd:
    """A run whose a1 and a4 pass, a4 reading `save`, with `changes` over it."""
    a4 = AssertionResult("a4", "pass", locator_index=0)
    return ending(run, assertions=(a4,), **changes)


def cast_result(end: ReplayEnd) -> RunResult:
    assert end.result is not None
    return end.result


HEAL_CITED = {
    "repair": "repairs/1.json",
    "record": "heals/1",
    "bound": 'button "Save"',
    "model_reported": {"confidence": 0.9, "rationale": "it moved"},
}
# The patch healing `save` leaves, and the proposal it makes.
PATCH = [
    {
        "op": "replace",
        "path": "/targets/save/locators",
        "value": [{"role": "button", "name": "Save"}],
    }
]


def test_a_healed_step_is_drift_consistent_citing_its_heal(tmp_path: Path) -> None:
    run = made(tmp_path)

    end = passing_with(run, **healing_the_step(run))
    written = document(end)

    assert (written["mode"], written["outcome"], written["exit_code"]) == (
        "run",
        "drift_consistent",
        7,
    )
    assert judged(end, "step", 1) == {
        "step": 1,
        "locator_index": 0,
        "misses": [],
        "verdict": "drift_consistent",
        "reason": None,
        "evidence": {"path": "evidence/1", "screenshot": None},
        "finding": None,
        "heal": HEAL_CITED,
    }
    assert judged(end, "assertion", "a4")["verdict"] == "drift_consistent"
    assert judged(end, "assertion", "a4")["heal"] == HEAL_CITED
    assert written["reasons"] == []


def test_a_subject_off_the_rebound_targets_passes_with_no_heal(tmp_path: Path) -> None:
    run = made(tmp_path)

    end = passing_with(run, **healing_the_step(run))

    assert judged(end, "step", 0)["verdict"] == "pass"
    assert judged(end, "step", 0)["heal"] is None
    assert judged(end, "assertion", "a1")["verdict"] == "pass"
    assert judged(end, "assertion", "a1")["heal"] is None
    assert judged(end, "invariant", "http_5xx")["verdict"] == "pass"
    assert judged(end, "invariant", "http_5xx")["heal"] is None


def test_a_healed_run_proposes_its_patch_on_the_files_blob(tmp_path: Path) -> None:
    run = made(tmp_path)

    written = document(passing_with(run, **healing_the_step(run)))

    proposal = written["heal_proposal"]
    assert proposal.pop("id").startswith("ah_")
    assert proposal == {
        "compiled": f".compiled/{run.spec.frontmatter.id}.json",
        "base": {"blob_sha": BLOB, "pr_head_sha": None},
        "patch": PATCH,
    }


def test_a_strict_document_has_no_heal_field(tmp_path: Path) -> None:
    written = document(ending(made(tmp_path)))

    assert written["mode"] == "strict"
    assert "heal_proposal" not in written
    assert all("heal" not in verdict for verdict in written["verdicts"])


def test_a_rebound_step_in_a_run_that_didnt_pass_is_inconclusive_and_proposes_nothing(
    tmp_path: Path,
) -> None:
    run = made(tmp_path)
    a4 = AssertionResult("a4", "pass", locator_index=0)
    failed = AssertionResult("a2", "failed", locator_index=0)

    end = ending(run, assertions=(a4, failed), **healing_the_step(run))
    written = document(end)

    assert written["outcome"] == "expectation_violated"
    assert written["heal_proposal"] is None
    assert judged(end, "step", 1)["verdict"] == "inconclusive"
    assert judged(end, "step", 1)["reason"] == (
        "heal rebound its target, but the run didn't pass"
    )
    assert judged(end, "step", 1)["heal"] == HEAL_CITED
    # A rebound assertion keeps its own deterministic verdict.
    assert judged(end, "assertion", "a4")["verdict"] == "pass"
    assert judged(end, "assertion", "a4")["heal"] == HEAL_CITED
    assert judged(end, "assertion", "a2")["verdict"] == "expectation_violated"
    assert judged(end, "assertion", "a2")["heal"] is None


def test_an_assertion_on_a_rebound_target_that_fails_is_a_violation_citing_the_heal(
    tmp_path: Path,
) -> None:
    run = made(tmp_path)
    failed = AssertionResult("a4", "failed", locator_index=0)

    end = ending(run, assertions=(failed,), **healing_the_step(run))

    entry = judged(end, "assertion", "a4")
    assert entry["verdict"] == "expectation_violated"
    assert entry["finding"]["category"] == "visual"
    assert entry["heal"] == HEAL_CITED
    assert document(end)["exit_code"] == 1


def repaired_step(run: Made, **changes: Any) -> Repaired:
    fields: dict[str, Any] = {
        "at": the_click(run.script),
        "misses": MISSED,
        "problems": (),
        "script": None,
        "basis": None,
    }
    return Repaired(**(fields | changes))


GAVE_UP = Basis(NO_SECRETS.redact("the healer found no element"), None, None, "heals/1")


@pytest.mark.parametrize(
    ("changes", "reason", "record"),
    [
        pytest.param(
            {"basis": GAVE_UP},
            "drift: heal gave up: the healer found no element (no match)",
            "heals/1",
            id="gave_up",
        ),
        pytest.param(
            {"problems": (NO_SECRETS.redact("/targets/save: refused"),)},
            "drift: replay refused the heal's repair: /targets/save: refused (no match)",
            None,
            id="refused",
        ),
    ],
)
def test_a_drift_whose_heal_came_to_nothing_is_inconclusive_and_cites_it(
    tmp_path: Path, changes: dict[str, Any], reason: str, record: str | None
) -> None:
    run = made(tmp_path)
    drifted = StepResult(1, "drifted", misses=MISSED)

    end = replace(
        ending(run, base=BLOB, repairs=(repaired_step(run, **changes),)),
        result=replace(
            cast_result(ending(run)),
            steps=(StepResult(0, "completed"), drifted),
            assertions=(),
            repairs=(repaired_step(run, **changes),),
        ),
    )

    entry = judged(end, "step", 1)
    assert (entry["verdict"], entry["reason"]) == ("inconclusive", reason)
    assert entry["heal"] == {
        "repair": "repairs/1.json",
        "record": record,
        "bound": None,
        "model_reported": None,
    }
    assert document(end)["heal_proposal"] is None


def test_a_step_that_drifted_again_after_its_repair_says_so(tmp_path: Path) -> None:
    run = made(tmp_path)
    taken = repaired_step(
        run,
        script=rebinding(run.script, save=NEW_SAVE),
        basis=Basis(BOUND_AS, REPORTED, None, "heals/1"),
    )

    end = replace(
        ending(run, base=BLOB, repairs=(taken,)),
        result=replace(
            cast_result(ending(run)),
            steps=(StepResult(0, "completed"), StepResult(1, "drifted", misses=MISSED)),
            assertions=(),
            repairs=(taken,),
        ),
    )

    entry = judged(end, "step", 1)
    assert entry["reason"] == "drift: drifted again after its repair (no match)"
    assert entry["heal"] == HEAL_CITED | {"bound": None, "model_reported": None}


def test_a_drift_no_repair_was_handed_says_so(tmp_path: Path) -> None:
    run = made(tmp_path)
    drifted = AssertionResult("a4", "binding_unresolved", MISSED)

    end = ending(run, assertions=(drifted,), base=BLOB)

    entry = judged(end, "assertion", "a4")
    assert entry["verdict"] == "inconclusive"
    assert entry["reason"] == "drift: it went to no heal (no match)"
    assert entry["heal"] is None


def test_an_assertion_handed_to_the_heal_cites_its_own_repair(tmp_path: Path) -> None:
    run = made(tmp_path)
    check = run.script.assertions[3]
    assert isinstance(check, VisibleUnoccluded)
    repaired = Repaired(AssertionAt(3, check), MISSED, (), None, GAVE_UP)
    drifted = AssertionResult("a4", "binding_unresolved", MISSED)

    end = ending(run, assertions=(drifted,), base=BLOB, repairs=(repaired,))

    entry = judged(end, "assertion", "a4")
    assert entry["reason"] == (
        "drift: heal gave up: the healer found no element (no match)"
    )
    assert entry["heal"]["repair"] == "repairs/assertion-3.json"


def test_each_rebound_subject_cites_the_repair_that_first_changed_its_target(
    tmp_path: Path,
) -> None:
    run = made(tmp_path)
    first = rebinding(run.script, save=NEW_SAVE)
    later = rebinding(first, save={"css": "button.save"})
    check = run.script.assertions[3]
    assert isinstance(check, VisibleUnoccluded)
    repairs = (
        Repaired(
            the_click(run.script),
            MISSED,
            (),
            first,
            Basis(BOUND_AS, REPORTED, None, "heals/1"),
        ),
        Repaired(
            AssertionAt(3, check),
            MISSED,
            (),
            later,
            Basis(NO_SECRETS.redact("later"), REPORTED, None, "heals/assertion-3"),
        ),
    )

    end = passing_with(run, base=BLOB, repairs=repairs)
    written = document(end)

    assert judged(end, "assertion", "a4")["heal"]["repair"] == "repairs/1.json"
    assert written["heal_proposal"]["patch"][0]["value"] == [{"css": "button.save"}]


def test_a_final_script_with_the_committed_locators_passes_and_proposes_nothing(
    tmp_path: Path,
) -> None:
    """A target that flipped back: both evaluations read the committed locators."""
    run = made(tmp_path)
    away = rebinding(run.script, save=NEW_SAVE)
    check = run.script.assertions[3]
    assert isinstance(check, VisibleUnoccluded)
    repairs = (
        Repaired(AssertionAt(3, check), MISSED, (), away, None),
        Repaired(AssertionAt(3, check), MISSED, (), run.script, None),
    )

    end = passing_with(run, base=BLOB, repairs=repairs)
    written = document(end)

    assert (written["outcome"], written["exit_code"]) == ("passed", 0)
    assert written["heal_proposal"] is None
    assert judged(end, "assertion", "a4")["verdict"] == "pass"


def test_a_run_that_cant_write_its_record_says_what_mode_it_was(tmp_path: Path) -> None:
    spec = made(tmp_path).spec

    assert unrecorded(spec, NO_SECRETS, "no record")["mode"] == "strict"
    assert "heal_proposal" not in unrecorded(spec, NO_SECRETS, "no record")
    healing = unrecorded(spec, NO_SECRETS, "no record", healing=True)
    assert (healing["mode"], healing["heal_proposal"]) == ("run", None)


def test_a_no_response_stop_keeps_the_verdicts(tmp_path: Path) -> None:
    run = made(tmp_path)
    stop = Stop("no_response", NO_SECRETS.redact(SILENT))

    written = document(ending(run, stops=(stop,), base=BLOB))

    assert (written["outcome"], written["exit_code"]) == ("no_response", 11)
    assert written["reasons"] == [SILENT]
    assert written["verdicts"] != []


def test_a_run_that_heals_and_raised_proposes_nothing(tmp_path: Path) -> None:
    run = made(tmp_path)

    raised = ReplayEnd(run.record, run.spec, run.script, None, CLEAN, (), (), BLOB)

    written = document(raised)
    assert (written["mode"], written["heal_proposal"]) == ("run", None)
    assert written["outcome"] == "inconclusive"
