"""The gate before a step acts through a target a repair rebound (#244;
ADR-0024's #244 amendment): it reads the element the binding resolved to, live,
so that a test secret isn't typed into the wrong element and a repair doesn't
act on an element the committed locators can't vouch for. A refusal, or a look
the page doesn't answer, fails the step with no intent. Real Chromium through
`replays.replay_run`, with scripted repairs."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from aqa_core.compiled import Click, CompiledScript, Target
from aqa_runner import executor
from aqa_runner.bound_secrets import BoundSecret
from aqa_runner.browser_session import BrowserSession
from aqa_runner.evaluation import Evaluation, look_for
from aqa_runner.locators import Resolved, Unresolved
from aqa_runner.repairs import Drift
from playwright.async_api import ElementHandle

from packages.runner.tests import executor_fixtures as fx
from packages.runner.tests.drift_fixtures import DETAILS
from packages.runner.tests.secret_fixtures import FAKE_VALUE
from packages.runner.tests.test_replay_repairs import (
    MOVED,
    Repairing,
    outcomes,
    rebound,
    repairing,
    secret_spec_at,
    shipping_spec,
    states,
)

RAN_ONLY_THE_START = [(0, "intent"), (0, "completed")]
RAN_THE_STEP = [*RAN_ONLY_THE_START, (1, "intent"), (1, "completed")]
BASIS = "network: POST /write/place"
# What a page does when a decoy is clicked: a write the run must never cause.
WRITES = "fetch('/write/decoy', {method: 'POST'})"
# A busy loop that starts once the read that schedules it has answered.
GOES_BUSY = "() => { setTimeout(() => { for (;;) {} }, 0); }"

OTHER_ROLE = (
    "a repair rebound this step's target to an element without its committed "
    "role, so the step did not run"
)
TAKES_NO_TEXT = (
    "a repair rebound this step's target to an element that takes no text, so "
    "the step did not run"
)
NOT_A_SELECT = (
    "a repair rebound this step's target to an element that is no select "
    "control, so the step did not run"
)
UNANCHORED = (
    "a repair rebound this step's target to an element no committed locator "
    "finds alone, so the step did not run"
)
BROKE = "the page broke the checks on this step's rebound target, so it did not run"
UNANSWERED = "the page didn't answer within 2 s"
FIELD_REFUSED = (
    'fill_secret refused TEST_PASSWORD: the field isn\'t an <input type="password">'
)


@pytest.fixture(autouse=True)
def secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """The value `secret_spec_at`'s spec binds; other specs bind nothing."""
    monkeypatch.setenv("AQA_SECRET_TEST_PASSWORD", FAKE_VALUE)


def with_page_part(html: str) -> str:
    """The moved page, with `html` before its footer."""
    return MOVED.replace("<footer>", f"{html}<footer>")


def bound_to(name: str, *locators: dict[str, Any]) -> Callable[[Drift], CompiledScript]:
    """A repair that offers the script it was handed with the target `name`
    found by `locators`."""

    def offer(drift: Drift) -> CompiledScript:
        was = drift.script.targets[name].model_dump(mode="json", exclude_none=True)
        target = Target.model_validate({**was, "locators": locators})
        return drift.script.model_copy(
            update={"targets": {**drift.script.targets, name: target}}
        )

    return offer


def one_step(step: dict[str, Any], **target: Any) -> CompiledScript:
    """A script of `step`, seq 1, through a target called `field`, whose fields
    are `target`'s."""
    return fx.compiled(
        [{"seq": 1, "target": "field", "side_effect": False, **step}],
        targets={"field": {"semantic": "the order's field", **target}},
    )


def assert_not_dispatched(run: Repairing, reason: str) -> None:
    """The step failed with `reason`, and no intent says it ever ran."""
    assert outcomes(run) == [(0, "completed", ()), (1, "failed", ())]
    assert run.result.steps[1].error == reason
    assert states(run) == RAN_ONLY_THE_START


async def making_the_button_a_link(drift: Drift) -> CompiledScript:
    """A repair that rebinds the toggle to the moved button, by a locator that
    doesn't read its role, and the page making it a link meanwhile."""
    await drift.session.page.evaluate(
        "document.querySelector('#order button').setAttribute('role', 'link')"
    )
    return bound_to("toggle", BY_THE_ORDER)(drift)


@pytest.mark.parametrize(
    ("html", "offer"),
    [
        pytest.param(
            with_page_part(
                f'<a class="decoy" href="#order" onclick="{WRITES}">Show details</a>'
            ),
            bound_to("toggle", {"css": "a.decoy"}),
            id="another_element",
        ),
        pytest.param(MOVED, making_the_button_a_link, id="changed_since_the_offer"),
    ],
)
def test_a_rebound_element_without_the_committed_role_is_not_dispatched(
    tmp_path: Path, html: str, offer: Callable[[Drift], Any]
) -> None:
    run = repairing(tmp_path, offer, html=html)

    assert_not_dispatched(run, OTHER_ROLE)
    assert ("POST", "/write/decoy") not in run.seen


BY_THE_ORDER = {"css": "button", "scope": {"css": "#order"}}
# A field no locator finds on the moved page, with no role locator.
NOTE = {"locators": [{"css": "input.note"}]}
FILL = {"action": "fill", "value": "x"}
FILL_SECRET = {"action": "fill_secret", "secret": "TEST_PASSWORD"}


@pytest.mark.parametrize(
    ("step", "spec_at", "html", "locator"),
    [
        pytest.param(
            FILL,
            shipping_spec,
            MOVED,
            BY_THE_ORDER,
            id="fill_a_button",
        ),
        pytest.param(
            FILL,
            shipping_spec,
            with_page_part('<input id="note" readonly>'),
            {"css": "#note"},
            id="fill_a_read_only_field",
        ),
        pytest.param(
            FILL_SECRET,
            secret_spec_at,
            with_page_part('<input id="note" type="password" readonly>'),
            {"css": "#note"},
            id="fill_secret_a_read_only_field",
        ),
    ],
)
def test_a_rebound_field_that_takes_no_text_is_not_dispatched(
    tmp_path: Path,
    step: dict[str, Any],
    spec_at: Callable[[Path], Path],
    html: str,
    locator: dict[str, Any],
) -> None:
    run = repairing(
        tmp_path,
        bound_to("field", locator),
        script=one_step(step, **NOTE),
        spec_at=spec_at,
        html=html,
    )

    assert_not_dispatched(run, TAKES_NO_TEXT)


SELECT = {"action": "select", "option": "M"}
SIZE = {"locators": [{"css": "select.size"}]}


def test_a_rebound_element_that_is_no_select_control_is_not_dispatched(
    tmp_path: Path,
) -> None:
    run = repairing(
        tmp_path,
        bound_to("field", {"css": "#size"}),
        script=one_step(SELECT, **SIZE),
        html=with_page_part('<div id="size" role="combobox" tabindex="0">M</div>'),
    )

    assert_not_dispatched(run, NOT_A_SELECT)


# "Place order" gone from the toolbar, and "Delete order" in the footer.
PLACE_ORDER = {
    "semantic": "the button that places the order",
    "locators": [
        fx.by_role("button", "Place order"),
        {"css": "button", "scope": {"css": "#toolbar"}},
    ],
}
DELETE_IN_THE_FOOTER = MOVED.replace(
    "<button>Contact us</button>",
    '<button class="delete" '
    "onclick=\"fetch('/write/delete', {method: 'POST'})\">Delete order</button>",
)


def test_a_side_effect_step_rebound_to_an_unanchored_element_is_not_dispatched(
    tmp_path: Path,
) -> None:
    script = fx.compiled(
        [
            {
                "seq": 1,
                "action": "click",
                "target": "place",
                "side_effect": True,
                "side_effect_basis": BASIS,
            }
        ],
        targets={"place": PLACE_ORDER},
    )

    run = repairing(
        tmp_path,
        bound_to("place", {"css": "button.delete", "scope": {"css": "footer"}}),
        script=script,
        html=DELETE_IN_THE_FOOTER,
    )

    assert ("POST", "/write/delete") not in run.seen
    assert_not_dispatched(run, UNANCHORED)


def test_a_fill_secret_rebound_to_an_unanchored_field_is_not_dispatched(
    tmp_path: Path,
) -> None:
    run = repairing(
        tmp_path,
        bound_to("field", {"css": "#other"}),
        script=one_step(FILL_SECRET, **NOTE),
        spec_at=secret_spec_at,
        html=with_page_part('<input id="other" type="password">'),
    )

    assert_not_dispatched(run, UNANCHORED)


# A login field: its label is gone, and its css locator still finds what the
# form holds now, alone, so the anchor passes.
LOGIN = {
    "locators": [{"label": "Password"}, {"css": "input", "scope": {"css": "#login"}}]
}
BY_THE_FORM = {"css": "input", "scope": {"css": "#login"}}


def test_a_fill_secret_rebound_to_a_field_its_binding_refuses_writes_no_intent(
    tmp_path: Path,
) -> None:
    run = repairing(
        tmp_path,
        bound_to("field", BY_THE_FORM),
        script=one_step(
            {**FILL_SECRET, "side_effect": True, "side_effect_basis": BASIS}, **LOGIN
        ),
        spec_at=secret_spec_at,
        html=with_page_part('<form id="login"><input aria-label="Note"></form>'),
    )

    assert_not_dispatched(run, FIELD_REFUSED)


SIDE_EFFECT_TOGGLE = DETAILS.model_copy(
    update={
        "steps": (
            Click(
                seq=1,
                action="click",
                target="toggle",
                side_effect=True,
                side_effect_basis=BASIS,
            ),
        )
    }
)


@pytest.mark.parametrize(
    ("offer", "script", "spec_at", "html"),
    [
        pytest.param(
            lambda drift: rebound(drift.script),
            SIDE_EFFECT_TOGGLE,
            shipping_spec,
            MOVED,
            id="side_effect_anchored",
        ),
        pytest.param(
            bound_to("field", {"css": "#note"}),
            one_step(FILL, **NOTE),
            shipping_spec,
            with_page_part('<input id="note">'),
            id="fill",
        ),
        pytest.param(
            bound_to("field", {"css": "#size"}),
            one_step(SELECT, **SIZE),
            shipping_spec,
            with_page_part('<select id="size"><option>M</option></select>'),
            id="select",
        ),
        pytest.param(
            bound_to("field", BY_THE_FORM),
            one_step(FILL_SECRET, **LOGIN),
            secret_spec_at,
            with_page_part('<form id="login"><input type="password"></form>'),
            id="fill_secret",
        ),
    ],
)
def test_a_rebound_step_the_gate_admits_runs(
    tmp_path: Path,
    offer: Callable[[Drift], CompiledScript],
    script: CompiledScript,
    spec_at: Callable[[Path], Path],
    html: str,
) -> None:
    run = repairing(tmp_path, offer, script=script, spec_at=spec_at, html=html)

    assert outcomes(run) == [(0, "completed", ()), (1, "completed", ())]
    assert states(run) == RAN_THE_STEP


def after_a_rebound_target_resolves(
    monkeypatch: pytest.MonkeyPatch,
    act: Callable[[BrowserSession], Awaitable[object]],
) -> None:
    """Make `act` run right after a target resolves, once a look has missed:
    between the re-run's resolution and its gate."""
    missed: list[object] = []

    async def looking(
        session: BrowserSession, evaluation: Evaluation, name: str, use: Any
    ) -> Resolved | Unresolved:
        found: Resolved | Unresolved = await look_for(session, evaluation, name, use)
        if isinstance(found, Unresolved):
            missed.append(found)
        elif missed:
            await act(session)
        return found

    monkeypatch.setattr(executor, "look_for", looking)


def test_a_rebound_step_whose_checks_the_page_never_answers_fails_with_no_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    after_a_rebound_target_resolves(
        monkeypatch, lambda session: session.page.evaluate(GOES_BUSY)
    )

    run = repairing(tmp_path, lambda drift: rebound(drift.script))

    assert_not_dispatched(run, UNANSWERED)


def test_a_fill_secret_whose_binding_check_the_page_never_answers_fails_with_no_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = BrowserSession.secret_refusal

    async def going_busy(
        session: BrowserSession, element: ElementHandle, bound: BoundSecret
    ) -> None:
        await session.page.evaluate(GOES_BUSY)
        await original(session, element, bound)

    monkeypatch.setattr(BrowserSession, "secret_refusal", going_busy)

    run = repairing(
        tmp_path,
        bound_to("field", BY_THE_FORM),
        script=one_step(FILL_SECRET, **LOGIN),
        spec_at=secret_spec_at,
        html=with_page_part('<form id="login"><input type="password"></form>'),
    )

    assert_not_dispatched(run, UNANSWERED)


def test_a_rebound_step_whose_page_closes_under_its_checks_fails_with_no_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    after_a_rebound_target_resolves(monkeypatch, lambda session: session.page.close())

    run = repairing(tmp_path, lambda drift: rebound(drift.script))

    assert_not_dispatched(run, BROKE)


def test_a_step_through_a_target_no_repair_rebound_goes_through_no_gate(
    tmp_path: Path,
) -> None:
    # A fill the button refuses, found as committed: the gate would refuse it
    # before its intent, so the intent says the gate wasn't in the way.
    run = repairing(
        tmp_path,
        lambda drift: drift.script,
        script=one_step(FILL, locators=[BY_THE_ORDER]),
    )

    assert run.handed == []
    assert outcomes(run) == [(0, "completed", ()), (1, "failed", ())]
    assert states(run) == [*RAN_ONLY_THE_START, (1, "intent")]
