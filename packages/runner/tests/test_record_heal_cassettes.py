"""The recorder's heal cases (#244; ADR-0024's #244 amendment): each drift page's
`aqa run` recorded in real Chromium from a local stand-in for the provider, with
a fake key, within the heal round's own bound, and promoted only when it ended
in the proposal its case expects. Role, mode and allowance come from the case's
row in the recorder's table. No test here reaches a provider."""

import json
from collections.abc import Callable, Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml
from langchain_core.messages import HumanMessage

from packages.runner.tests import heal_fixtures
from packages.runner.tests.explore_graph_fixtures import ref
from packages.runner.tests.provider_stand_in import (
    Answer,
    Answering,
    Start,
    answer,
    called,
    standing_in,
)
from packages.runner.tests.record_cassettes import (
    CASES,
    HEAL_ROUND_USD,
    ROUND_USD,
    record,
)

pytestmark = pytest.mark.usefixtures("reset_tracing")

NAMES = ["heal_moved", "heal_renamed", "heal_moved_assertion", "heal_renamed_assertion"]
# Which button each case's model should pick.
PICKS = {
    "heal_moved": "Show details",
    "heal_renamed": "View details",
    "heal_moved_assertion": "Show details",
    "heal_renamed_assertion": "View details",
}


def snapshot(body: bytes) -> str:
    """The text of the last message a request sends: its blocks."""
    content = json.loads(body)["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


def picking(name: str, **usage: int) -> Callable[[bytes], Answer]:
    """The stand-in's answer: rebind the button called `name`, sure of it."""

    def answered(body: bytes) -> Answer:
        found = ref([HumanMessage(content=snapshot(body))], "button", name)
        args = {"ref": found, "confidence": 0.9, "rationale": "it fits the meaning"}
        return called(1, "rebind", args, **usage)

    return answered


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[Start]:
    with standing_in(monkeypatch) as start:
        yield start


@pytest.fixture
def library(tmp_path: Path) -> Path:
    path = tmp_path / "library"
    path.mkdir()
    return path


def receipt(attempt: Path) -> dict[str, Any]:
    kept: dict[str, Any] = json.loads((attempt / "receipt.json").read_text())
    return kept


@pytest.mark.parametrize("name", NAMES)
def test_a_heal_capture_that_ends_in_its_proposal_is_promoted(
    provider: Start, library: Path, tmp_path: Path, name: str
) -> None:
    attempt = tmp_path / "heal" / "001"
    provider([picking(PICKS[name])])

    costs = record(name, attempt, library=library)

    kept = receipt(attempt)
    assert (kept["outcome"], kept["explore"], kept["disallowed"]) == (
        "accepted",
        {"exit_code": 7, "outcome": "drift_consistent", "published": None},
        None,
    )
    assert (kept["spend"]["run_usd"], kept["spend"]["round_left_usd"]) == (
        "0.1",
        "0.25",
    )
    assert [(cost.role, cost.mode, cost.status) for cost in costs] == [
        ("healer", "heal", "ok")
    ]
    assert len(kept["responses"]) == 1
    promoted = (library / f"{name}.yaml").read_text()
    assert promoted == (attempt / f"{name}.yaml").read_text()
    assert len(yaml.safe_load(promoted)["interactions"]) == 1


NO_TOOL = answer(1, [{"type": "text", "text": "nothing fits"}], "end_turn")


@pytest.mark.parametrize(
    ("answering", "ended"),
    [
        pytest.param(
            NO_TOOL,
            (
                {"exit_code": 3, "outcome": "inconclusive", "published": None},
                "it ended inconclusive, which this case doesn't allow",
            ),
            id="gave_up",
        ),
        pytest.param(
            picking("Contact us"),
            (
                {"exit_code": 7, "outcome": "drift_consistent", "published": None},
                "its proposal binds another element than this case's",
            ),
            id="a_decoy_that_passes",
        ),
    ],
)
def test_a_heal_capture_that_ends_otherwise_is_never_promoted(
    provider: Start,
    library: Path,
    tmp_path: Path,
    answering: Answering,
    ended: tuple[dict[str, object], str],
) -> None:
    attempt = tmp_path / "heal" / "001"
    provider([answering])

    with pytest.raises(ValueError, match="disallowed"):
        record("heal_moved_assertion", attempt, library=library)

    assert not (library / "heal_moved_assertion.yaml").exists()
    kept = receipt(attempt)
    assert (kept["outcome"], kept["explore"], kept["disallowed"]) == (
        "disallowed",
        *ended,
    )


def test_a_response_past_its_one_turn_is_never_promoted(
    provider: Start, library: Path, tmp_path: Path
) -> None:
    # It took two responses to a proposal, where the case expects one.
    nonsense = called(1, "nonsense", {})
    attempt = tmp_path / "heal" / "001"
    provider([nonsense, picking("Show details")])

    with pytest.raises(ValueError, match="rejected"):
        record("heal_moved", attempt, library=library)

    assert not (library / "heal_moved.yaml").exists()
    kept = receipt(attempt)
    assert kept["outcome"] == "rejected"
    assert [cost["role"] for cost in kept["cost_records"]] == ["healer", "healer"]


def test_a_heal_round_past_its_limit_refuses_the_request(
    provider: Start, library: Path, tmp_path: Path
) -> None:
    earlier = tmp_path / "heal" / "000"
    earlier.mkdir(parents=True)
    spent = {"spend": {"spent_usd": str(HEAL_ROUND_USD), "overrun": False}}
    (earlier / "receipt.json").write_text(json.dumps(spent))
    attempt = tmp_path / "heal" / "001"
    stand_in = provider([picking("Show details")])

    with pytest.raises(ValueError, match="spend_refused"):
        record("heal_moved", attempt, library=library)

    assert stand_in.sent == []
    assert not (library / "heal_moved.yaml").exists()
    assert receipt(attempt)["spend"]["refused"] is True


def test_a_heal_case_is_priced_and_labelled_as_the_healers(
    provider: Start,
    library: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A navigator and a healer on different models and prices: a recorder that
    # read the navigator's would refuse the request as another model's.
    monkeypatch.setattr(
        heal_fixtures,
        "RECORDING_CONFIG",
        heal_fixtures.RECORDING_CONFIG
        + "roles: { navigator: { model: claude-haiku-4-5 }, "
        "healer: { model: claude-opus-5-5 } }\n",
    )
    attempt = tmp_path / "heal" / "001"
    stand_in = provider([picking("Show details", input_tokens=1000, output_tokens=100)])

    costs = record("heal_moved", attempt, library=library)

    assert json.loads(stand_in.sent[0])["model"] == "claude-opus-5-5"
    [cost] = costs
    assert (cost.role, cost.mode, cost.model) == ("healer", "heal", "claude-opus-5-5")
    intent = json.loads((attempt / "intent.json").read_text())
    assert (intent["role"], intent["model"]) == ("healer", "claude-opus-5-5")
    kept = receipt(attempt)["cost_records"]
    assert [each["role"] for each in kept] == ["healer"]
    assert Decimal(receipt(attempt)["spend"]["spent_usd"]) == cost.cost_usd


def test_each_case_says_what_it_is_recorded_as() -> None:
    heal = CASES["heal_moved"]
    explore = CASES["explore_fixture"]

    assert (heal.role, heal.mode, heal.round_usd) == ("healer", "heal", HEAL_ROUND_USD)
    assert (explore.role, explore.mode, explore.round_usd) == (
        "navigator",
        "explore",
        ROUND_USD,
    )
    assert heal.round_usd == Decimal("0.25")
